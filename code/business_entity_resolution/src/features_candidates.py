"""Track 4 Phase 2: features for the real blocking candidate pairs.

Reads the top-K candidate table ``blocking.py`` left in the shared DuckDB
warehouse (``train_scored`` = candidates for ALL train S1, ``test_scored`` =
ALL test S1), joins both sides' normalized fields from ``data/norm`` (no raw
text re-normalization), and streams batches through a worker pool running
``features.compute_normed_pair_features``. Adds the blocking/competition
features that only exist once real candidates do:

    n_keys_hit, cheap_score, rank_in_s1, n_candidates_for_s1   (blocking)
    n_competitors, rank_among_competitors                       (competition)
    label                                                       (train only)

Competition features count how many S1 records compete for the same S2/S3
record. They are computed over the full candidate population of each split --
all 2.2M train S1, all 1.73M test S1 -- so train and test see the same
distribution (a 300k train sample used to make these ~7x smaller in train
than in test).

Every train candidate is featurized, not just a training sample: the stage-2
model needs stage-1 scores for all of a record's competitors.

Usage: ``python features_candidates.py train`` or ``... test``.
"""

import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

from config import FEATURES_DIR, TRAIN_GT_LONG_PARQUET, norm_parquet_path
from db import connect
from features import (NAME_FIELDS, ADDR_FIELDS, NORMED_INPUT_COLUMNS, POOL_CHUNKSIZE,
                      WORKERS, compute_normed_pair_features, make_pool)
from perf import print_sysinfo, progress_line, stage
from resources import profile

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BLOCKING_COLS = ["n_keys_hit", "cheap_score", "rank_in_s1", "n_candidates_for_s1",
                 "n_competitors", "rank_among_competitors"]
FEATURE_PARQUET = {
    "train": FEATURES_DIR / "train_candidates_features.parquet",
    "test": FEATURES_DIR / "test_candidates_features.parquet",
}


NORM_COLS = NAME_FIELDS + ADDR_FIELDS
# Candidate pairs per chunk per GB of DuckDB memory: each chunk materializes
# its S1 rows and the S2/S3 rows its candidates reference (~1 KB each with all
# normalized string fields) before joining.
PAIRS_PER_CHUNK_PER_GB = 700_000


def build_competition_table(con, split):
    """``{split}_fc_comp``: blocking + competition features for every candidate, with an S1 chunk id."""
    avg = con.execute(f"SELECT count(*) / greatest(count(DISTINCT s1_id), 1) FROM {split}_scored").fetchone()[0]
    s1_per_chunk = max(1_000, int(profile().duckdb_light_gb * PAIRS_PER_CHUNK_PER_GB / max(avg, 1)))
    con.execute(f"""
        CREATE OR REPLACE TABLE {split}_fc_comp AS
        SELECT s1_id, match_id, match_source, n_keys_hit, cheap_score, rank_in_s1,
               count(*) OVER (PARTITION BY s1_id) AS n_candidates_for_s1,
               count(*) OVER (PARTITION BY match_id, match_source) AS n_competitors,
               row_number() OVER (PARTITION BY match_id, match_source
                                  ORDER BY cheap_score DESC, s1_id) AS rank_among_competitors,
               (dense_rank() OVER (ORDER BY s1_id) - 1) // {s1_per_chunk} AS chunk
        FROM {split}_scored
        ORDER BY s1_id
    """)
    return con.execute(f"SELECT max(chunk) + 1 FROM {split}_fc_comp").fetchone()[0] or 0


def chunk_pairs_sql(con, split, chunk, with_label):
    """Materialize one S1 chunk's candidates with both sides' normalized fields; return the SELECT."""
    s1_norm = norm_parquet_path(split, "S1").as_posix()
    con.execute(f"CREATE OR REPLACE TEMP TABLE _c AS SELECT * FROM {split}_fc_comp WHERE chunk = {chunk}")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE _a AS
                    SELECT id, country, {', '.join(NORM_COLS)} FROM '{s1_norm}'
                    WHERE id IN (SELECT DISTINCT s1_id FROM _c)""")
    parts = []
    for src in ("S2", "S3"):
        path = norm_parquet_path(split, src).as_posix()
        parts.append(f"""SELECT '{src}' AS source, id, country, {', '.join(NORM_COLS)} FROM '{path}'
                         WHERE id IN (SELECT DISTINCT match_id FROM _c WHERE match_source = '{src}')""")
    con.execute(f"CREATE OR REPLACE TEMP TABLE _b AS {' UNION ALL '.join(parts)}")
    label_col = ", (g.s1_id IS NOT NULL)::INTEGER AS label" if with_label else ""
    label_join = (f"LEFT JOIN '{TRAIN_GT_LONG_PARQUET.as_posix()}' g ON g.s1_id = c.s1_id "
                  f"AND g.match_id = c.match_id AND g.match_source = c.match_source") if with_label else ""
    s1_cols = ", ".join(f"a.{c} AS s1_{c}" for c in NORM_COLS)
    o_cols = ", ".join(f"b.{c} AS o_{c}" for c in NORM_COLS)
    return f"""
        SELECT c.s1_id, c.match_id, c.match_source AS other_source,
               a.country AS s1_country, b.country AS other_country, {s1_cols}, {o_cols},
               (b.addr_clean = '' AND b.numbers = '')::DOUBLE AS other_addr_empty,
               c.n_keys_hit, c.cheap_score, c.rank_in_s1, c.n_candidates_for_s1,
               c.n_competitors, c.rank_among_competitors {label_col}
        FROM _c c
        JOIN _a a ON a.id = c.s1_id
        JOIN _b b ON b.id = c.match_id AND b.source = c.match_source
        {label_join}
    """


def featurize_candidates(con, split, with_label):
    """Featurize every candidate of a split, S1 chunk by S1 chunk, into one parquet file."""
    with stage(f"competition features over all {split} candidates"):
        n_chunks = build_competition_table(con, split)
    n_total = con.execute(f"SELECT count(*) FROM {split}_fc_comp").fetchone()[0]
    out_path = FEATURE_PARQUET[split]
    batch_rows = max(100_000, WORKERS * POOL_CHUNKSIZE * 4)
    print(f"  {split}: {n_total} candidate pairs in {n_chunks} S1 chunks, {WORKERS} feature workers")

    extra = ["s1_id", "match_id", "s1_country"] + BLOCKING_COLS + (["label"] if with_label else [])
    cols = list(dict.fromkeys(NORMED_INPUT_COLUMNS + extra))

    writer = None
    n_seen = 0
    t0 = time.perf_counter()
    with make_pool(WORKERS) as pool:
        for chunk in range(n_chunks):
            sql = chunk_pairs_sql(con, split, chunk, with_label)
            result = con.execute(f"SELECT {', '.join(cols)} FROM ({sql})")
            for batch in result.to_arrow_reader(batch_rows):
                df = batch.to_pandas()
                feats = compute_normed_pair_features(df, pool)
                for c in extra:
                    feats[c] = df[c].values
                num_cols = [c for c in feats.columns
                            if feats[c].dtype.kind == "f" and c != "cheap_score"]
                feats[num_cols] = feats[num_cols].astype("float32")
                table = pa.Table.from_pandas(feats, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table.schema)
                elif table.schema != writer.schema:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                n_seen += len(df)
            print(f"  chunk {chunk + 1}/{n_chunks} -- {progress_line(split, n_seen, n_total, t0)}")
    if writer is not None:
        writer.close()
    for t in ("_a", "_b", "_c"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    print(f"  saved -> {out_path}")


def main():
    """Featurize all train candidates (labeled) or all test candidates."""
    which = sys.argv[1] if len(sys.argv) > 1 else "train"
    if which not in ("train", "test"):
        raise SystemExit(f"unknown arg {which!r}, expected 'train' or 'test'")
    print_sysinfo()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    con = connect(role="light")
    with stage(f"featurize {which} candidates"):
        featurize_candidates(con, which, with_label=(which == "train"))
    con.close()


if __name__ == "__main__":
    main()
