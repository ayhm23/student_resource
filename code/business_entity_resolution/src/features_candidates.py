"""Track 4 Phase 2: features for the real blocking candidate pairs.

Reuses ``features.compute_pair_features`` (Phase 1) unchanged, and adds the
blocking/competition/meta features that only exist once real candidates do:

    n_keys_hit, cheap_score, rank_in_s1, n_candidates_for_s1   (blocking)
    n_competitors, rank_among_competitors                       (competition)
    other_source                                                (meta; already
                                                                   passed through
                                                                   by compute_pair_features)
    label                                                        (train only:
                                                                   1 if the pair
                                                                   is in the
                                                                   ground truth)

Reads the ``{prefix}_scored`` table that ``blocking.py`` left in the shared
DuckDB warehouse (``db.connect()`` points at the same file, so nothing needs
re-exporting from blocking.py). Streams in chunks through a multiprocessing
pool exactly like ``features.py``'s own batch writer, so this scales to the
tens of millions of test candidate pairs without holding them all in memory.

Usage: ``python features_candidates.py train`` or ``... test``.
"""

import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import FEATURES_DIR, RAW_DIR, TRAIN_GT_LONG_PARQUET, raw_parquet_path
from db import connect
from features import WORKERS, compute_pair_features, make_pool
import time

from perf import print_sysinfo, stage, progress_line

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BATCH_ROWS = 100_000


def s23_raw_union_sql(split):
    """SQL for a (source, id, business_name, business_address, country) union of S2+S3 raw text."""
    s2 = raw_parquet_path(split, "S2").as_posix()
    s3 = raw_parquet_path(split, "S3").as_posix()
    return (f"SELECT 'S2' AS source, id, business_name, business_address, country FROM '{s2}' "
            f"UNION ALL SELECT 'S3' AS source, id, business_name, business_address, country FROM '{s3}'")


def build_candidate_pairs_view(con, split, scored_table, with_label):
    """Create a view joining the scored candidates with raw text, blocking stats, and (train) labels."""
    s1_path = raw_parquet_path(split, "S1").as_posix()
    s23_sql = s23_raw_union_sql(split)

    con.execute(f"CREATE OR REPLACE VIEW v_{scored_table}_s23_raw AS {s23_sql}")
    con.execute(f"""
        CREATE OR REPLACE VIEW v_{scored_table}_competition AS
        SELECT match_id, match_source, count(DISTINCT s1_id) AS n_competitors
        FROM {scored_table} GROUP BY match_id, match_source
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW v_{scored_table}_ranked AS
        SELECT sc.*,
               row_number() OVER (PARTITION BY sc.match_id, sc.match_source ORDER BY sc.cheap_score DESC) AS rank_among_competitors,
               count(*) OVER (PARTITION BY sc.s1_id) AS n_candidates_for_s1
        FROM {scored_table} sc
    """)

    label_col = ""
    label_join = ""
    if with_label:
        label_col = ", (g.s1_id IS NOT NULL) AS label"
        label_join = (f"LEFT JOIN '{TRAIN_GT_LONG_PARQUET.as_posix()}' g "
                       f"ON g.s1_id = r.s1_id AND g.match_id = r.match_id AND g.match_source = r.match_source")

    con.execute(f"""
        CREATE OR REPLACE VIEW v_{scored_table}_pairs AS
        SELECT
            r.s1_id, r.match_id, r.match_source AS other_source,
            s1.business_name AS s1_name, s1.business_address AS s1_addr, s1.country AS s1_country,
            o.business_name AS other_name, o.business_address AS other_addr, o.country AS other_country,
            r.n_keys_hit, r.cheap_score, r.rank_in_s1, r.n_candidates_for_s1,
            comp.n_competitors, r.rank_among_competitors
            {label_col}
        FROM v_{scored_table}_ranked r
        JOIN '{s1_path}' s1 ON s1.id = r.s1_id
        JOIN v_{scored_table}_s23_raw o ON o.id = r.match_id AND o.source = r.match_source
        JOIN v_{scored_table}_competition comp ON comp.match_id = r.match_id AND comp.match_source = r.match_source
        {label_join}
    """)


BLOCKING_COLS = ["n_keys_hit", "cheap_score", "rank_in_s1", "n_candidates_for_s1",
                  "n_competitors", "rank_among_competitors"]


def featurize_candidates(con, split, scored_table, out_path, with_label):
    """Stream the candidate-pairs view through compute_pair_features and write it to parquet."""
    build_candidate_pairs_view(con, split, scored_table, with_label)
    n_total = con.execute(f"SELECT count(*) FROM v_{scored_table}_pairs").fetchone()[0]
    print(f"  {scored_table}: {n_total} candidate pairs to featurize")

    cols = ["s1_id", "match_id", "s1_name", "s1_addr", "s1_country",
            "other_name", "other_addr", "other_country", "other_source"] + BLOCKING_COLS
    if with_label:
        cols.append("label")
    result = con.execute(f"SELECT {', '.join(cols)} FROM v_{scored_table}_pairs")

    writer = None
    n_seen = 0
    t0 = time.perf_counter()
    with make_pool(WORKERS) as pool:
        for batch in result.to_arrow_reader(BATCH_ROWS):
            df = batch.to_pandas()
            feats = compute_pair_features(df, pool=pool)
            feats["s1_id"] = df["s1_id"].values
            feats["match_id"] = df["match_id"].values
            feats["s1_country"] = df["s1_country"].values
            for c in BLOCKING_COLS:
                feats[c] = df[c].values
            if with_label:
                feats["label"] = df["label"].astype(int).values
            table = pa.Table.from_pandas(feats, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            n_seen += len(df)
            print(f"  {progress_line(scored_table, n_seen, n_total, t0)}")
    if writer is not None:
        writer.close()
    print(f"  saved -> {out_path}")


def main():
    """Featurize either the train-sample candidates (labeled) or the full test candidates."""
    which = sys.argv[1] if len(sys.argv) > 1 else "train"
    print_sysinfo()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    con = connect(memory_limit_gb=3, threads=2)

    if which == "train":
        with stage("featurize train-sample candidates"):
            featurize_candidates(con, "train", "trainsample_scored",
                                  FEATURES_DIR / "trainsample_candidates_features.parquet",
                                  with_label=True)
    elif which == "test":
        with stage("featurize test candidates"):
            featurize_candidates(con, "test", "test_scored",
                                  FEATURES_DIR / "test_candidates_features.parquet",
                                  with_label=False)
    else:
        raise SystemExit(f"unknown arg {which!r}, expected 'train' or 'test'")

    con.close()


if __name__ == "__main__":
    main()
