"""Step 2 ingest: convert raw challenge TSVs into typed parquet, once.

Every downstream Step 2 script (mining, normalization, blocking) reads from
these parquet files instead of the raw TSVs, so the expensive CSV parsing and
id-splitting happens exactly once. Uses DuckDB's vectorized CSV reader (not a
per-row Python loop) so it scales to the full ~22M-row dataset without
exceeding memory.

Per source file (train/test x S1/S2/S3), writes ``data/raw/{split}_{source}.parquet``
with columns:
    id (BIGINT)               -- the numeric part of entity_id (no leading zeros
                                  exist in this dataset, verified against the raw
                                  files, so int64 round-trips losslessly)
    business_name (VARCHAR)
    business_address (VARCHAR)
    country (VARCHAR)

The original ``source`` prefix ("S1"/"S2"/"S3") is implied by which file/table a
row is in; ``blocking.py`` and friends reconstruct "S1-<id>" only when writing
final output.

The ground truth is written twice:
    data/raw/train_gt_wide.parquet  -- (s1_id, matched_entity_ids) as given
    data/raw/train_gt_long.parquet  -- one row per (s1_id, match_id, match_source)
                                        pair, exploded via SQL (no Python loop)
"""

from config import RAW_DIR, SOURCE_TSV, TRAIN_GROUND_TRUTH, raw_parquet_path
from db import connect
from perf import print_sysinfo, stage

# Disables DuckDB's quote handling (these files are not quoted) and remaps
# empty CSV fields to a nullstr sentinel that never occurs in the data, so
# empty strings are preserved as '' instead of being silently turned into
# NULL -- the DuckDB analogue of pandas' keep_default_na=False.
CSV_OPTS = "delim='\\t', header=true, quote='', escape='', all_varchar=true, nullstr='\x01'"


def ingest_source(con, split, source):
    """Convert one raw source TSV into its typed parquet file."""
    src_path = SOURCE_TSV[(split, source)]
    out_path = raw_parquet_path(split, source)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT
                CAST(regexp_extract(entity_id, '(\\d+)$', 1) AS BIGINT) AS id,
                business_name,
                business_address,
                country
            FROM read_csv('{src_path.as_posix()}', {CSV_OPTS})
        ) TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """)
    n = con.execute(f"SELECT count(*) FROM '{out_path.as_posix()}'").fetchone()[0]
    print(f"  {split}_{source}: {n} rows -> {out_path}")


def ingest_ground_truth(con):
    """Convert train_ground_truth.tsv into a wide and a long (exploded) parquet."""
    wide_path = RAW_DIR / "train_gt_wide.parquet"
    long_path = RAW_DIR / "train_gt_long.parquet"
    con.execute(f"""
        COPY (
            SELECT
                CAST(regexp_extract(source1_entity_id, '(\\d+)$', 1) AS BIGINT) AS s1_id,
                matched_entity_ids
            FROM read_csv('{TRAIN_GROUND_TRUTH.as_posix()}', {CSV_OPTS})
        ) TO '{wide_path.as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT
                s1_id,
                CAST(regexp_extract(m, '(\\d+)$', 1) AS BIGINT) AS match_id,
                substr(m, 1, 2) AS match_source
            FROM (
                SELECT s1_id, unnest(string_split(matched_entity_ids, ',')) AS m
                FROM '{wide_path.as_posix()}'
                WHERE matched_entity_ids <> ''
            )
        ) TO '{long_path.as_posix()}' (FORMAT PARQUET)
    """)
    n_wide = con.execute(f"SELECT count(*) FROM '{wide_path.as_posix()}'").fetchone()[0]
    n_long = con.execute(f"SELECT count(*) FROM '{long_path.as_posix()}'").fetchone()[0]
    print(f"  train_gt_wide: {n_wide} rows -> {wide_path}")
    print(f"  train_gt_long: {n_long} matched pairs -> {long_path}")


def main():
    """Ingest all six source files plus the ground truth into parquet."""
    print_sysinfo()
    con = connect()
    with stage("ingest sources"):
        for split, source in SOURCE_TSV:
            ingest_source(con, split, source)
    with stage("ingest ground truth"):
        ingest_ground_truth(con)
    con.close()


if __name__ == "__main__":
    main()
