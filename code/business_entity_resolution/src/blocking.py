"""Step 2 Part C/D/E/F: token IDF, blocking key generation, recall measurement
on a train sample, and full test candidate generation.

Everything here is DuckDB SQL over the normalized parquet from ``normalize.py``
-- no per-row Python loops over the full candidate space. Country is part of
every key (open-set string equality, never a fixed list).

Pipeline (see ``main()``):
  1. compute_idf         -- Part C: per (split, country) token document frequency
  2. build_key_tables     -- per-record top rare tokens/numbers -> the 4 key families
  3. join_keys            -- S1 keys x S2/S3 keys -> candidate pairs + cheap score
  4. measure_recall       -- Part E: on a stratified 300k train S1 sample vs full pools
  5. write_test_candidates -- Part F: full test S1 vs full test S2/S3 pools
"""

import random
import sys

from config import (BLOCKING_REPORT_PATH, CAND_DIR, DICTS_DIR, OUTPUT_DIR,
                     TEST_CANDIDATES_PARQUET, TRAIN_GT_LONG_PARQUET,
                     TRAIN_SAMPLE_S1_PARQUET, norm_parquet_path,
                     raw_parquet_path)
from db import connect
from io_utils import write_id_list_tsv
from perf import print_sysinfo, stage

random.seed(42)

sys.stdout.reconfigure(encoding="utf-8")

TOP_NAME_TOKENS = 2
TOP_ADDR_WORDS = 2
TOP_NUMBERS = 3
CAP_DEFAULT = 50  # tuned down from an initial 300: at 300 the raw candidate
# join produced 56M rows / 49M distinct (s1,match) pairs from just the 300k
# train sample, which exceeded available memory on this 16GB machine well
# before it could even reach the top-K cut. See BLOCKING_REPORT.md.
CAP_K2_NOSTATE = 15  # stricter cap for the (country, name_token) fallback with no state
TOP_K_PER_S1 = 40
MIN_NOSPACE_LEN = 3


def norm_view(con, split, source):
    """Create/replace a view of one source's norm parquet, tagged with its source."""
    path = norm_parquet_path(split, source).as_posix()
    con.execute(f"CREATE OR REPLACE VIEW v_{split}_{source}_norm AS "
                f"SELECT '{source}' AS source, * FROM '{path}'")


def all_norm_union(con, split):
    """Create a view unioning S1+S2+S3 normalized records for one split."""
    parts = [f"SELECT * FROM v_{split}_{s}_norm" for s in ("S1", "S2", "S3")]
    con.execute(f"CREATE OR REPLACE VIEW v_{split}_all_norm AS {' UNION ALL '.join(parts)}")


def s23_norm_union(con, split):
    """Create a view unioning S2+S3 normalized records for one split."""
    con.execute(f"CREATE OR REPLACE VIEW v_{split}_S23_norm AS "
                f"SELECT * FROM v_{split}_S2_norm UNION ALL SELECT * FROM v_{split}_S3_norm")


# ---------------------------------------------------------------------------
# Part C: IDF
# ---------------------------------------------------------------------------

def compute_idf(con, split):
    """Compute per-(country, token) document frequency and IDF for name_core and addr_words."""
    con.execute(f"""
        CREATE OR REPLACE TABLE country_totals_{split} AS
        SELECT country, count(*) AS n FROM v_{split}_all_norm GROUP BY country
    """)
    for field, table in (("name_core", f"idf_name_{split}"), ("addr_words", f"idf_addr_{split}")):
        con.execute(f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT d.country, d.token, d.df, t.n, ln(t.n::DOUBLE / d.df) AS idf
            FROM (
                SELECT country, token, count(*) AS df
                FROM (
                    SELECT DISTINCT country, id, unnest(string_split({field}, ' ')) AS token
                    FROM v_{split}_all_norm
                    WHERE {field} <> ''
                )
                GROUP BY country, token
            ) d
            JOIN country_totals_{split} t USING (country)
        """)
        out_path = DICTS_DIR / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out_path.as_posix()}' (FORMAT PARQUET)")


# ---------------------------------------------------------------------------
# Part D: per-record top rare tokens/numbers, then the 4 key families
# ---------------------------------------------------------------------------

def build_keys_table(con, split, side_view, out_table):
    """Build a per-record table of top rare name/addr tokens and numbers, for one side.

    ``side_view`` must expose: source, id, country, state, name_core, name_alt,
    name_nospace, addr_words, numbers. Ranks name_core/addr_words tokens by
    (country-specific) IDF descending and numbers by digit-length descending,
    keeping the top few of each as DuckDB LISTs for later UNNEST cross-joins.
    """
    # Each token family is materialized as its OWN table, sequentially, rather
    # than as CTEs inside one big query: with everything in one query DuckDB's
    # planner can end up holding all four unnest+window-function pipelines'
    # intermediate state in memory at once, which is what pushed this past
    # the memory cap on the full S2/S3 pool. Doing them one at a time (each
    # written out and freed before the next starts) uses far less peak RAM.
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_name AS
        SELECT source, id, list(token ORDER BY rn) AS top_name_tokens FROM (
            SELECT source, id, token,
                   row_number() OVER (PARTITION BY source, id ORDER BY coalesce(i.idf, 0) DESC) AS rn
            FROM (
                SELECT source, id, country, unnest(string_split(name_core, ' ')) AS token
                FROM {side_view} WHERE name_core <> ''
            ) x
            LEFT JOIN idf_name_{split} i USING (country, token)
        ) WHERE rn <= {TOP_NAME_TOKENS} GROUP BY source, id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_alt AS
        SELECT source, id, list(token ORDER BY rn) AS top_alt_tokens FROM (
            SELECT source, id, token,
                   row_number() OVER (PARTITION BY source, id ORDER BY coalesce(i.idf, 0) DESC) AS rn
            FROM (
                SELECT source, id, country, unnest(string_split(name_alt, ' ')) AS token
                FROM {side_view} WHERE name_alt <> ''
            ) x
            LEFT JOIN idf_name_{split} i USING (country, token)
        ) WHERE rn <= {TOP_NAME_TOKENS} GROUP BY source, id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_addr AS
        SELECT source, id, list(token ORDER BY rn) AS top_addr_words FROM (
            SELECT source, id, token,
                   row_number() OVER (PARTITION BY source, id ORDER BY coalesce(i.idf, 0) DESC) AS rn
            FROM (
                SELECT source, id, country, unnest(string_split(addr_words, ' ')) AS token
                FROM {side_view} WHERE addr_words <> ''
            ) x
            LEFT JOIN idf_addr_{split} i USING (country, token)
        ) WHERE rn <= {TOP_ADDR_WORDS} GROUP BY source, id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_nums AS
        SELECT source, id, list(num ORDER BY rn) AS top_numbers FROM (
            SELECT source, id, num,
                   row_number() OVER (PARTITION BY source, id ORDER BY length(num) DESC) AS rn
            FROM (
                SELECT source, id, unnest(string_split(numbers, ',')) AS num
                FROM {side_view} WHERE numbers <> ''
            )
        ) WHERE rn <= {TOP_NUMBERS} GROUP BY source, id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT
            s.source, s.id, s.country, s.state, s.name_alt, s.name_nospace,
            replace(s.name_alt, ' ', '') AS name_alt_nospace,
            na.top_name_tokens, aa.top_alt_tokens, ada.top_addr_words, nua.top_numbers
        FROM {side_view} s
        LEFT JOIN {out_table}_name na USING (source, id)
        LEFT JOIN {out_table}_alt aa USING (source, id)
        LEFT JOIN {out_table}_addr ada USING (source, id)
        LEFT JOIN {out_table}_nums nua USING (source, id)
    """)
    for suffix in ("name", "alt", "addr", "nums"):
        con.execute(f"DROP TABLE IF EXISTS {out_table}_{suffix}")


def generate_key_rows(con, keys_table, out_table):
    """Explode the top-token/number LISTs into (id, source, country, key_type, key_value) rows."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        -- K1: address (country, number, rare_word), up to 3 numbers x 2 rare words
        SELECT source, id, country, 'K1' AS key_type,
               country || '|' || num || '|' || word AS key_value
        FROM {keys_table}, UNNEST(top_numbers) AS t1(num), UNNEST(top_addr_words) AS t2(word)

        UNION ALL
        -- K2: name+place, with state
        SELECT source, id, country, 'K2' AS key_type,
               country || '|' || state || '|' || tok AS key_value
        FROM {keys_table}, UNNEST(top_name_tokens) AS t(tok)
        WHERE state <> ''

        UNION ALL
        -- K2 fallback: no state, stricter cap applied later via key_type tag
        SELECT source, id, country, 'K2_nostate' AS key_type,
               country || '|' || tok AS key_value
        FROM {keys_table}, UNNEST(top_name_tokens) AS t(tok)
        WHERE state = ''

        UNION ALL
        -- K3: name_nospace exact + first-8-chars
        SELECT source, id, country, 'K3_exact' AS key_type, country || '|' || name_nospace AS key_value
        FROM {keys_table} WHERE length(name_nospace) >= {MIN_NOSPACE_LEN}

        UNION ALL
        SELECT source, id, country, 'K3_pfx8' AS key_type,
               country || '|' || substr(name_nospace, 1, 8) AS key_value
        FROM {keys_table} WHERE length(name_nospace) >= {MIN_NOSPACE_LEN}

        UNION ALL
        -- K4: K2/K3 computed on name_alt (only when a wrapper wa found)
        SELECT source, id, country, 'K4_state' AS key_type,
               country || '|' || state || '|' || tok AS key_value
        FROM {keys_table}, UNNEST(top_alt_tokens) AS t(tok)
        WHERE state <> '' AND name_alt <> ''

        UNION ALL
        SELECT source, id, country, 'K4_nostate' AS key_type,
               country || '|' || tok AS key_value
        FROM {keys_table}, UNNEST(top_alt_tokens) AS t(tok)
        WHERE state = '' AND name_alt <> ''

        UNION ALL
        SELECT source, id, country, 'K4_exact' AS key_type, country || '|' || name_alt_nospace AS key_value
        FROM {keys_table} WHERE length(name_alt_nospace) >= {MIN_NOSPACE_LEN}
    """)


def cap_s23_keys(con, s23_key_table, out_table):
    """Drop any S2/S3-side key whose block size exceeds the per-key-type cap."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH counts AS (
            SELECT key_type, key_value, count(*) AS n
            FROM {s23_key_table} GROUP BY key_type, key_value
        )
        SELECT k.* FROM {s23_key_table} k
        JOIN counts c USING (key_type, key_value)
        WHERE c.n <= CASE WHEN k.key_type IN ('K2_nostate', 'K4_nostate') THEN {CAP_K2_NOSTATE} ELSE {CAP_DEFAULT} END
    """)


def join_candidate_pairs(con, s1_key_table, s23_key_table_capped, out_table):
    """Join S1 keys against capped S2/S3 keys, matching within the same key_type.

    Joining on (country, key_type, key_value) -- not just (country, key_value)
    -- matters: different key families build key_value strings independently
    (e.g. K3_exact's "country|name_nospace" vs K2_nostate's "country|token")
    and could coincidentally collide without the key_type in the join.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT a.id AS s1_id, b.id AS match_id, b.source AS match_source, a.key_type
        FROM {s1_key_table} a
        JOIN {s23_key_table_capped} b USING (country, key_type, key_value)
    """)


def build_token_tables(con, split):
    """Materialize per-record (id, token) tables for name_core and addr_words.

    Used for the IDF shared-token score component as a proper set-based join
    (pairs JOIN tokens JOIN tokens JOIN idf), never as a per-pair correlated
    subquery -- that pattern re-scans/re-splits text per row and does not
    scale past a few thousand pairs, let alone tens of millions.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE {split}_s1_name_tok AS
        SELECT id, country, unnest(string_split(name_core, ' ')) AS token
        FROM v_{split}_S1_norm WHERE name_core <> ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {split}_s23_name_tok AS
        SELECT id, source, unnest(string_split(name_core, ' ')) AS token
        FROM v_{split}_S23_norm WHERE name_core <> ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {split}_s1_addr_tok AS
        SELECT id, country, unnest(string_split(addr_words, ' ')) AS token
        FROM v_{split}_S1_norm WHERE addr_words <> ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {split}_s23_addr_tok AS
        SELECT id, source, unnest(string_split(addr_words, ' ')) AS token
        FROM v_{split}_S23_norm WHERE addr_words <> ''
    """)


def score_and_topk(con, split, pairs_table, out_table, top_k=TOP_K_PER_S1):
    """Aggregate key hits per pair, add shared-token IDF sum, keep the top K per S1.

    The S2/S3-side token tables (``{split}_s23_name_tok``/``addr_tok``) cover
    the FULL multi-million-row pool. Joining directly against them per pair
    forces DuckDB to hash the entire pool, which is what blew the memory cap
    here. Filtering them down to only the (id, source) pairs that actually
    appear in this call's candidate set first -- a tiny fraction of the full
    pool -- makes the subsequent shared-token join cheap.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_agg AS
        SELECT s1_id, match_id, match_source, count(DISTINCT key_type) AS n_keys_hit
        FROM {pairs_table}
        GROUP BY s1_id, match_id, match_source
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_cand_s23 AS
        SELECT DISTINCT match_id AS id, match_source AS source FROM {out_table}_agg
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_s23_name_tok AS
        SELECT t.* FROM {split}_s23_name_tok t JOIN {out_table}_cand_s23 c USING (id, source)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_s23_addr_tok AS
        SELECT t.* FROM {split}_s23_addr_tok t JOIN {out_table}_cand_s23 c USING (id, source)
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_shared_name AS
        SELECT p.s1_id, p.match_id, p.match_source, sum(i.idf) AS shared_name_idf
        FROM {out_table}_agg p
        JOIN {split}_s1_name_tok a ON a.id = p.s1_id
        JOIN {out_table}_s23_name_tok b ON b.id = p.match_id AND b.source = p.match_source AND b.token = a.token
        JOIN idf_name_{split} i ON i.country = a.country AND i.token = a.token
        GROUP BY p.s1_id, p.match_id, p.match_source
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_shared_addr AS
        SELECT p.s1_id, p.match_id, p.match_source, sum(i.idf) AS shared_addr_idf
        FROM {out_table}_agg p
        JOIN {split}_s1_addr_tok a ON a.id = p.s1_id
        JOIN {out_table}_s23_addr_tok b ON b.id = p.match_id AND b.source = p.match_source AND b.token = a.token
        JOIN idf_addr_{split} i ON i.country = a.country AND i.token = a.token
        GROUP BY p.s1_id, p.match_id, p.match_source
    """)
    for t in ("cand_s23", "s23_name_tok", "s23_addr_tok"):
        con.execute(f"DROP TABLE IF EXISTS {out_table}_{t}")
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT p.s1_id, p.match_id, p.match_source, p.n_keys_hit,
               (p.n_keys_hit + coalesce(sn.shared_name_idf, 0) + coalesce(sa.shared_addr_idf, 0)) AS cheap_score,
               row_number() OVER (PARTITION BY p.s1_id ORDER BY
                   (p.n_keys_hit + coalesce(sn.shared_name_idf, 0) + coalesce(sa.shared_addr_idf, 0)) DESC,
                   p.match_id) AS rank_in_s1
        FROM {out_table}_agg p
        LEFT JOIN {out_table}_shared_name sn USING (s1_id, match_id, match_source)
        LEFT JOIN {out_table}_shared_addr sa USING (s1_id, match_id, match_source)
    """)


S1_BATCH_SIZE = 150_000


def build_all_keys(con, split, s1_view, prefix):
    """Build keys, join, and score candidate pairs -- one (country, S1 batch) at a time.

    ``s1_view`` lets Part E score a 300k S1 sample against the FULL S2/S3 pool
    while Part F scores all of test S1 the same way -- one code path for both.
    Returns the name of the final scored-with-rank table (union of every
    country/batch).

    The S2/S3 side is built ONCE per country (it doesn't depend on which S1
    batch is being scored) and reused across that country's S1 batches. The
    S1 side is additionally split into fixed-size batches of
    ``S1_BATCH_SIZE`` records within each country: Part E's 300k-row sample
    (effectively ~120-180k per country) ran fine per-country, but Part F's
    full test S1 set has 660k-810k records in its largest countries, and
    scoring a country that large in one pass still exhausted both memory and
    a temp-spill limit of 116 GB on the D: scratch drive. Batching the S1
    side bounds every pass to roughly the same size regardless of how big a
    single country's test population is. Country alone still guarantees
    correctness (blocking keys always include country, so results never
    depend on how S1 is chunked within a country).
    """
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM v_{split}_S23_norm").fetchall()]
    scored_parts = []
    pairs_raw_parts = []
    for i, country in enumerate(countries):
        # CREATE VIEW can't take a prepared-statement parameter (DuckDB
        # BinderException) -- inline the literal, escaping any single quote.
        safe_country = country.replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW v_{split}_s1_country AS "
            f"SELECT * FROM {s1_view} WHERE country = '{safe_country}'"
        )
        con.execute(
            f"CREATE OR REPLACE VIEW v_{split}_S23_norm_part AS "
            f"SELECT * FROM v_{split}_S23_norm WHERE country = '{safe_country}'"
        )

        # S2/S3 side: built once, reused by every S1 batch of this country.
        s23_keys_raw, s23_keys = f"{split}_s23_keys_raw_c{i}", f"{split}_s23_keys_c{i}"
        s23_keys_capped = f"{split}_s23_keys_capped_c{i}"
        build_keys_table(con, split, f"v_{split}_S23_norm_part", s23_keys_raw)
        generate_key_rows(con, s23_keys_raw, s23_keys)
        cap_s23_keys(con, s23_keys, s23_keys_capped)
        con.execute(f"DROP TABLE IF EXISTS {s23_keys_raw}")
        con.execute(f"DROP TABLE IF EXISTS {s23_keys}")

        n_country = con.execute(f"SELECT count(*) FROM v_{split}_s1_country").fetchone()[0]
        n_batches = max(1, -(-n_country // S1_BATCH_SIZE))  # ceil division
        con.execute(f"""
            CREATE OR REPLACE VIEW v_{split}_s1_country_numbered AS
            SELECT *, (row_number() OVER (ORDER BY id) - 1) // {S1_BATCH_SIZE} AS _batch
            FROM v_{split}_s1_country
        """)
        for b in range(n_batches):
            con.execute(
                f"CREATE OR REPLACE VIEW v_{split}_s1_batch AS "
                f"SELECT * EXCLUDE (_batch) FROM v_{split}_s1_country_numbered WHERE _batch = {b}"
            )
            s1_keys_raw, s1_keys = f"{prefix}_s1_keys_raw_c{i}_b{b}", f"{prefix}_s1_keys_c{i}_b{b}"
            build_keys_table(con, split, f"v_{split}_s1_batch", s1_keys_raw)
            generate_key_rows(con, s1_keys_raw, s1_keys)

            pairs_raw = f"{prefix}_pairs_raw_c{i}_b{b}"
            join_candidate_pairs(con, s1_keys, s23_keys_capped, pairs_raw)
            scored_part = f"{prefix}_scored_c{i}_b{b}"
            score_and_topk(con, split, pairs_raw, scored_part)
            scored_parts.append(scored_part)
            pairs_raw_parts.append(pairs_raw)

            con.execute(f"DROP TABLE IF EXISTS {s1_keys_raw}")
            con.execute(f"DROP TABLE IF EXISTS {s1_keys}")

        con.execute(f"DROP TABLE IF EXISTS {s23_keys_capped}")

    # kept (not dropped per-batch) since measure_recall's per-key-type
    # recall breakdown needs the raw, un-aggregated key hits
    union_sql = " UNION ALL ".join(f"SELECT * FROM {t}" for t in pairs_raw_parts)
    con.execute(f"CREATE OR REPLACE TABLE {prefix}_pairs_raw AS {union_sql}")
    for t in pairs_raw_parts:
        con.execute(f"DROP TABLE IF EXISTS {t}")

    union_sql = " UNION ALL ".join(f"SELECT * FROM {t}" for t in scored_parts)
    con.execute(f"CREATE OR REPLACE TABLE {prefix}_scored AS {union_sql}")
    for t in scored_parts:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return f"{prefix}_scored"


RECALL_K_SWEEP = (10, 20, 30, 40, 60)
RECALL_TARGET = 0.97


def measure_recall(con):
    """Part E: measure blocking recall on the 300k train S1 sample vs full train pools.

    Returns (report_lines, chosen_k) where ``report_lines`` is ready to drop
    into BLOCKING_REPORT.md and ``chosen_k`` is the smallest swept K reaching
    RECALL_TARGET union recall (or the largest swept K, if none reach it).
    """
    sample_path = TRAIN_SAMPLE_S1_PARQUET.as_posix()
    gt_path = TRAIN_GT_LONG_PARQUET.as_posix()
    lines = []

    con.execute(f"""
        CREATE OR REPLACE VIEW v_train_S1_sample_norm AS
        SELECT n.* FROM v_train_S1_norm n JOIN '{sample_path}' s ON n.id = s.id
    """)
    n_sample = con.execute("SELECT count(*) FROM v_train_S1_sample_norm").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE trainsample_gt AS
        SELECT g.s1_id, g.match_id, g.match_source
        FROM '{gt_path}' g JOIN '{sample_path}' s ON g.s1_id = s.id
    """)
    n_true_pairs = con.execute("SELECT count(*) FROM trainsample_gt").fetchone()[0]
    n_nonsingleton = con.execute("SELECT count(DISTINCT s1_id) FROM trainsample_gt").fetchone()[0]
    lines.append(f"Sample S1 entities: {n_sample}")
    lines.append(f"True (S1,match) pairs among sample: {n_true_pairs}, across {n_nonsingleton} non-singleton S1 entities")

    with stage("Part E: build+score sample-S1 vs full-pool candidates"):
        scored_table = build_all_keys(con, "train", "v_train_S1_sample_norm", "trainsample")

    con.execute("""
        CREATE OR REPLACE TABLE trainsample_gt_hits AS
        SELECT g.s1_id, g.match_id, g.match_source, s.country,
               list(DISTINCT p.key_type) FILTER (WHERE p.key_type IS NOT NULL) AS hit_key_types
        FROM trainsample_gt g
        JOIN v_train_S1_sample_norm s ON s.id = g.s1_id
        LEFT JOIN trainsample_pairs_raw p
            ON p.s1_id = g.s1_id AND p.match_id = g.match_id AND p.match_source = g.match_source
        GROUP BY g.s1_id, g.match_id, g.match_source, s.country
    """)

    # --- per-key-type and union recall, before the top-K cut ---
    lines.append("\nPair recall BEFORE top-K cut (per key type, and union):")
    key_types = [r[0] for r in con.execute("SELECT DISTINCT key_type FROM trainsample_pairs_raw").fetchall()]
    for kt in sorted(key_types):
        hit = con.execute(
            "SELECT count(*) FROM trainsample_gt_hits WHERE list_contains(hit_key_types, ?)", [kt]
        ).fetchone()[0]
        lines.append(f"  {kt}: {hit}/{n_true_pairs} ({100.0 * hit / n_true_pairs:.2f}%)")
    hit_union = con.execute(
        "SELECT count(*) FROM trainsample_gt_hits WHERE len(hit_key_types) > 0"
    ).fetchone()[0]
    union_recall_before = hit_union / n_true_pairs
    lines.append(f"  UNION (any key): {hit_union}/{n_true_pairs} ({100.0 * union_recall_before:.2f}%)")

    lines.append("\nUnion recall BEFORE cut, by country:")
    for country, total, hit in con.execute("""
        SELECT country, count(*), sum(CASE WHEN len(hit_key_types) > 0 THEN 1 ELSE 0 END)
        FROM trainsample_gt_hits GROUP BY country
    """).fetchall():
        lines.append(f"  {country}: {hit}/{total} ({100.0 * hit / total:.2f}%)")

    lines.append("\nUnion recall BEFORE cut, by match source:")
    for source, total, hit in con.execute("""
        SELECT match_source, count(*), sum(CASE WHEN len(hit_key_types) > 0 THEN 1 ELSE 0 END)
        FROM trainsample_gt_hits GROUP BY match_source
    """).fetchall():
        lines.append(f"  {source}: {hit}/{total} ({100.0 * hit / total:.2f}%)")

    # --- recall vs K sweep (after cut) ---
    lines.append("\nUnion recall AFTER top-K cut, swept over K:")
    recall_at_k = {}
    for k in RECALL_K_SWEEP:
        con.execute(f"""
            CREATE OR REPLACE TABLE trainsample_hits_k{k} AS
            SELECT g.s1_id, g.match_id, g.match_source,
                   (s.match_id IS NOT NULL) AS hit
            FROM trainsample_gt g
            LEFT JOIN (SELECT s1_id, match_id, match_source FROM {scored_table} WHERE rank_in_s1 <= {k}) s
                ON s.s1_id = g.s1_id AND s.match_id = g.match_id AND s.match_source = g.match_source
        """)
        hit_k = con.execute(f"SELECT sum(CASE WHEN hit THEN 1 ELSE 0 END) FROM trainsample_hits_k{k}").fetchone()[0]
        recall_k = hit_k / n_true_pairs
        recall_at_k[k] = recall_k
        lines.append(f"  K={k}: {hit_k}/{n_true_pairs} ({100.0 * recall_k:.2f}%)")

    chosen_k = next((k for k in RECALL_K_SWEEP if recall_at_k[k] >= RECALL_TARGET), max(RECALL_K_SWEEP))
    best_recall = recall_at_k[chosen_k]
    lines.append(f"\nChosen K = {chosen_k} (smallest swept K reaching {RECALL_TARGET:.0%} union recall; "
                 f"achieved {100.0 * best_recall:.2f}%)"
                 if best_recall >= RECALL_TARGET else
                 f"\nNo swept K reached {RECALL_TARGET:.0%} union recall; largest K={chosen_k} achieved "
                 f"only {100.0 * best_recall:.2f}%. See 'what I'd try next' in the report.")

    # --- per-S1 all/any-found, at chosen K ---
    con.execute(f"""
        CREATE OR REPLACE TABLE trainsample_per_s1_k AS
        SELECT g.s1_id, count(*) AS n_true,
               sum(CASE WHEN s.match_id IS NOT NULL THEN 1 ELSE 0 END) AS n_found
        FROM trainsample_gt g
        LEFT JOIN (SELECT s1_id, match_id, match_source FROM {scored_table} WHERE rank_in_s1 <= {chosen_k}) s
            ON s.s1_id = g.s1_id AND s.match_id = g.match_id AND s.match_source = g.match_source
        GROUP BY g.s1_id
    """)
    n_all_found, n_any_found, n_ent = con.execute("""
        SELECT sum(CASE WHEN n_found = n_true THEN 1 ELSE 0 END),
               sum(CASE WHEN n_found > 0 THEN 1 ELSE 0 END),
               count(*)
        FROM trainsample_per_s1_k
    """).fetchone()
    lines.append(f"\nAt K={chosen_k}: non-singleton S1 with ALL true matches found: "
                 f"{n_all_found}/{n_ent} ({100.0 * n_all_found / n_ent:.2f}%); "
                 f"with >=1 found: {n_any_found}/{n_ent} ({100.0 * n_any_found / n_ent:.2f}%)")

    # --- candidates per S1 stats, at chosen K ---
    con.execute(f"""
        CREATE OR REPLACE TABLE trainsample_cand_counts AS
        SELECT s1_id, count(*) AS n FROM {scored_table} WHERE rank_in_s1 <= {chosen_k} GROUP BY s1_id
    """)
    mean_n, median_n, p95_n, max_n, total_pairs = con.execute("""
        SELECT avg(n), median(n), quantile_cont(n, 0.95), max(n), sum(n) FROM trainsample_cand_counts
    """).fetchone()
    lines.append(f"\nCandidates per S1 at K={chosen_k}: mean={mean_n:.1f}, median={median_n:.0f}, "
                 f"p95={p95_n:.0f}, max={max_n}, total candidate pairs={total_pairs}")

    # --- 30 random missed pairs (never hit by any key), with raw + normalized fields ---
    missed = con.execute(f"""
        SELECT s1_id, match_id, match_source FROM trainsample_gt_hits
        WHERE hit_key_types IS NULL OR len(hit_key_types) = 0
        ORDER BY random() LIMIT 30
    """).fetchall()
    lines.append(f"\n30 random MISSED true pairs (never matched by any key), out of "
                 f"{n_true_pairs - hit_union} total misses:")
    for s1_id, match_id, match_source in missed:
        s1_row = con.execute(f"""
            SELECT r.business_name, r.business_address, n.name_core, n.addr_words, n.numbers, n.state
            FROM '{raw_parquet_path("train", "S1").as_posix()}' r
            JOIN v_train_S1_norm n USING (id) WHERE r.id = ?
        """, [s1_id]).fetchone()
        m_row = con.execute(f"""
            SELECT r.business_name, r.business_address, n.name_core, n.addr_words, n.numbers, n.state
            FROM '{raw_parquet_path("train", match_source).as_posix()}' r
            JOIN v_train_{match_source}_norm n USING (id) WHERE r.id = ?
        """, [match_id]).fetchone()
        s1_name_tok = set((s1_row[2] or "").split())
        m_name_tok = set((m_row[2] or "").split())
        s1_nums = set((s1_row[4] or "").split(","))
        m_nums = set((m_row[4] or "").split(","))
        if not (s1_name_tok & m_name_tok) and not (s1_nums & m_nums - {""}):
            reason = "no shared name token AND no shared number"
        elif not (s1_name_tok & m_name_tok):
            reason = "no shared name_core token"
        elif not (s1_nums & m_nums - {""}):
            reason = "no shared number"
        elif s1_row[5] and m_row[5] and s1_row[5] != m_row[5]:
            reason = "state mismatch after normalization"
        else:
            reason = "shared tokens/numbers exist but all keys still missed (rare-word/cap edge case)"
        lines.append(f"  [{reason}]")
        lines.append(f"    S1-{s1_id}: {s1_row[0]!r} | {s1_row[1]!r}  (name_core={s1_row[2]!r}, addr_words={s1_row[3]!r}, numbers={s1_row[4]!r}, state={s1_row[5]!r})")
        lines.append(f"    {match_source}-{match_id}: {m_row[0]!r} | {m_row[1]!r}  (name_core={m_row[2]!r}, addr_words={m_row[3]!r}, numbers={m_row[4]!r}, state={m_row[5]!r})")

    return lines, chosen_k


# ---------------------------------------------------------------------------
# Part F: full test candidates
# ---------------------------------------------------------------------------

def write_test_candidates(con, chosen_k):
    """Part F: block all test S1 against the full test S2/S3 pools; write outputs."""
    lines = []
    with stage("Part F: build+score all test-S1 vs full test pools"):
        scored_table = build_all_keys(con, "test", "v_test_S1_norm", "test")

    con.execute(f"""
        CREATE OR REPLACE TABLE test_final_candidates AS
        SELECT s1_id, match_id, match_source, n_keys_hit, cheap_score, rank_in_s1
        FROM {scored_table} WHERE rank_in_s1 <= {chosen_k}
    """)
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY test_final_candidates TO '{TEST_CANDIDATES_PARQUET.as_posix()}' (FORMAT PARQUET)")

    total_s1 = con.execute(f"SELECT count(*) FROM '{raw_parquet_path('test', 'S1').as_posix()}'").fetchone()[0]
    s1_with_cands = con.execute("SELECT count(DISTINCT s1_id) FROM test_final_candidates").fetchone()[0]
    lines.append(f"Test S1 entities: {total_s1}; with >=1 candidate: {s1_with_cands} "
                 f"({100.0 * s1_with_cands / total_s1:.2f}%); with 0 candidates: {total_s1 - s1_with_cands}")

    lines.append("\nCandidates per S1 by country:")
    rows = con.execute(f"""
        SELECT s.country, count(*) AS n_s1, avg(coalesce(c.n, 0)) AS mean_cand,
               median(coalesce(c.n, 0)) AS median_cand, max(coalesce(c.n, 0)) AS max_cand,
               sum(CASE WHEN c.n IS NULL OR c.n = 0 THEN 1 ELSE 0 END) AS zero_cand
        FROM '{raw_parquet_path("test", "S1").as_posix()}' s
        LEFT JOIN (SELECT s1_id, count(*) AS n FROM test_final_candidates GROUP BY s1_id) c ON c.s1_id = s.id
        GROUP BY s.country
    """).fetchall()
    for country, n_s1, mean_cand, median_cand, max_cand, zero_cand in rows:
        lines.append(f"  {country}: n_S1={n_s1}, mean={mean_cand:.1f}, median={median_cand:.0f}, "
                     f"max={max_cand}, zero-candidate S1s={zero_cand} ({100.0 * zero_cand / n_s1:.2f}%)")
    lines.append("\n(France has no training labels; a much higher zero-candidate or low-candidate rate "
                 "for France above than US/India would mean the mined dictionaries/keys, which are all "
                 "trained on US/India pairs, generalize poorly to French names/addresses.)")

    # write candidate_pairs.tsv (Part F output) using the same shape as matching_results.tsv
    ids_all = [r[0] for r in con.execute(f"SELECT id FROM '{raw_parquet_path('test', 'S1').as_posix()}'").fetchall()]
    cand_map = {s1_id: [] for s1_id in ids_all}
    for s1_id, match_id, match_source in con.execute(
        "SELECT s1_id, match_id, match_source FROM test_final_candidates ORDER BY s1_id, rank_in_s1"
    ).fetchall():
        cand_map[s1_id].append(f"{match_source}-{match_id}")
    str_cand_map = {f"S1-{k}": v for k, v in cand_map.items()}
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_id_list_tsv(OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids", str_cand_map)
    lines.append(f"\nWrote {OUTPUT_DIR / 'candidate_pairs.tsv'} ({len(str_cand_map)} rows).")

    return lines


def main():
    """Run Parts C-F: IDF, key generation, train-sample recall, test candidates."""
    print_sysinfo()
    con = connect(memory_limit_gb=4, threads=2)

    report = []
    for split in ("train", "test"):
        for source in ("S1", "S2", "S3"):
            norm_view(con, split, source)
        all_norm_union(con, split)
        s23_norm_union(con, split)
        with stage(f"compute IDF ({split})"):
            compute_idf(con, split)
        with stage(f"build token tables ({split})"):
            build_token_tables(con, split)

    report.append("=== Part E: recall measurement on 300k train S1 sample vs full train pools ===")
    e_lines, chosen_k = measure_recall(con)
    report.extend(e_lines)

    report.append("\n\n=== Part F: full test candidate generation ===")
    f_lines = write_test_candidates(con, chosen_k)
    report.extend(f_lines)

    report_text = "\n".join(report)
    print("\n" + report_text)
    BLOCKING_REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nSaved raw report text to {BLOCKING_REPORT_PATH}")

    con.close()


if __name__ == "__main__":
    main()
