"""Step 2/3 Part C/D/E/F: token IDF, blocking key generation, candidate
generation for ALL train S1 and ALL test S1, and recall measurement.

Everything here is DuckDB SQL over the normalized parquet from ``normalize.py``
-- no per-row Python loops over the full candidate space. Country is part of
every key (open-set string equality, never a fixed list).

Train candidates are generated for every train S1 record (not a sample): the
competition features downstream (how many S1 records compete for the same
S2/S3 record) must be computed over the same population size as test, or the
model trains on a different distribution than it predicts on.

Pipeline (see ``main()``):
  1. compute_idf          -- Part C: per (split, country) token document frequency
  2. build_keys_table     -- per-record top rare tokens/numbers -> key families K1-K7
  3. join + score_and_topk -- S1 keys x capped S2/S3 keys -> pairs, cheap score, top-K per S1
  4. measure_recall       -- Part E: recall on the stratified 300k train S1 sample
  5. write_test_candidates -- Part F: full test S1 vs full test S2/S3 pools

Block-size caps, top-K and batch sizes come from the machine resource profile
(``resources.py``): larger machines get larger caps (more recall).
"""

import random
import sys
import time

from config import (BLOCKING_REPORT_PATH, CAND_DIR, DICTS_DIR,
                     TEST_CANDIDATES_PARQUET, TRAIN_GT_LONG_PARQUET,
                     TRAIN_SAMPLE_S1_PARQUET, norm_parquet_path,
                     raw_parquet_path)
from db import connect
from perf import print_sysinfo, stage, progress_line
from resources import profile

random.seed(42)

# line_buffering=True: when stdout is redirected to a file (every background
# run in this project is), Python otherwise fully-buffers output and none of
# the per-country/per-batch progress prints below actually reach the file
# until the whole process exits -- which defeats the point of adding them.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TOP_NAME_TOKENS = 2
TOP_ADDR_WORDS = 2
TOP_NUMBERS = 3
# Block-size caps: an S2/S3 key shared by more than this many records is
# dropped (too generic to be informative, and the join explodes). 50/15 is
# what fits a 16 GB laptop; the profile raises them to 100/25 or 150/40 on
# bigger machines, which measurably recovers recall (the original cap=300
# found more true pairs before running out of memory).
CAP_DEFAULT = profile().cap_default
CAP_K2_NOSTATE = profile().cap_strict
STRICT_CAP_KEY_TYPES = ("K2_nostate", "K4_nostate", "K5_nostate", "K6")
TOP_K_PER_S1 = profile().top_k
MIN_NOSPACE_LEN = 3
# K7 only builds deletion variants of numbers at least this long: a 2-digit
# number's variants are single digits, which match nearly everything.
K7_MIN_NUM_LEN = 3


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
    """Compute per-(country, token) document frequency and IDF for name_core/addr_words/name_skeleton.

    name_skeleton gets its own independent IDF (Step 3 Track A5, key K5):
    rarity is measured in skeleton-space, not inherited from the name_core
    ranking, since a token that's common in word-space (e.g. a frequent
    legal-form remnant) may skeletonize to something rarer or vice versa.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE country_totals_{split} AS
        SELECT country, count(*) AS n FROM v_{split}_all_norm GROUP BY country
    """)
    for field, table in (("name_core", f"idf_name_{split}"), ("addr_words", f"idf_addr_{split}"),
                         ("name_skeleton", f"idf_skeleton_{split}")):
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
    # Step 3 Track A5, key K5: name_skeleton's own top rare tokens (ranked by
    # a skeleton-space IDF, not inherited from name_core -- see compute_idf).
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_skel AS
        SELECT source, id, list(token ORDER BY rn) AS top_skeleton_tokens FROM (
            SELECT source, id, token,
                   row_number() OVER (PARTITION BY source, id ORDER BY coalesce(i.idf, 0) DESC) AS rn
            FROM (
                SELECT source, id, country, unnest(string_split(name_skeleton, ' ')) AS token
                FROM {side_view} WHERE name_skeleton <> ''
            ) x
            LEFT JOIN idf_skeleton_{split} i USING (country, token)
        ) WHERE rn <= {TOP_NAME_TOKENS} GROUP BY source, id
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT
            s.source, s.id, s.country, s.state, s.name_alt, s.name_nospace,
            replace(s.name_alt, ' ', '') AS name_alt_nospace,
            na.top_name_tokens, aa.top_alt_tokens, ada.top_addr_words, nua.top_numbers,
            sk.top_skeleton_tokens
        FROM {side_view} s
        LEFT JOIN {out_table}_name na USING (source, id)
        LEFT JOIN {out_table}_alt aa USING (source, id)
        LEFT JOIN {out_table}_addr ada USING (source, id)
        LEFT JOIN {out_table}_nums nua USING (source, id)
        LEFT JOIN {out_table}_skel sk USING (source, id)
    """)
    for suffix in ("name", "alt", "addr", "nums", "skel"):
        con.execute(f"DROP TABLE IF EXISTS {out_table}_{suffix}")


def generate_key_rows(con, keys_table, out_table, side):
    """Explode the top-token/number LISTs into (id, source, country, key_type, key_value) rows.

    ``side`` is "s1" or "s23". Only K7 depends on it: K7 is a fuzzy house-
    number key (dropped digit, e.g. 1951<->195, 376<->76, A-192<->A-92) built
    SymSpell-style from single-digit deletions. To match "one side's number
    with a digit deleted" against "the other side's number as written" --
    and never exact-vs-exact, which K1 already covers -- the two variant
    kinds are tagged crosswise per side:

        S1 deletions  (K7d) == S2/S3 originals (tagged K7d)   S1 has the extra digit
        S1 originals  (K7o) == S2/S3 deletions (tagged K7o)   S2/S3 has the extra digit

    Joins match on key_type, so this crosswise tagging is all it takes.
    """
    orig_tag, del_tag = ("K7o", "K7d") if side == "s1" else ("K7d", "K7o")
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

        UNION ALL
        -- K5 (Step 3 Track A5): name_skeleton's own rare tokens, with state --
        -- survives transliteration/typo spelling variance that breaks K2/K3's
        -- exact-token matching (see normalize.py's skeleton_token).
        SELECT source, id, country, 'K5' AS key_type,
               country || '|' || state || '|' || tok AS key_value
        FROM {keys_table}, UNNEST(top_skeleton_tokens) AS t(tok)
        WHERE state <> '' AND tok <> ''

        UNION ALL
        SELECT source, id, country, 'K5_nostate' AS key_type,
               country || '|' || tok AS key_value
        FROM {keys_table}, UNNEST(top_skeleton_tokens) AS t(tok)
        WHERE state = '' AND tok <> ''

        UNION ALL
        -- K6 (Step 3 Track A5): sorted pair of the 2 rarest name_core tokens,
        -- no address needed at all -- covers entities with an empty/mismatched
        -- candidate address, which K1 (address-only) can never reach.
        SELECT source, id, country, 'K6' AS key_type,
               country || '|' || least(top_name_tokens[1], top_name_tokens[2])
                       || '|' || greatest(top_name_tokens[1], top_name_tokens[2]) AS key_value
        FROM {keys_table} WHERE len(top_name_tokens) >= 2

        UNION ALL
        -- K7 originals: (country, number as written, rare address word)
        SELECT source, id, country, '{orig_tag}' AS key_type,
               country || '|' || num || '|' || word AS key_value
        FROM {keys_table}, UNNEST(top_numbers) AS t1(num), UNNEST(top_addr_words) AS t2(word)

        UNION ALL
        -- K7 deletions: every single-digit deletion of numbers >= K7_MIN_NUM_LEN
        -- digits (leading zeros stripped, since numbers are stored as integers)
        SELECT DISTINCT source, id, country, '{del_tag}' AS key_type,
               country || '|' || v || '|' || word AS key_value
        FROM (
            SELECT source, id, country, word,
                   unnest(list_transform(range(1, length(num) + 1),
                          lambda i: ltrim(substr(num, 1, i - 1) || substr(num, i + 1), '0'))) AS v
            FROM {keys_table}, UNNEST(top_numbers) AS t1(num), UNNEST(top_addr_words) AS t2(word)
            WHERE length(num) >= {K7_MIN_NUM_LEN}
        ) WHERE v <> ''
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
        -- K6 (sorted rare-token-pair, no address) also gets the stricter cap:
        -- Step 3 Track A6 measurement showed it alone reaches 44.9% recall,
        -- meaning it produces large, redundant blocks in dense countries
        -- (India); that same leakiness pushed a full test run past a 116GB
        -- temp-disk limit even with per-batch chunking.
        WHERE c.n <= CASE WHEN k.key_type IN {STRICT_CAP_KEY_TYPES} THEN {CAP_K2_NOSTATE} ELSE {CAP_DEFAULT} END
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
    # The top-K cut happens HERE, before anything downstream: every pair kept
    # is featurized and scored by the model, so an uncut table would make
    # feature volume grow with the blocking caps instead of with top_k.
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT * FROM (
            SELECT p.s1_id, p.match_id, p.match_source, p.n_keys_hit,
                   (p.n_keys_hit + coalesce(sn.shared_name_idf, 0) + coalesce(sa.shared_addr_idf, 0)) AS cheap_score,
                   row_number() OVER (PARTITION BY p.s1_id ORDER BY
                       (p.n_keys_hit + coalesce(sn.shared_name_idf, 0) + coalesce(sa.shared_addr_idf, 0)) DESC,
                       p.match_source, p.match_id) AS rank_in_s1
            FROM {out_table}_agg p
            LEFT JOIN {out_table}_shared_name sn USING (s1_id, match_id, match_source)
            LEFT JOIN {out_table}_shared_addr sa USING (s1_id, match_id, match_source)
        ) WHERE rank_in_s1 <= {top_k}
    """)
    for t in ("agg", "shared_name", "shared_addr"):
        con.execute(f"DROP TABLE IF EXISTS {out_table}_{t}")


# S1 rows per blocking batch, sized from the profile (75k was the measured
# safe size at 4 GB of DuckDB memory with caps 50/15).
S1_BATCH_SIZE = profile().s1_batch_size


S23_KEY_CHUNK = S1_BATCH_SIZE * 4
KEY_SOURCE_COLS = ("source, id, country, state, name_core, name_alt, name_nospace, "
                   "addr_words, numbers, name_skeleton")


def build_s23_keys(con, split, country_view, out_table, i):
    """S2/S3 key rows for one country, built in record chunks (keys depend only on the record + IDF).

    Ranking every record's tokens in one query needs a window sort over all
    of a country's S2/S3 tokens at once, which ran out of memory for the US
    pool at ~2 GB; per-chunk it stays small on any machine.
    """
    base = f"{split}_s23_base_c{i}"
    con.execute(f"CREATE OR REPLACE TABLE {base} AS SELECT {KEY_SOURCE_COLS} FROM {country_view}")
    n = con.execute(f"SELECT count(*) FROM {base}").fetchone()[0]
    n_chunks = max(1, -(-n // S23_KEY_CHUNK))
    parts = []
    for c in range(n_chunks):
        con.execute(f"CREATE OR REPLACE VIEW v_{split}_s23_chunk AS "
                    f"SELECT * FROM {base} WHERE hash(id, source) % {n_chunks} = {c}")
        raw, keys = f"{out_table}_raw_{c}", f"{out_table}_part_{c}"
        build_keys_table(con, split, f"v_{split}_s23_chunk", raw)
        generate_key_rows(con, raw, keys, side="s23")
        con.execute(f"DROP TABLE IF EXISTS {raw}")
        parts.append(keys)
    con.execute(f"CREATE OR REPLACE TABLE {out_table} AS "
                + " UNION ALL ".join(f"SELECT * FROM {t}" for t in parts))
    for t in parts + [base]:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    print(f"  S2/S3 keys for this country: {n} records in {n_chunks} chunk(s)")


def build_all_keys(con, split, s1_view, prefix, keep_raw_filter=None):
    """Build keys, join, and score candidate pairs -- one (country, S1 batch) at a time.

    Returns the name of the final top-K scored table ``{prefix}_scored``
    (union of every country/batch). The raw per-key hits are only needed for
    the per-key-type recall breakdown, so they are kept (as
    ``{prefix}_pairs_raw``) only for S1 rows matching ``keep_raw_filter`` (a
    SQL condition on ``s1_id``), and dropped otherwise -- for all 2.2M train S1
    they would be hundreds of millions of rows.

    The S2/S3 side is built ONCE per country and reused across that country's
    S1 batches. The S1 side is split into ``S1_BATCH_SIZE``-row batches within
    each country, which bounds every join to roughly the same size no matter
    how large one country's population is. Blocking keys always include the
    country and are computed per record, so results never depend on how S1 is
    chunked.
    """
    countries = [r[0] for r in con.execute(f"SELECT DISTINCT country FROM v_{split}_S23_norm").fetchall()]
    scored_parts = []
    pairs_raw_parts = []

    # Cheap pre-pass so batch progress can report a real overall percentage,
    # not just position-within-country (which by itself can't tell you if
    # you're 5% or 90% through the whole run).
    country_row_counts = {}
    for country in countries:
        safe_country = country.replace("'", "''")
        country_row_counts[country] = con.execute(
            f"SELECT count(*) FROM {s1_view} WHERE country = '{safe_country}'"
        ).fetchone()[0]
    total_batches = sum(max(1, -(-n // S1_BATCH_SIZE)) for n in country_row_counts.values())
    batches_done = 0
    run_t0 = time.perf_counter()

    print(f"  [{prefix}] {len(countries)} countries to process: {countries} "
          f"({total_batches} total batches across all countries)")
    for i, country in enumerate(countries):
        country_t0 = time.perf_counter()
        print(f"  [{prefix}] country {i + 1}/{len(countries)} ({country}): start")
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
        s23_keys, s23_keys_capped = f"{split}_s23_keys_c{i}", f"{split}_s23_keys_capped_c{i}"
        build_s23_keys(con, split, f"v_{split}_S23_norm_part", s23_keys, i)
        cap_s23_keys(con, s23_keys, s23_keys_capped)
        con.execute(f"DROP TABLE IF EXISTS {s23_keys}")

        n_country = con.execute(f"SELECT count(*) FROM v_{split}_s1_country").fetchone()[0]
        n_batches = max(1, -(-n_country // S1_BATCH_SIZE))  # ceil division
        numbered = f"{prefix}_s1_numbered_c{i}"
        con.execute(f"""
            CREATE OR REPLACE TABLE {numbered} AS
            SELECT *, (row_number() OVER (ORDER BY id) - 1) // {S1_BATCH_SIZE} AS _batch
            FROM v_{split}_s1_country
        """)
        print(f"  [{prefix}] country {i + 1}/{len(countries)} ({country}): {n_country} S1 rows, "
              f"{n_batches} batch(es) of up to {S1_BATCH_SIZE}")
        for b in range(n_batches):
            batch_t0 = time.perf_counter()
            con.execute(
                f"CREATE OR REPLACE VIEW v_{split}_s1_batch AS "
                f"SELECT * EXCLUDE (_batch) FROM {numbered} WHERE _batch = {b}"
            )
            s1_keys_raw, s1_keys = f"{prefix}_s1_keys_raw_c{i}_b{b}", f"{prefix}_s1_keys_c{i}_b{b}"
            build_keys_table(con, split, f"v_{split}_s1_batch", s1_keys_raw)
            generate_key_rows(con, s1_keys_raw, s1_keys, side="s1")

            pairs_raw = f"{prefix}_pairs_raw_c{i}_b{b}"
            join_candidate_pairs(con, s1_keys, s23_keys_capped, pairs_raw)
            scored_part = f"{prefix}_scored_c{i}_b{b}"
            score_and_topk(con, split, pairs_raw, scored_part)
            scored_parts.append(scored_part)
            if keep_raw_filter:
                kept = f"{pairs_raw}_kept"
                con.execute(f"CREATE OR REPLACE TABLE {kept} AS SELECT * FROM {pairs_raw} WHERE {keep_raw_filter}")
                pairs_raw_parts.append(kept)
            con.execute(f"DROP TABLE IF EXISTS {pairs_raw}")

            con.execute(f"DROP TABLE IF EXISTS {s1_keys_raw}")
            con.execute(f"DROP TABLE IF EXISTS {s1_keys}")
            batches_done += 1
            batch_dt = time.perf_counter() - batch_t0
            print(f"  [{prefix}] country {i + 1}/{len(countries)} ({country}), "
                  f"batch {b + 1}/{n_batches}: done in {batch_dt:.1f}s -- "
                  f"{progress_line('overall', batches_done, total_batches, run_t0)}")

        con.execute(f"DROP TABLE IF EXISTS {s23_keys_capped}")
        con.execute(f"DROP TABLE IF EXISTS {numbered}")
        print(f"  [{prefix}] country {i + 1}/{len(countries)} ({country}): "
              f"ALL batches done in {time.perf_counter() - country_t0:.1f}s")

    if pairs_raw_parts:
        union_sql = " UNION ALL ".join(f"SELECT * FROM {t}" for t in pairs_raw_parts)
        con.execute(f"CREATE OR REPLACE TABLE {prefix}_pairs_raw AS {union_sql}")
        for t in pairs_raw_parts:
            con.execute(f"DROP TABLE IF EXISTS {t}")

    union_sql = " UNION ALL ".join(f"SELECT * FROM {t}" for t in scored_parts)
    con.execute(f"CREATE OR REPLACE TABLE {prefix}_scored AS {union_sql}")
    for t in scored_parts:
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return f"{prefix}_scored"


RECALL_K_SWEEP = tuple(k for k in (5, 10, 20, 30, 40, 50) if k <= TOP_K_PER_S1)
RECALL_TARGET = 0.97
SAMPLE_FILTER = f"s1_id IN (SELECT id FROM '{TRAIN_SAMPLE_S1_PARQUET.as_posix()}')"


def build_train_candidates(con):
    """Block ALL train S1 against the full train S2/S3 pools (raw key hits kept for the recall sample)."""
    with stage("Part E: build+score all train-S1 vs full train pools"):
        return build_all_keys(con, "train", "v_train_S1_norm", "train", keep_raw_filter=SAMPLE_FILTER)


def measure_recall(con):
    """Part E: measure blocking recall on the 300k train S1 sample (candidates from the full train run).

    Returns (report_lines, chosen_k) where ``report_lines`` is ready to drop
    into BLOCKING_REPORT.md and ``chosen_k`` is the smallest swept K reaching
    RECALL_TARGET union recall (or the largest swept K, if none reach it).
    """
    sample_path = TRAIN_SAMPLE_S1_PARQUET.as_posix()
    gt_path = TRAIN_GT_LONG_PARQUET.as_posix()
    lines = [f"Caps: default={CAP_DEFAULT}, strict={CAP_K2_NOSTATE} {STRICT_CAP_KEY_TYPES}; "
             f"top_k per S1={TOP_K_PER_S1}; S1 batch={S1_BATCH_SIZE}"]

    con.execute(f"""
        CREATE OR REPLACE VIEW v_train_S1_sample_norm AS
        SELECT n.* FROM v_train_S1_norm n JOIN '{sample_path}' s ON n.id = s.id
    """)
    con.execute(f"CREATE OR REPLACE VIEW trainsample_scored AS SELECT * FROM train_scored WHERE {SAMPLE_FILTER}")
    con.execute("CREATE OR REPLACE VIEW trainsample_pairs_raw AS SELECT * FROM train_pairs_raw")
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

    scored_table = "trainsample_scored"

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

def write_test_candidates(con):
    """Part F: block all test S1 against the full test S2/S3 pools; export the top-K candidates."""
    lines = []
    with stage("Part F: build+score all test-S1 vs full test pools"):
        scored_table = build_all_keys(con, "test", "v_test_S1_norm", "test")

    CAND_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT s1_id, match_id, match_source, n_keys_hit, cheap_score, rank_in_s1 "
                f"FROM {scored_table}) TO '{TEST_CANDIDATES_PARQUET.as_posix()}' (FORMAT PARQUET)")

    total_s1 = con.execute(f"SELECT count(*) FROM '{raw_parquet_path('test', 'S1').as_posix()}'").fetchone()[0]
    s1_with_cands = con.execute(f"SELECT count(DISTINCT s1_id) FROM {scored_table}").fetchone()[0]
    n_pairs = con.execute(f"SELECT count(*) FROM {scored_table}").fetchone()[0]
    lines.append(f"Test S1 entities: {total_s1}; with >=1 candidate: {s1_with_cands} "
                 f"({100.0 * s1_with_cands / total_s1:.2f}%); with 0 candidates: {total_s1 - s1_with_cands}; "
                 f"candidate pairs (top-{TOP_K_PER_S1}): {n_pairs}")

    lines.append("\nCandidates per S1 by country:")
    rows = con.execute(f"""
        SELECT s.country, count(*) AS n_s1, avg(coalesce(c.n, 0)) AS mean_cand,
               median(coalesce(c.n, 0)) AS median_cand, max(coalesce(c.n, 0)) AS max_cand,
               sum(CASE WHEN c.n IS NULL OR c.n = 0 THEN 1 ELSE 0 END) AS zero_cand
        FROM '{raw_parquet_path("test", "S1").as_posix()}' s
        LEFT JOIN (SELECT s1_id, count(*) AS n FROM {scored_table} GROUP BY s1_id) c ON c.s1_id = s.id
        GROUP BY s.country
    """).fetchall()
    for country, n_s1, mean_cand, median_cand, max_cand, zero_cand in rows:
        lines.append(f"  {country}: n_S1={n_s1}, mean={mean_cand:.1f}, median={median_cand:.0f}, "
                     f"max={max_cand}, zero-candidate S1s={zero_cand} ({100.0 * zero_cand / n_s1:.2f}%)")
    lines.append("\n(France has no training labels; a much higher zero-candidate or low-candidate rate "
                 "for France above than US/India would mean the mined dictionaries/keys, which are all "
                 "trained on US/India pairs, generalize poorly to French names/addresses.)")
    lines.append("(output/candidate_pairs.tsv is written by train_model.py together with "
                 "matching_results.tsv, from exactly the pairs the model scores.)")
    return lines


def prepare_split(con, split):
    """Views, IDF and token tables one split's blocking needs."""
    for source in ("S1", "S2", "S3"):
        norm_view(con, split, source)
    all_norm_union(con, split)
    s23_norm_union(con, split)
    with stage(f"compute IDF ({split})"):
        compute_idf(con, split)
    with stage(f"build token tables ({split})"):
        build_token_tables(con, split)


def main():
    """Run Parts C-F: IDF, key generation, all-train-S1 candidates + recall, test candidates.

    CLI arg (optional):
      train-only  -- train candidates + recall report only (cheap check of a blocking change)
      test-only   -- test candidates only (resume after a finished train pass)
    """
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg not in ("", "train-only", "test-only"):
        raise SystemExit(f"unknown arg {arg!r}; expected nothing, 'train-only' or 'test-only'")
    do_train, do_test = arg != "test-only", arg != "train-only"

    print_sysinfo()
    print(f"[blocking] caps default={CAP_DEFAULT} strict={CAP_K2_NOSTATE}, top_k={TOP_K_PER_S1}, "
          f"S1 batch={S1_BATCH_SIZE}")
    con = connect(role="heavy")
    report = []

    if do_train:
        prepare_split(con, "train")
        build_train_candidates(con)
        report.append("=== Part E: recall on the 300k train S1 sample (candidates from all-train-S1 blocking) ===")
        e_lines, _ = measure_recall(con)
        report.extend(e_lines)

    if do_test:
        prepare_split(con, "test")
        report.append("\n\n=== Part F: full test candidate generation ===")
        report.extend(write_test_candidates(con))

    report_text = "\n".join(report)
    print("\n" + report_text)
    if arg == "":
        out_path = BLOCKING_REPORT_PATH
    else:
        out_path = BLOCKING_REPORT_PATH.with_name(f"BLOCKING_REPORT_{arg.replace('-', '_')}.md")
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved raw report text to {out_path}")

    con.close()


if __name__ == "__main__":
    main()
