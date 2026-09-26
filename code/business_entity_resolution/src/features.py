"""Track 4 Phase 1: pairwise name/address similarity features for entity resolution.

``compute_pair_features`` is the reusable public entry point: given a list (or
DataFrame) of raw-text candidate pairs, it normalizes both sides with
``normalize.normalize_name``/``normalize_addr`` and returns one feature row per
input pair. It is deliberately independent of where the pairs came from, so it
can be reused unchanged once real blocking candidates exist (Phase 2).

The ``if __name__ == "__main__"`` block below builds a labeled Phase 1
sanity-check set: every train ground-truth positive pair (``train_gt_joined``)
plus an equal-sized set of random same-country negative pairs, computes
features for both, prints a positive-vs-negative separation report, and saves
the labeled table to ``data/features/train_gt_vs_negatives.parquet``.

Design notes baked into the feature functions:

* "Where relevant" (the rapidfuzz/Jaro-Winkler/nospace/sorted-chars/IDF-Jaccard
  name metrics), each side contributes up to two name variants -- the
  wrapper-stripped ``name_main`` and, when a wrapper separator was found, the
  wrapped-out ``name_alt`` (re-run through ``normalize_name`` on its own so it
  gets its own core/nospace/sorted_chars) -- and the score is the max over the
  four (variant_a, variant_b) combinations, since the "real" name can end up
  on either side of a wrapper. The IDF-Jaccard/rarest-token trio picks the
  single combination that maximizes the Jaccard score and reports its shared-
  and unshared-rarest-token IDF together, so the three numbers describe one
  consistent name pairing rather than three independently cherry-picked ones.
* legal_form and name length use only the primary (main) normalized name --
  these are per-record properties, not pairwise similarity scores, so a
  variant max does not apply.
* Every similarity/statistic that cannot be computed (empty field on either
  side, missing IDF table, no shared tokens union) is NaN, never 0 -- 0 is a
  valid similarity score.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

import normalize
from config import DICTS_DIR, FEATURES_DIR, RAW_DIR, raw_parquet_path
from db import connect
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8")

WORKERS = 2  # conservative: this machine is often down to a few GB free RAM
POOL_CHUNKSIZE = 2000
BATCH_ROWS = 100_000
NEG_SEED = 42

IDF_NAME_TRAIN_PARQUET = DICTS_DIR / "idf_name_train.parquet"
IDF_ADDR_TRAIN_PARQUET = DICTS_DIR / "idf_addr_train.parquet"

_W = {}
_WARNED = {"name": False, "addr": False}


def _load_idf_map(path):
    """Load an idf_{name,addr}_train.parquet file into a {(country, token): idf} dict."""
    df = pd.read_parquet(path, columns=["country", "token", "idf"])
    return dict(zip(zip(df["country"], df["token"]), df["idf"]))


def _init_pool(dicts_dir, idf_name_path, idf_addr_path):
    """Multiprocessing pool initializer: load normalize's dicts and the IDF maps once per worker."""
    normalize._init_worker(dicts_dir)
    _W["idf_name"] = _load_idf_map(idf_name_path) if idf_name_path else None
    _W["idf_addr"] = _load_idf_map(idf_addr_path) if idf_addr_path else None


def _resolve_idf_paths(idf_name_path, idf_addr_path):
    """Return (name_path_or_None, addr_path_or_None), warning once per process if a file is absent."""
    name_path = Path(idf_name_path) if idf_name_path else IDF_NAME_TRAIN_PARQUET
    addr_path = Path(idf_addr_path) if idf_addr_path else IDF_ADDR_TRAIN_PARQUET
    name_ok, addr_ok = name_path.exists(), addr_path.exists()
    if not name_ok and not _WARNED["name"]:
        print(f"[features] WARNING: {name_path} not found -- name IDF features will be NaN.")
        _WARNED["name"] = True
    if not addr_ok and not _WARNED["addr"]:
        print(f"[features] WARNING: {addr_path} not found -- address IDF features will be NaN.")
        _WARNED["addr"] = True
    return (str(name_path) if name_ok else None, str(addr_path) if addr_ok else None)


def _is_blank(text):
    """Return True if a raw text field is missing or blank after stripping."""
    if text is None:
        return True
    if isinstance(text, float) and pd.isna(text):
        return True
    return str(text).strip() == ""


def _parse_numbers(numbers_str):
    """Parse a normalize_addr ``numbers`` comma-joined string into a set of ints ('' -> empty set)."""
    return {int(x) for x in numbers_str.split(",")} if numbers_str else set()


def _name_variants(name_dict):
    """Return [main_variant] plus an [alt_variant] (own core/nospace/sorted_chars) when a wrapper exists."""
    variants = [{
        "core": name_dict["name_core"],
        "nospace": name_dict["name_nospace"],
        "sorted_chars": name_dict["name_sorted_chars"],
    }]
    alt = name_dict.get("name_alt", "")
    if alt:
        alt_dict = normalize.normalize_name(alt)
        variants.append({
            "core": alt_dict["name_core"],
            "nospace": alt_dict["name_nospace"],
            "sorted_chars": alt_dict["name_sorted_chars"],
        })
    return variants


def _idf_jaccard_and_rarest(tokens_a, tokens_b, idf_map, country):
    """IDF-weighted Jaccard of two token lists, plus the rarest shared/unshared token IDF.

    Returns (jaccard, rarest_shared_idf, rarest_unshared_idf), each NaN if not
    computable (empty side, empty union, or no idf_map available). Unknown
    tokens (not in ``idf_map`` for this country) get idf 0.0.
    """
    if not tokens_a or not tokens_b or idf_map is None:
        return np.nan, np.nan, np.nan
    set_a, set_b = set(tokens_a), set(tokens_b)
    union = set_a | set_b
    if not union:
        return np.nan, np.nan, np.nan
    inter = set_a & set_b

    def idf(tok):
        return idf_map.get((country, tok), 0.0)

    w_inter = sum(idf(t) for t in inter)
    w_union = sum(idf(t) for t in union)
    jaccard = w_inter / w_union if w_union > 0 else np.nan
    rarest_shared = max((idf(t) for t in inter), default=np.nan)
    unshared = union - inter
    rarest_unshared = max((idf(t) for t in unshared), default=np.nan)
    return jaccard, rarest_shared, rarest_unshared


def _name_features(s1_n, o_n, country, idf_map):
    """Compute all name-side features for one pair from their normalize_name() dicts."""
    variants_a = _name_variants(s1_n)
    variants_b = _name_variants(o_n)

    ratio = partial = tsort = tset = jw = -1.0
    nospace_ratio = nospace_contains = sorted_ratio = -1.0
    best_jaccard, best_shared_idf, best_unshared_idf = -1.0, np.nan, np.nan

    for va in variants_a:
        for vb in variants_b:
            if va["core"] and vb["core"]:
                ratio = max(ratio, fuzz.ratio(va["core"], vb["core"]))
                partial = max(partial, fuzz.partial_ratio(va["core"], vb["core"]))
                tsort = max(tsort, fuzz.token_sort_ratio(va["core"], vb["core"]))
                tset = max(tset, fuzz.token_set_ratio(va["core"], vb["core"]))
                jw = max(jw, JaroWinkler.similarity(va["core"], vb["core"]))

                jac, shared_idf, unshared_idf = _idf_jaccard_and_rarest(
                    va["core"].split(), vb["core"].split(), idf_map, country)
                if not np.isnan(jac) and jac > best_jaccard:
                    best_jaccard, best_shared_idf, best_unshared_idf = jac, shared_idf, unshared_idf

            if va["nospace"] and vb["nospace"]:
                nospace_ratio = max(nospace_ratio, fuzz.ratio(va["nospace"], vb["nospace"]))
                contains = 1.0 if (va["nospace"] in vb["nospace"] or vb["nospace"] in va["nospace"]) else 0.0
                nospace_contains = max(nospace_contains, contains)

            if va["sorted_chars"] and vb["sorted_chars"]:
                sorted_ratio = max(sorted_ratio, fuzz.ratio(va["sorted_chars"], vb["sorted_chars"]))

    def _clean(v):
        return np.nan if v == -1.0 else v

    legal_a, legal_b = s1_n.get("legal_form", ""), o_n.get("legal_form", "")
    if not legal_a or not legal_b:
        legal_form_match = "missing"
    else:
        legal_form_match = "same" if legal_a == legal_b else "different"

    len_a, len_b = len(s1_n.get("name_clean", "")), len(o_n.get("name_clean", ""))

    return {
        "name_ratio": _clean(ratio),
        "name_partial_ratio": _clean(partial),
        "name_token_sort_ratio": _clean(tsort),
        "name_token_set_ratio": _clean(tset),
        "name_jaro_winkler": _clean(jw),
        "name_nospace_ratio": _clean(nospace_ratio),
        "name_nospace_contains": _clean(nospace_contains),
        "name_sorted_chars_ratio": _clean(sorted_ratio),
        "name_idf_jaccard": _clean(best_jaccard),
        "name_idf_rarest_shared": best_shared_idf,
        "name_idf_rarest_unshared": best_unshared_idf,
        "legal_form_match": legal_form_match,
        "s1_name_len": len_a,
        "other_name_len": len_b,
        "name_len_absdiff": abs(len_a - len_b),
    }


def _addr_features(s1_a, o_a, other_addr_raw, country, idf_map):
    """Compute all address-side features for one pair from their normalize_addr() dicts."""
    addr_a, addr_b = s1_a["addr_clean"], o_a["addr_clean"]
    if addr_a and addr_b:
        tset = fuzz.token_set_ratio(addr_a, addr_b)
        tsort = fuzz.token_sort_ratio(addr_a, addr_b)
    else:
        tset = tsort = np.nan

    nums_a, nums_b = _parse_numbers(s1_a["numbers"]), _parse_numbers(o_a["numbers"])
    if nums_a and nums_b:
        inter, union = nums_a & nums_b, nums_a | nums_b
        num_jaccard = len(inter) / len(union) if union else np.nan
        num_shared_count = len(inter)
        num_max_equal = 1.0 if max(nums_a) == max(nums_b) else 0.0
    else:
        num_jaccard = num_shared_count = num_max_equal = np.nan

    state_a, state_b = s1_a["state"], o_a["state"]
    if not state_a or not state_b:
        state_match = "missing"
    else:
        state_match = "same" if state_a == state_b else "different"

    words_a = s1_a["addr_words"].split() if s1_a["addr_words"] else []
    words_b = o_a["addr_words"].split() if o_a["addr_words"] else []
    idf_jaccard, _, _ = _idf_jaccard_and_rarest(words_a, words_b, idf_map, country)

    return {
        "addr_token_set_ratio": tset,
        "addr_token_sort_ratio": tsort,
        "addr_num_jaccard": num_jaccard,
        "addr_num_max_equal": num_max_equal,
        "addr_num_shared_count": num_shared_count,
        "addr_state_match": state_match,
        "addr_idf_jaccard": idf_jaccard,
        "other_addr_empty": 1.0 if _is_blank(other_addr_raw) else 0.0,
    }


def _compute_one(pair):
    """Normalize both sides of one raw-text pair dict and compute its full feature row."""
    s1_name = pair.get("s1_name") or ""
    s1_addr = pair.get("s1_addr") or ""
    s1_country = pair.get("s1_country") or ""
    other_name = pair.get("other_name") or ""
    other_addr = pair.get("other_addr") or ""
    other_country = pair.get("other_country") or ""
    other_source = pair.get("other_source") or ""

    s1_n = normalize.normalize_name(s1_name)
    o_n = normalize.normalize_name(other_name)
    s1_a = normalize.normalize_addr(s1_addr, s1_country)
    o_a = normalize.normalize_addr(other_addr, other_country)

    country = s1_country or other_country
    feats = {}
    feats.update(_name_features(s1_n, o_n, country, _W.get("idf_name")))
    feats.update(_addr_features(s1_a, o_a, other_addr, country, _W.get("idf_addr")))
    feats["other_source"] = other_source
    return feats


def make_pool(workers=WORKERS, dicts_dir=None, idf_name_path=None, idf_addr_path=None):
    """Build a multiprocessing.Pool pre-initialized with normalize's dicts and the IDF maps.

    Callers that process many batches (see __main__ below) should build one
    pool up front with this and pass it to every ``compute_pair_features``
    call, rather than letting a Pool be spawned per batch -- matching
    normalize.py's own one-pool-for-the-whole-run pattern. Re-spawning a fresh
    Pool per batch is both slow (process startup cost) and, empirically on
    this machine under memory pressure, occasionally left a worker's
    ``normalize._G`` uninitialized before its first task ran.
    """
    dicts_dir = str(dicts_dir) if dicts_dir else str(DICTS_DIR)
    name_path, addr_path = _resolve_idf_paths(idf_name_path, idf_addr_path)
    return Pool(processes=workers, initializer=_init_pool, initargs=(dicts_dir, name_path, addr_path))


def compute_pair_features(pairs, pool=None, workers=WORKERS, chunksize=POOL_CHUNKSIZE,
                           dicts_dir=None, idf_name_path=None, idf_addr_path=None):
    """Compute name/address similarity features for a batch of raw-text candidate pairs.

    ``pairs`` is a list of dicts, or a pandas DataFrame, with at minimum the
    columns/keys ``s1_name, s1_addr, s1_country, other_name, other_addr,
    other_country, other_source``. Any extra columns (ids, labels, ...) are
    ignored. Returns a DataFrame with one feature row per input pair, in the
    same order, holding the name/address features documented in this module's
    docstring plus a passthrough ``other_source`` column.

    Pass an existing ``pool`` (from ``make_pool``) to reuse workers across
    many calls; otherwise a one-off Pool is created and torn down just for
    this call.
    """
    if isinstance(pairs, pd.DataFrame):
        records = pairs.to_dict("records")
    else:
        records = list(pairs)
    if not records:
        return pd.DataFrame()

    if pool is not None:
        results = pool.map(_compute_one, records, chunksize=chunksize)
        return pd.DataFrame(results)

    with make_pool(workers, dicts_dir, idf_name_path, idf_addr_path) as p:
        results = p.map(_compute_one, records, chunksize=chunksize)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# __main__: positives (train ground truth) vs. random same-country negatives
# ---------------------------------------------------------------------------

def _positive_pairs_query():
    """SQL selecting every train positive pair from train_gt_joined.parquet, renamed to the pair schema."""
    path = (RAW_DIR / "train_gt_joined.parquet").as_posix()
    return f"""
        SELECT s1_id, match_id AS other_id, match_source AS other_source,
               s1_name, s1_addr, s1_country,
               match_name AS other_name, match_addr AS other_addr, match_country AS other_country
        FROM '{path}'
    """


def _build_negative_view(con):
    """Create the _neg_raw table: one random same-country, non-matching S2/S3 record per positive pair.

    For each positive pair, a deterministic (seed-42) uniformly shuffled
    row-number of the same country's S2 or S3 pool (matching that pair's
    match_source) is picked via a plain equi-join on (country, shuffled rank).
    Since each stratum's positive count is well below its pool size, this
    assigns each positive row a distinct pool record (no collisions besides
    the rare case where the draw equals a true match, which is anti-joined
    out against the *full* ground truth -- an s1_id can have several true
    matches, not just the one on the positive row the draw was keyed to).
    """
    gt_path = (RAW_DIR / "train_gt_joined.parquet").as_posix()
    s2_path = raw_parquet_path("train", "S2").as_posix()
    s3_path = raw_parquet_path("train", "S3").as_posix()
    con.execute(f"SELECT setseed({(NEG_SEED % 100) / 100.0})")

    con.execute(f"""
        CREATE OR REPLACE TABLE _pool_counts AS
        SELECT 'S2' AS source, country, count(*) AS pool_n FROM '{s2_path}' GROUP BY country
        UNION ALL
        SELECT 'S3' AS source, country, count(*) AS pool_n FROM '{s3_path}' GROUP BY country
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE _pos_rn AS
        SELECT g.s1_id, g.match_id AS true_match_id, g.match_source,
               g.s1_name, g.s1_addr, g.s1_country,
               row_number() OVER (PARTITION BY g.s1_country, g.match_source
                                   ORDER BY g.s1_id, g.match_id) AS rn,
               pc.pool_n
        FROM '{gt_path}' g
        JOIN _pool_counts pc ON pc.source = g.match_source AND pc.country = g.s1_country
    """)
    for source, path in (("S2", s2_path), ("S3", s3_path)):
        con.execute(f"""
            CREATE OR REPLACE TABLE _pool_{source} AS
            SELECT id, business_name, business_address, country,
                   row_number() OVER (PARTITION BY country ORDER BY random()) AS prn
            FROM '{path}'
        """)
    con.execute(f"""
        CREATE OR REPLACE TABLE _neg_raw AS
        SELECT p.s1_id, pl.id AS other_id, p.match_source AS other_source,
               p.s1_name, p.s1_addr, p.s1_country,
               pl.business_name AS other_name, pl.business_address AS other_addr, pl.country AS other_country
        FROM _pos_rn p JOIN _pool_S2 pl
            ON p.match_source = 'S2' AND p.s1_country = pl.country
               AND pl.prn = ((p.rn - 1) % p.pool_n) + 1
        WHERE NOT EXISTS (
            SELECT 1 FROM '{gt_path}' g2
            WHERE g2.s1_id = p.s1_id AND g2.match_id = pl.id AND g2.match_source = 'S2'
        )
        UNION ALL
        SELECT p.s1_id, pl.id AS other_id, p.match_source AS other_source,
               p.s1_name, p.s1_addr, p.s1_country,
               pl.business_name AS other_name, pl.business_address AS other_addr, pl.country AS other_country
        FROM _pos_rn p JOIN _pool_S3 pl
            ON p.match_source = 'S3' AND p.s1_country = pl.country
               AND pl.prn = ((p.rn - 1) % p.pool_n) + 1
        WHERE NOT EXISTS (
            SELECT 1 FROM '{gt_path}' g2
            WHERE g2.s1_id = p.s1_id AND g2.match_id = pl.id AND g2.match_source = 'S3'
        )
    """)


def _write_labeled_batches(con, query, label, n_expected, pool, writer_state):
    """Stream ``query`` in BATCH_ROWS chunks, featurize each with ``pool``, and append to the shared writer.

    ``writer_state`` is a ``{"writer": pq.ParquetWriter|None, "path": Path}``
    dict shared across positive/negative calls so all output goes to one
    growing parquet file -- written incrementally (like normalize.py's own
    per-batch ParquetWriter use) so a crash partway through does not lose
    already-computed batches, and so the full labeled table never has to be
    held in Python memory at once.
    """
    result = con.execute(query)
    n_seen = 0
    for batch in result.to_arrow_reader(BATCH_ROWS):
        df_batch = batch.to_pandas()
        feats = compute_pair_features(df_batch, pool=pool)
        feats["label"] = label
        feats["s1_id"] = df_batch["s1_id"].values
        feats["other_id"] = df_batch["other_id"].values
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer_state["writer"] is None:
            writer_state["writer"] = pq.ParquetWriter(str(writer_state["path"]), table.schema)
        writer_state["writer"].write_table(table)
        n_seen += len(df_batch)
        print(f"  label={label}: {n_seen}/{n_expected} pairs featurized")


NUMERIC_FEATURE_COLS = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaro_winkler", "name_nospace_ratio", "name_nospace_contains", "name_sorted_chars_ratio",
    "name_idf_jaccard", "name_idf_rarest_shared", "name_idf_rarest_unshared",
    "s1_name_len", "other_name_len", "name_len_absdiff",
    "addr_token_set_ratio", "addr_token_sort_ratio", "addr_num_jaccard", "addr_num_max_equal",
    "addr_num_shared_count", "addr_idf_jaccard", "other_addr_empty",
]
CATEGORICAL_FEATURE_COLS = ["legal_form_match", "addr_state_match", "other_source"]


def print_separation_report(parquet_path):
    """Print positive-vs-negative mean/std/NaN-rate for numeric features and value-share for categoricals.

    Aggregates directly from the saved parquet with DuckDB (never
    materializes the full labeled table in Python). Flags any numeric feature
    whose standardized gap ``|mean_pos - mean_neg| / pooled_std`` is below 0.2
    (weak separation) or whose NaN rate exceeds 50% in either class.
    """
    path = Path(parquet_path).as_posix()
    con = connect(memory_limit_gb=2, threads=2)
    n_pos, n_neg = con.execute(
        f"SELECT sum((label=1)::INT), sum((label=0)::INT) FROM '{path}'"
    ).fetchone()

    print(f"\n{'feature':<26}{'pos_mean':>10}{'pos_std':>10}{'neg_mean':>10}{'neg_std':>10}"
          f"{'nan%_pos':>10}{'nan%_neg':>10}  flag")
    flags = []
    for col in NUMERIC_FEATURE_COLS:
        p_mean, p_std, n_mean, n_std, p_nan, n_nan = con.execute(f"""
            SELECT
                avg(CASE WHEN label = 1 THEN {col} END),
                stddev(CASE WHEN label = 1 THEN {col} END),
                avg(CASE WHEN label = 0 THEN {col} END),
                stddev(CASE WHEN label = 0 THEN {col} END),
                100.0 * avg(CASE WHEN label = 1 THEN ({col} IS NULL)::INT END),
                100.0 * avg(CASE WHEN label = 0 THEN ({col} IS NULL)::INT END)
            FROM '{path}'
        """).fetchone()
        p_mean, p_std = p_mean or np.nan, p_std or np.nan
        n_mean, n_std = n_mean or np.nan, n_std or np.nan
        pooled_std = np.nanmean([p_std, n_std])
        gap = abs(p_mean - n_mean) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan
        flag = ""
        if np.isnan(gap) or gap < 0.2:
            flag += "WEAK "
        if p_nan > 50 or n_nan > 50:
            flag += "NAN-HEAVY"
        if flag:
            flags.append((col, flag.strip()))
        print(f"{col:<26}{p_mean:>10.3f}{p_std:>10.3f}{n_mean:>10.3f}{n_std:>10.3f}"
              f"{p_nan:>10.1f}{n_nan:>10.1f}  {flag}")

    print(f"\n{'feature':<26}{'value':<12}{'pos_share':>10}{'neg_share':>10}")
    for col in CATEGORICAL_FEATURE_COLS:
        rows = con.execute(f"""
            SELECT {col} AS val, label, count(*) AS n FROM '{path}'
            WHERE {col} IS NOT NULL GROUP BY {col}, label
        """).fetchall()
        counts = {}
        for val, label, n in rows:
            counts.setdefault(val, {0: 0, 1: 0})[label] = n
        for val in sorted(counts):
            p_share = 100.0 * counts[val][1] / n_pos if n_pos else np.nan
            n_share = 100.0 * counts[val][0] / n_neg if n_neg else np.nan
            print(f"{col:<26}{str(val):<12}{p_share:>10.1f}{n_share:>10.1f}")

    con.close()
    if flags:
        print("\n[features] weak/NaN-heavy features flagged:")
        for col, flag in flags:
            print(f"  {col}: {flag}")
    else:
        print("\n[features] no weak/NaN-heavy features flagged.")


def main():
    """Build the Phase 1 positive-vs-negative labeled feature table and print the separation report."""
    print_sysinfo()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "train_gt_vs_negatives.parquet"
    con = connect(memory_limit_gb=2, threads=2)

    with stage("count positives"):
        n_pos = con.execute(
            f"SELECT count(*) FROM '{(RAW_DIR / 'train_gt_joined.parquet').as_posix()}'"
        ).fetchone()[0]
    print(f"  positives: {n_pos}")

    writer_state = {"writer": None, "path": out_path}
    with make_pool(WORKERS) as pool:
        with stage("featurize positives"):
            _write_labeled_batches(con, _positive_pairs_query(), 1, n_pos, pool, writer_state)

        with stage("build random same-country negatives"):
            _build_negative_view(con)
            n_neg = con.execute("SELECT count(*) FROM _neg_raw").fetchone()[0]
        print(f"  negatives: {n_neg} (dropped collisions with any true match: {n_pos - n_neg})")

        with stage("featurize negatives"):
            _write_labeled_batches(con, "SELECT * FROM _neg_raw", 0, n_neg, pool, writer_state)

    if writer_state["writer"] is not None:
        writer_state["writer"].close()
    con.close()
    print(f"  saved -> {out_path}")

    print_separation_report(out_path)


if __name__ == "__main__":
    main()
