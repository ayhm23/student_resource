"""Step 2 Part A: mine normalization dictionaries from TRAIN matched pairs only.

Everything here is derived from ``data/raw/train_gt_long.parquet`` (the exploded
ground truth) joined back to the train Source1/2/3 text, plus a handful of
hand-written candidate lists (address abbreviations, US/India state names,
legal forms, honorifics) that are cross-checked against real matched pairs
before being trusted. No test labels exist and none are used here.

Outputs (all under ``data/dicts/``):
    devanagari_name_map.json   token -> {map, support, agreement, total_seen}
    devanagari_addr_map.json   same, for address tokens
    address_abbrev_map.json    verified {full: short} address-word abbreviations
    state_map_us.json          verified {full: code} US state names
    state_map_india.json       verified {full: code} India state names
    name_wrappers.json         top separator tokens/bigrams with counts
    legal_forms.json           {token: count} legal-form tokens seen as the
                                last token of a business_name in train
    filler_words.json          {token: count} filler-word tokens seen as the
                                last token of a business_name in train
    honorifics.json            {token: count} honorific tokens seen as the
                                first token of a business_name in train

A human-readable summary (support/coverage numbers for every dictionary) is
printed to stdout and saved to ``data/dicts/mining_summary.txt`` so it can be
folded into ``output/BLOCKING_REPORT.md``.
"""

import json
import re
import sys
from collections import Counter, defaultdict

from unidecode import unidecode

from config import RAW_DIR, DICTS_DIR, raw_parquet_path
from db import connect
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8")

JOINED_PAIRS_PARQUET = RAW_DIR / "train_gt_joined.parquet"
BATCH_ROWS = 200_000
MIN_SUPPORT = 5
MIN_AGREEMENT = 0.7

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
WORD_RE = re.compile(r"\w+", re.UNICODE)

ADDR_ABBR_CANDIDATES = {
    "road": "rd", "street": "st", "drive": "dr", "avenue": "ave", "lane": "ln",
    "court": "ct", "boulevard": "blvd", "place": "pl", "circle": "cir",
    "trail": "trl", "highway": "hwy", "terrace": "ter", "parkway": "pkwy",
}

US_STATE_CANDIDATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}

INDIA_STATE_CANDIDATES = {
    "maharashtra": "mh", "gujarat": "gj", "karnataka": "ka", "telangana": "tg",
    "rajasthan": "rj", "haryana": "hr", "kerala": "kl", "keralam": "kl",
    "bihar": "br", "punjab": "pb", "orissa": "od", "odisha": "od",
    "tamil nadu": "tn", "uttar pradesh": "up", "madhya pradesh": "mp",
    "west bengal": "wb", "andhra pradesh": "ap", "jharkhand": "jh",
    "chhattisgarh": "cg", "assam": "as", "delhi": "dl", "new delhi": "dl",
    "uttarakhand": "uk", "himachal pradesh": "hp", "goa": "ga",
    "tripura": "tr", "meghalaya": "ml", "manipur": "mn", "nagaland": "nl",
    "mizoram": "mz", "sikkim": "sk", "arunachal pradesh": "ar",
    "jammu and kashmir": "jk", "puducherry": "py", "chandigarh": "ch",
}

# Verified by frequency in train, not by pairwise co-occurrence (see task 4/5
# docstring note): what matters is that each candidate genuinely occurs as a
# name boundary token in this dataset. French legal forms are added by hand
# per the task spec -- train has no France, so they will show a 0 count here
# and are trusted for test-time normalization only.
LEGAL_FORM_CANDIDATES = [
    "ltd", "limited", "inc", "incorporated", "llc", "lp", "pc", "corp",
    "corporation", "co", "company",
    # French legal forms (no train support expected; added by hand)
    "sarl", "sas", "sasu", "eurl", "sa", "snc", "sci",
]
FILLER_WORD_CANDIDATES = [
    "services", "service", "center", "centre", "partners", "group",
    "solutions", "associates", "clinic", "technologies", "foundation",
    "institute", "trading", "brothers", "care",
]
HONORIFIC_CANDIDATES = [
    "m/s", "ms", "mr", "mrs", "dr", "smt", "shri", "sri", "prof",
]

# Known mining artifacts from output/substitutions.txt: multi-word diffs that
# coincidentally collapsed to one token per side, or single-character-drop
# typos. Never trusted as real substitutions.
NOISE_ARTIFACTS = {("private", "center"), ("inc", "nc"), ("and", "nd"),
                    ("street", "saint"), ("plot", "h")}


def tokenize(text):
    """Lowercase-agnostic Unicode word tokenizer (keeps Devanagari as \\w)."""
    return WORD_RE.findall(text)


def has_devanagari(text):
    """Return True if ``text`` contains any Devanagari (U+0900-U+097F) char."""
    return bool(DEVANAGARI_RE.search(text))


def build_joined_pairs(con):
    """Join every train ground-truth matched pair with its S1 and S2/S3 text.

    Uses a single UNION ALL + equi-join (source, id) rather than two OR'd LEFT
    JOINs, which DuckDB otherwise plans very poorly (tested: >4 min vs <1s).
    """
    s1_path = raw_parquet_path("train", "S1").as_posix()
    s2_path = raw_parquet_path("train", "S2").as_posix()
    s3_path = raw_parquet_path("train", "S3").as_posix()
    gt_long_path = (RAW_DIR / "train_gt_long.parquet").as_posix()
    con.execute(f"""
        CREATE OR REPLACE VIEW s23 AS
        SELECT 'S2' AS source, id, business_name, business_address, country FROM '{s2_path}'
        UNION ALL
        SELECT 'S3' AS source, id, business_name, business_address, country FROM '{s3_path}'
    """)
    con.execute(f"""
        COPY (
            SELECT
                g.s1_id, g.match_id, g.match_source,
                a.business_name AS s1_name, a.business_address AS s1_addr, a.country AS s1_country,
                b.business_name AS match_name, b.business_address AS match_addr, b.country AS match_country
            FROM '{gt_long_path}' g
            JOIN '{s1_path}' a ON g.s1_id = a.id
            JOIN s23 b ON g.match_source = b.source AND g.match_id = b.id
        ) TO '{JOINED_PAIRS_PARQUET.as_posix()}' (FORMAT PARQUET)
    """)


def iter_batches(con, columns):
    """Yield pandas DataFrames of ``columns`` from the joined pairs parquet."""
    cols = ", ".join(columns)
    result = con.execute(f"SELECT {cols} FROM '{JOINED_PAIRS_PARQUET.as_posix()}'")
    reader = result.to_arrow_reader(BATCH_ROWS)
    for batch in reader:
        yield batch.to_pandas()


# ---------------------------------------------------------------------------
# Task 1: Devanagari -> Latin token map
# ---------------------------------------------------------------------------

def mine_devanagari_map(con, a_col, b_col):
    """Mine a Devanagari->Latin token map from one text field of matched pairs.

    For each pair where exactly one side contains Devanagari text and both
    sides tokenize to the same length, aligns tokens positionally and tallies
    (devanagari_token -> latin_token) co-occurrence. Keeps a mapping only when
    its best candidate has support >= MIN_SUPPORT and agreement >= MIN_AGREEMENT.
    """
    pair_counts = defaultdict(Counter)
    n_pairs_considered = 0
    for df in iter_batches(con, [a_col, b_col]):
        for a_text, b_text in zip(df[a_col], df[b_col]):
            if not a_text or not b_text:
                continue
            a_dev, b_dev = has_devanagari(a_text), has_devanagari(b_text)
            if a_dev == b_dev:
                continue
            dev_text, lat_text = (a_text, b_text) if a_dev else (b_text, a_text)
            dev_tokens = tokenize(dev_text)
            lat_tokens = [t.lower() for t in tokenize(lat_text)]
            if not dev_tokens or len(dev_tokens) != len(lat_tokens):
                continue
            n_pairs_considered += 1
            for dt, lt in zip(dev_tokens, lat_tokens):
                if DEVANAGARI_RE.search(dt):
                    pair_counts[dt][lt] += 1

    mapping = {}
    for dt, sub in pair_counts.items():
        total = sum(sub.values())
        best_lt, best_n = sub.most_common(1)[0]
        agreement = best_n / total
        if best_n >= MIN_SUPPORT and agreement >= MIN_AGREEMENT:
            mapping[dt] = {"map": best_lt, "support": best_n,
                           "agreement": round(agreement, 3), "total_seen": total}
    return mapping, n_pairs_considered, len(pair_counts)


def devanagari_token_universe(con, split):
    """Return a Counter of Devanagari token frequencies across one split.

    Scans business_name and business_address of S1/S2/S3 for ``split``,
    filtering to rows that contain Devanagari text (cheap DuckDB regex scan)
    before tokenizing in Python, so the full non-Devanagari majority of the
    dataset is never materialized.
    """
    counter = Counter()
    devanagari_re2 = r"[\x{0900}-\x{097F}]"
    for source in ("S1", "S2", "S3"):
        path = raw_parquet_path(split, source).as_posix()
        rows = con.execute(
            f"""
            SELECT business_name, business_address FROM '{path}'
            WHERE regexp_matches(business_name, ?)
               OR regexp_matches(business_address, ?)
            """,
            [devanagari_re2, devanagari_re2],
        ).fetchall()
        for name, addr in rows:
            for text in (name, addr):
                for tok in tokenize(text):
                    if DEVANAGARI_RE.search(tok):
                        counter[tok] += 1
    return counter


# ---------------------------------------------------------------------------
# Task 2: address canonical map (verified against real matched pairs)
# ---------------------------------------------------------------------------

def build_reverse_index(candidates):
    """Map each candidate token to the (full, short) pair(s) it belongs to."""
    idx = defaultdict(list)
    for full, short in candidates.items():
        idx[full].append((full, short))
        idx[short].append((full, short))
    return idx


def verify_candidates(con, candidates, min_hits=MIN_SUPPORT):
    """Verify hand-written (full, short) address/state candidates on real pairs.

    A candidate is confirmed when, across the matched pairs, one side's
    address contains the full form (and not the short form) while the other
    side contains the short form (and not the full form), at least
    ``min_hits`` times. Returns (verified_dict, hit_counts).

    Matching is done with a single longest-first alternation regex over the
    space-joined token stream (not a token-set intersection), so multi-word
    candidates like "new hampshire" are detected as an adjacent phrase rather
    than silently failing to match at all (a token-set approach can only ever
    match single-token candidates).
    """
    idx = build_reverse_index(candidates)
    all_keys = sorted(idx.keys(), key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in all_keys) + r")\b")
    hits = Counter()
    for df in iter_batches(con, ["s1_addr", "match_addr"]):
        for a_text, b_text in zip(df["s1_addr"], df["match_addr"]):
            if not a_text or not b_text:
                continue
            ta_text = " ".join(tokenize(unidecode(a_text).lower()))
            tb_text = " ".join(tokenize(unidecode(b_text).lower()))
            ta = set(pattern.findall(ta_text))
            tb = set(pattern.findall(tb_text))
            if not ta or not tb:
                continue
            for tok in ta:
                for full, short in idx[tok]:
                    if tok == full and short in tb and full not in tb:
                        hits[(full, short)] += 1
                    elif tok == short and full in tb and short not in tb:
                        hits[(full, short)] += 1
    verified = {full: short for full, short in candidates.items()
                if hits[(full, short)] >= min_hits}
    return verified, hits


# ---------------------------------------------------------------------------
# Task 3: name wrapper separators (dba, née, ...)
# ---------------------------------------------------------------------------

def mine_wrappers(con, min_len=6):
    """Mine separator tokens used when an S1 name is wrapped inside a longer match name."""
    uni = Counter()
    bi = Counter()
    for df in iter_batches(con, ["s1_name", "match_name"]):
        for s1n, mn in zip(df["s1_name"], df["match_name"]):
            if not s1n or not mn:
                continue
            a = unidecode(s1n).lower().strip()
            b = unidecode(mn).lower().strip()
            if len(a) < min_len or a == b:
                continue
            idx = b.find(a)
            if idx <= 0:
                continue
            prefix = b[:idx].strip()
            if not prefix:
                continue
            toks = prefix.split()
            uni[toks[-1]] += 1
            if len(toks) >= 2:
                bi[" ".join(toks[-2:])] += 1
    return uni, bi


# ---------------------------------------------------------------------------
# Tasks 4 & 5: legal forms / filler words / honorifics, verified by frequency
# ---------------------------------------------------------------------------

def mine_boundary_tokens(con, last_candidates, first_candidates):
    """Count candidate legal-form/filler (last token) and honorific (first token) hits.

    Scans train S1+S2+S3 business_name exactly once (batched, not fetchall)
    and tallies both boundary positions in the same pass. Uses a plain
    whitespace split (not the Unicode word tokenizer) so tokens like "m/s"
    survive intact.
    """
    last_counts = Counter({c: 0 for c in last_candidates})
    first_counts = Counter({c: 0 for c in first_candidates})
    last_set = set(last_candidates)
    first_set = set(first_candidates)
    for source in ("S1", "S2", "S3"):
        path = raw_parquet_path("train", source).as_posix()
        result = con.execute(f"SELECT business_name FROM '{path}'")
        for batch in result.to_arrow_reader(BATCH_ROWS):
            for (name,) in batch.to_pandas().itertuples(index=False):
                if not name:
                    continue
                toks = name.lower().split()
                if not toks:
                    continue
                last_tok = toks[-1].strip(".,;:()")
                first_tok = toks[0].strip(".,;:()")
                if last_tok in last_set:
                    last_counts[last_tok] += 1
                if first_tok in first_set:
                    first_counts[first_tok] += 1
    return dict(last_counts), dict(first_counts)


def main():
    """Run all Part A mining tasks and write dictionaries + a summary report."""
    low_ram = print_sysinfo()
    DICTS_DIR.mkdir(parents=True, exist_ok=True)
    con = connect()

    with stage("build joined train matched-pairs table"):
        build_joined_pairs(con)
        n_joined = con.execute(f"SELECT count(*) FROM '{JOINED_PAIRS_PARQUET.as_posix()}'").fetchone()[0]
    print(f"  joined pairs: {n_joined}")

    summary_lines = []
    summary_lines.append(f"Joined train matched pairs: {n_joined}")

    # --- Task 1: Devanagari maps ---
    with stage("mine Devanagari->Latin name map"):
        name_map, name_pairs_considered, name_distinct = mine_devanagari_map(con, "s1_name", "match_name")
    with stage("mine Devanagari->Latin address map"):
        addr_map, addr_pairs_considered, addr_distinct = mine_devanagari_map(con, "s1_addr", "match_addr")
    (DICTS_DIR / "devanagari_name_map.json").write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (DICTS_DIR / "devanagari_addr_map.json").write_text(json.dumps(addr_map, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_lines.append(f"\n=== Task 1: Devanagari->Latin maps ===")
    summary_lines.append(f"NAME: {name_pairs_considered} equal-length dev/lat pairs considered, "
                          f"{name_distinct} distinct dev tokens seen, {len(name_map)} kept "
                          f"(support>={MIN_SUPPORT}, agreement>={MIN_AGREEMENT}).")
    summary_lines.append(f"ADDRESS: {addr_pairs_considered} equal-length dev/lat pairs considered, "
                          f"{addr_distinct} distinct dev tokens seen, {len(addr_map)} kept.")

    with stage("Devanagari coverage in train/test"):
        combined_map_keys = set(name_map) | set(addr_map)
        for split in ("train", "test"):
            universe = devanagari_token_universe(con, split)
            covered = sum(n for tok, n in universe.items() if tok in combined_map_keys)
            total = sum(universe.values())
            distinct_covered = sum(1 for tok in universe if tok in combined_map_keys)
            pct = 100.0 * covered / total if total else 0.0
            distinct_pct = 100.0 * distinct_covered / len(universe) if universe else 0.0
            summary_lines.append(
                f"{split} Devanagari coverage: {covered}/{total} token occurrences "
                f"({pct:.1f}%), {distinct_covered}/{len(universe)} distinct tokens ({distinct_pct:.1f}%)."
            )

    # --- Task 2: address abbreviations + state maps ---
    with stage("verify address abbreviation candidates"):
        addr_abbr, addr_hits = verify_candidates(con, ADDR_ABBR_CANDIDATES)
    with stage("verify US state candidates"):
        us_states, us_hits = verify_candidates(con, US_STATE_CANDIDATES)
    with stage("verify India state candidates"):
        india_states, india_hits = verify_candidates(con, INDIA_STATE_CANDIDATES)
    (DICTS_DIR / "address_abbrev_map.json").write_text(json.dumps(addr_abbr, indent=2), encoding="utf-8")
    (DICTS_DIR / "state_map_us.json").write_text(json.dumps(us_states, indent=2), encoding="utf-8")
    (DICTS_DIR / "state_map_india.json").write_text(json.dumps(india_states, indent=2), encoding="utf-8")
    summary_lines.append(f"\n=== Task 2: canonical address/state maps (verified, min_hits={MIN_SUPPORT}) ===")
    summary_lines.append(f"Address abbreviations: {len(addr_abbr)}/{len(ADDR_ABBR_CANDIDATES)} candidates verified.")
    for full, short in ADDR_ABBR_CANDIDATES.items():
        summary_lines.append(f"  {full} -> {short}: hits={addr_hits[(full, short)]}"
                              f"{' [KEPT]' if full in addr_abbr else ' [dropped]'}")
    summary_lines.append(f"US states: {len(us_states)}/{len(US_STATE_CANDIDATES)} candidates verified.")
    summary_lines.append(f"India states: {len(india_states)}/{len(INDIA_STATE_CANDIDATES)} candidates verified.")
    dropped_india = [f for f in INDIA_STATE_CANDIDATES if f not in india_states]
    if dropped_india:
        summary_lines.append(f"  India states dropped (no/low support in train): {dropped_india}")
    dropped_us = [f for f in US_STATE_CANDIDATES if f not in us_states]
    if dropped_us:
        summary_lines.append(f"  US states dropped (no/low support in train): {dropped_us}")

    # --- Task 3: name wrappers ---
    with stage("mine name wrapper separators"):
        wrap_uni, wrap_bi = mine_wrappers(con)
    top_wrappers = (wrap_uni.most_common(20), wrap_bi.most_common(20))
    (DICTS_DIR / "name_wrappers.json").write_text(
        json.dumps({"unigram": dict(wrap_uni.most_common(50)),
                    "bigram": dict(wrap_bi.most_common(50))}, indent=2),
        encoding="utf-8")
    summary_lines.append(f"\n=== Task 3: name wrapper separators ===")
    summary_lines.append("Top unigram separators: " + ", ".join(f"{t}:{n}" for t, n in top_wrappers[0]))
    summary_lines.append("Top bigram separators: " + ", ".join(f"{t}:{n}" for t, n in top_wrappers[1]))

    # --- Tasks 4 & 5: legal forms, filler words, honorifics ---
    with stage("count legal-form / filler / honorific tokens"):
        last_candidates = LEGAL_FORM_CANDIDATES + FILLER_WORD_CANDIDATES
        last_counts, honorific_counts = mine_boundary_tokens(con, last_candidates, HONORIFIC_CANDIDATES)
        legal_counts = {t: last_counts[t] for t in LEGAL_FORM_CANDIDATES}
        filler_counts = {t: last_counts[t] for t in FILLER_WORD_CANDIDATES}
    (DICTS_DIR / "legal_forms.json").write_text(json.dumps(legal_counts, indent=2), encoding="utf-8")
    (DICTS_DIR / "filler_words.json").write_text(json.dumps(filler_counts, indent=2), encoding="utf-8")
    (DICTS_DIR / "honorifics.json").write_text(json.dumps(honorific_counts, indent=2), encoding="utf-8")
    summary_lines.append(f"\n=== Tasks 4/5: legal forms / filler words / honorifics (train frequency) ===")
    summary_lines.append(f"Legal forms (last token of business_name): {legal_counts}")
    summary_lines.append(f"Filler words (last token of business_name): {filler_counts}")
    summary_lines.append(f"Honorifics (first token of business_name): {honorific_counts}")

    summary_lines.append(f"\nKnown mining artifacts excluded from any substitution use: {sorted(NOISE_ARTIFACTS)}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (DICTS_DIR / "mining_summary.txt").write_text(summary_text, encoding="utf-8")

    con.close()


if __name__ == "__main__":
    main()
