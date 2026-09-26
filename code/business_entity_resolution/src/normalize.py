"""Step 2 Part B/C: normalize business_name/business_address into matching
fields, and compute name/address token document frequencies (IDF).

Reads ``data/raw/{split}_{source}.parquet`` (from ``ingest.py``) and the
dictionaries mined by ``mine_dicts.py``, and writes
``data/norm/{split}_{source}.parquet`` with one row per input record:

    id, country,
    name_clean, name_main, name_alt, name_core, name_nospace,
    name_sorted_chars, legal_form, name_skeleton,
    addr_clean, numbers, state, addr_words, addr_skeleton

``numbers`` is a comma-joined string of integers (leading zeros stripped;
Step 3 Track A4 also recovers hyphen-split short numbers like "2-0"->20).
``name_core``/``name_nospace``/``addr_words`` are space-joined token strings.
``name_skeleton``/``addr_skeleton`` (Step 3 Track A2) are the same tokens as
``name_core``/``addr_words`` folded to a consonant skeleton (vowels/h/y
dropped except each token's first letter, phonetic consonant merges,
repeated letters collapsed) -- meant to survive transliteration/typo
spelling variance that breaks exact/prefix token matching.

Per-record text work is pure Python (regex/unidecode), so it is parallelized
with ``multiprocessing.Pool`` over chunks read from DuckDB -- never a
row-by-row pandas ``.apply`` over the full file.
"""

import json
import re
import sys
import unicodedata
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from unidecode import unidecode

from config import (DICTS_DIR, NORM_DIR, SOURCES, SPLITS, norm_parquet_path,
                     raw_parquet_path)
from db import connect
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8")

CHUNK_ROWS = 100_000
POOL_CHUNKSIZE = 2000
WORKERS = 4  # conservative: this machine often has only a few GB free RAM

WORD_RE = re.compile(r"\w+", re.UNICODE)
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
DOTTED_INITIALS_RE = re.compile(r"\b(?:[a-z]\.){2,}")
NONALNUM_RE = re.compile(r"[^a-z0-9\s]")
HASH_DIGITS_RE = re.compile(r"#\d+")
MULTISPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d+")
HYPHEN_NUM_RE = re.compile(r"\b(\d+)-(\d+)\b")

# Step 3 Track A2: a "skeleton" folds transliterated non-Latin text (and
# ordinary Latin typos) toward a script/spelling-invariant form so that e.g.
# unidecode's "yuunivrsl" (from Telugu) and the Latin "universal" compare as
# similar tokens. Tested against real train pairs (see translit_compare.py):
# unidecode already beats indic_transliteration as the upstream transliterator
# for 7 of 9 Indic scripts, so the skeleton is built on unidecode's output
# (already applied in _base_clean), not a second transliteration pass.
CONSONANT_SUBS = [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"),
                   ("gh", "g"), ("ch", "c"), ("sh", "s"), ("z", "j"), ("q", "k"),
                   ("w", "v"), ("ck", "k")]
SKELETON_DROP_RE = re.compile(r"[aeiouhy]")
SKELETON_REPEAT_RE = re.compile(r"(.)\1+")

NAME_STOPWORDS = {"a", "an", "the", "and", "of", "et", "de", "la", "le", "du"}
DOMAIN_TOKENS = {"com", "net", "org"}
ADDR_DROP_KEYWORDS = {"city", "no", "door", "flat", "plot", "house", "shop"}
# Best-effort seed list from the 8 France test records seen in Step 1 EDA --
# there is no French training data to mine a real region dictionary from.
FRANCE_REGION_SEEDS = {"nouvelle aquitaine", "hauts de france", "pays de la loire"}

_G = {}


def _load_json(dicts_dir, name):
    """Load one dictionary JSON file, or {} if it doesn't exist yet."""
    p = Path(dicts_dir) / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _init_worker(dicts_dir):
    """Multiprocessing pool initializer: load all mined dictionaries once per worker."""
    dev_name = _load_json(dicts_dir, "devanagari_name_map.json")
    dev_addr = _load_json(dicts_dir, "devanagari_addr_map.json")
    addr_abbr = _load_json(dicts_dir, "address_abbrev_map.json")
    us_states = _load_json(dicts_dir, "state_map_us.json")
    india_states = _load_json(dicts_dir, "state_map_india.json")
    wrappers = _load_json(dicts_dir, "name_wrappers.json")
    legal_counts = _load_json(dicts_dir, "legal_forms.json")
    honorific_counts = _load_json(dicts_dir, "honorifics.json")

    other_states = {seed: seed.replace(" ", "_") for seed in FRANCE_REGION_SEEDS}

    # Country-scoped state maps: a bare short code like "de" (Delaware) or
    # "mt" (Montana) must never be recognized for a French/Indian record just
    # because it happens to collide with a common word there ("de"/"du" are
    # French prepositions) -- state lookup is keyed by the record's own
    # country, never a single merged global map.
    _G["state_maps"] = {"US": us_states, "India": india_states}
    _G["state_codes"] = {c: set(m.values()) for c, m in _G["state_maps"].items()}
    _G["state_keys_by_len"] = {c: sorted(m.keys(), key=lambda k: -len(k.split()))
                                for c, m in _G["state_maps"].items()}
    _G["other_state_map"] = other_states
    _G["other_state_codes"] = set(other_states.values())
    _G["other_state_keys_by_len"] = sorted(other_states.keys(), key=lambda k: -len(k.split()))

    _G["dev_name"] = {k: v["map"] for k, v in dev_name.items()}
    _G["dev_addr"] = {k: v["map"] for k, v in dev_addr.items()}
    _G["addr_abbr"] = addr_abbr
    _G["street_tokens"] = set(addr_abbr.keys()) | set(addr_abbr.values())
    _G["legal_forms"] = set(legal_counts.keys()) | {"sarl", "sas", "sasu", "eurl", "sa", "snc", "sci"}
    _G["honorifics"] = set(honorific_counts.keys())
    _G["wrap_separators"] = {_clean_separator(t) for t in wrappers.get("unigram", {})} if wrappers else set()


def _clean_separator(tok):
    """Normalize a mined wrapper-separator token the same way name tokens are cleaned."""
    return re.sub(r"[^a-z0-9]", "", unidecode(tok).lower())


def _base_clean(text, devanagari_map):
    """Shared name/address cleaning pipeline; returns a deduped token list.

    NFKC normalize -> map Devanagari tokens (fallback unidecode) -> unidecode
    remaining accents -> lowercase -> "&"->" and " -> drop "#<digits>" tags ->
    join dotted initials (l.l.c.->llc) -> strip punctuation -> collapse spaces
    -> dedupe repeated tokens (order-preserving).
    """
    text = unicodedata.normalize("NFKC", text)

    def repl(m):
        tok = m.group(0)
        if DEVANAGARI_RE.search(tok):
            mapped = devanagari_map.get(tok)
            return mapped if mapped else unidecode(tok)
        return tok

    text = WORD_RE.sub(repl, text)
    text = unidecode(text)
    text = text.lower()
    text = text.replace("&", " and ")
    text = HASH_DIGITS_RE.sub(" ", text)
    text = DOTTED_INITIALS_RE.sub(lambda m: m.group(0).replace(".", ""), text)
    text = NONALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    tokens = text.split()
    return list(dict.fromkeys(tokens))


def skeleton_token(tok):
    """Fold one lowercase-ASCII token to a consonant skeleton (Step 3 Track A2).

    Applies phonetic consonant merges, then drops vowels/h/y everywhere
    except the token's own first letter (so distinct tokens don't collapse
    to the same empty skeleton), then collapses repeated letters -- smooths
    over the doubled-letter/extra-vowel artifacts transliteration commonly
    introduces (e.g. unidecode's "yuunivrsl" and plain "universal" both fold
    towards "unvrsl"-like forms).
    """
    if not tok:
        return ""
    s = tok
    for a, b in CONSONANT_SUBS:
        s = s.replace(a, b)
    first, rest = s[0], s[1:]
    rest = SKELETON_DROP_RE.sub("", rest)
    return SKELETON_REPEAT_RE.sub(r"\1", first + rest)


def skeleton_join(tokens):
    """Skeletonize a list of tokens and space-join them, parallel in order to ``tokens``."""
    return " ".join(skeleton_token(t) for t in tokens if t)


def normalize_name(name):
    """Compute name_clean/name_main/name_alt/name_core/name_nospace/name_sorted_chars/legal_form/name_skeleton."""
    tokens = _base_clean(name, _G["dev_name"])
    name_clean = " ".join(tokens)

    name_main, name_alt = name_clean, ""
    sep_idx = None
    for i in range(1, len(tokens) - 1):
        if tokens[i] in _G["wrap_separators"]:
            sep_idx = i  # keep the LAST separator occurrence
    if sep_idx is not None:
        name_alt = " ".join(tokens[:sep_idx])
        name_main = " ".join(tokens[sep_idx + 1:])

    main_toks = name_main.split()
    while main_toks and main_toks[0] in _G["honorifics"]:
        main_toks.pop(0)
    legal_found = [t for t in main_toks if t in _G["legal_forms"]]
    legal_form = legal_found[0] if legal_found else ""
    core_toks = [t for t in main_toks if t not in _G["legal_forms"]]
    name_core = " ".join(core_toks)

    ns_toks = [t for t in core_toks if t not in NAME_STOPWORDS and t not in DOMAIN_TOKENS]
    if ns_toks and ns_toks[0] == "www":
        ns_toks = ns_toks[1:]
    name_nospace = "".join(ns_toks)
    name_sorted_chars = "".join(sorted(c for c in name_nospace if c.isalpha()))
    name_skeleton = skeleton_join(core_toks)

    return {
        "name_clean": name_clean, "name_main": name_main, "name_alt": name_alt,
        "name_core": name_core, "name_nospace": name_nospace,
        "name_sorted_chars": name_sorted_chars, "legal_form": legal_form,
        "name_skeleton": name_skeleton,
    }


def _apply_state_and_abbrev(tokens, country):
    """Replace multi-word then single-word state names and street abbreviations.

    State recognition is scoped to ``country``: US/India use their own
    verified maps, and anything else (open set, e.g. France) only gets the
    best-effort region-name seed list. This matters because a bare short
    code can collide with an ordinary word in another country's language --
    e.g. "de" is Delaware's code but also the French preposition "of/from",
    and without this scoping every French address containing "de" (nearly
    all of them) would be mis-tagged state="de".
    """
    if country in _G["state_maps"]:
        state_map = _G["state_maps"][country]
        state_codes = _G["state_codes"][country]
        state_keys_by_len = _G["state_keys_by_len"][country]
    else:
        state_map = _G["other_state_map"]
        state_codes = _G["other_state_codes"]
        state_keys_by_len = _G["other_state_keys_by_len"]

    tokens = list(tokens)
    n = len(tokens)
    used = [False] * n
    state_found = ""
    for key in state_keys_by_len:
        key_toks = key.split()
        klen = len(key_toks)
        if klen < 2:
            continue
        for start in range(0, n - klen + 1):
            if any(used[start:start + klen]):
                continue
            if tokens[start:start + klen] == key_toks:
                for j in range(start, start + klen):
                    used[j] = True
                tokens[start] = state_map[key]
                for j in range(start + 1, start + klen):
                    tokens[j] = None
                if not state_found:
                    state_found = state_map[key]
    tokens = [t for t in tokens if t is not None]

    out = []
    for t in tokens:
        if t in _G["addr_abbr"]:
            out.append(_G["addr_abbr"][t])
        elif t in state_map and " " not in t:
            if not state_found:
                state_found = state_map[t]
            out.append(state_map[t])
        elif t in state_codes:
            # already a short code (e.g. "il", "mh") rather than a full name
            if not state_found:
                state_found = t
            out.append(t)
        else:
            out.append(t)
    return out, state_found


def _drop_addr_keywords(tokens):
    """Drop 'city', the 'po box <n>' phrase, and no/door/flat/plot/house/shop keywords."""
    out = []
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t == "po" and i + 1 < n and tokens[i + 1] == "box":
            i += 2
            if i < n and tokens[i].isdigit():
                i += 1
            continue
        if t in ADDR_DROP_KEYWORDS:
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def extract_numbers(address):
    """Extract address numbers (Step 3 Track A4: also recovers hyphen-split short numbers).

    Plain digit runs are extracted as before (leading zeros drop naturally
    via ``int()``; "#10702", "L-02/1147" etc. already work since ``\\d+``
    ignores surrounding non-digit characters). Additionally, a "23-27" or
    "2-0" style hyphenated pair whose JOINED length is <=4 also contributes
    its joined form (20, 2327), since this dataset's corruption sometimes
    splits what should be one short house number across a hyphen -- the
    original individual parts are kept too, since a real address range
    ("23-27 Main St") is also common and both readings should stay available
    to the blocking keys.
    """
    numbers = {int(m) for m in NUMBER_RE.findall(address)}
    for m in HYPHEN_NUM_RE.finditer(address):
        joined = m.group(1) + m.group(2)
        if len(joined) <= 4:
            numbers.add(int(joined))
    return sorted(numbers)


def normalize_addr(address, country):
    """Compute addr_clean/numbers/state/addr_words/addr_skeleton for one address string."""
    numbers = extract_numbers(address)
    tokens = _base_clean(address, _G["dev_addr"])
    tokens, state_found = _apply_state_and_abbrev(tokens, country)
    tokens = _drop_addr_keywords(tokens)
    addr_clean = " ".join(tokens)
    street_tokens = _G["street_tokens"]
    addr_words = [t for t in tokens if not t.isdigit() and t != state_found and t not in street_tokens]
    return {
        "addr_clean": addr_clean,
        "numbers": ",".join(str(x) for x in numbers),
        "state": state_found,
        "addr_words": " ".join(addr_words),
        "addr_skeleton": skeleton_join(addr_words),
    }


def _process_row(row):
    """Normalize one (id, name, address, country) row; runs inside a worker process."""
    _id, name, address, country = row
    out = {}
    out.update(normalize_name(name or ""))
    out.update(normalize_addr(address or "", country))
    return out


def process_source(split, source, pool):
    """Stream one raw source parquet through the pool, writing its norm parquet."""
    raw_path = raw_parquet_path(split, source).as_posix()
    out_path = norm_parquet_path(split, source)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    con = connect()
    result = con.execute(f"SELECT id, business_name, business_address, country FROM '{raw_path}'")
    reader = result.to_arrow_reader(CHUNK_ROWS)

    writer = None
    total = 0
    for batch in reader:
        df = batch.to_pandas()
        rows = list(zip(df["id"], df["business_name"], df["business_address"], df["country"]))
        results = pool.map(_process_row, rows, chunksize=POOL_CHUNKSIZE)
        out_df = pd.DataFrame(results)
        out_df.insert(0, "id", df["id"].values)
        out_df.insert(1, "country", df["country"].values)
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)
        writer.write_table(table)
        total += len(df)
    if writer is not None:
        writer.close()
    con.close()
    print(f"  {split}_{source}: {total} rows normalized -> {out_path}")


def print_examples(con):
    """Print 20 random before/after normalization examples per (split, country)."""
    for split in SPLITS:
        countries = ["US", "India", "France"] if split == "test" else ["US", "India"]
        union_sql = " UNION ALL ".join(
            f"""SELECT r.business_name, r.business_address, r.country,
                       n.name_clean, n.name_core, n.name_nospace,
                       n.addr_clean, n.numbers, n.state, n.addr_words
                FROM '{raw_parquet_path(split, s).as_posix()}' r
                JOIN '{norm_parquet_path(split, s).as_posix()}' n USING (id)"""
            for s in SOURCES
        )
        for country in countries:
            print(f"\n--- {split} / {country}: 20 random before/after examples ---")
            rows = con.execute(
                f"SELECT * FROM ({union_sql}) WHERE country = ? ORDER BY random() LIMIT 20",
                [country],
            ).fetchall()
            for r in rows:
                (bname, baddr, _country, name_clean, name_core, name_nospace,
                 addr_clean, numbers, state, addr_words) = r
                print(f"  BEFORE name={bname!r} addr={baddr!r}")
                print(f"  AFTER  name_clean={name_clean!r} name_core={name_core!r} "
                      f"name_nospace={name_nospace!r}")
                print(f"         addr_clean={addr_clean!r} numbers={numbers!r} "
                      f"state={state!r} addr_words={addr_words!r}")


def main():
    """Normalize every train/test source file and print before/after examples."""
    print_sysinfo()
    with Pool(processes=WORKERS, initializer=_init_worker, initargs=(str(DICTS_DIR),)) as pool:
        for split in SPLITS:
            for source in SOURCES:
                with stage(f"normalize {split}_{source}"):
                    process_source(split, source, pool)

    con = connect()
    with stage("print before/after examples"):
        print_examples(con)
    con.close()


if __name__ == "__main__":
    main()
