"""Step 1 noise mining: build an abbreviation/substitution dictionary from
train matched pairs.

Read-only analysis over the provided training ground truth only (no external
data). For every matched (Source1, Source2/3) pair we tokenize the business
name and address on non-alphanumeric characters and, when exactly one token
differs on each side, record it as a candidate substitution (e.g. "corp" ->
"corporation"). The most frequent substitutions are printed and saved to
``output/substitutions.txt`` for later use as an abbreviation dictionary.
"""

import csv
import gc
import re
import sys
from collections import Counter

import pandas as pd

# Force UTF-8 stdout: Windows consoles/redirected files default to cp1252,
# which cannot encode the Devanagari/accented text in business_name/address.
sys.stdout.reconfigure(encoding="utf-8")

from config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH, OUTPUT_DIR
from io_utils import split_ids

CHUNK = 300_000
TOKEN_RE = re.compile(r"[a-z0-9]+")
TOP_N = 60


def chunks(path, usecols):
    """Yield DataFrame chunks of ``path`` restricted to ``usecols``."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        usecols=usecols,
        chunksize=CHUNK,
    )


def tokenize(text):
    """Lowercase ``text`` and split it into alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())


def collect_matched_ids():
    """Stream the ground truth and return (matched_s2_ids, matched_s3_ids) sets."""
    s2_ids, s3_ids = set(), set()
    for chunk in chunks(TRAIN_GROUND_TRUTH, usecols=["matched_entity_ids"]):
        for raw in chunk["matched_entity_ids"]:
            for mid in split_ids(raw):
                if mid.startswith("S2-"):
                    s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    s3_ids.add(mid)
    return s2_ids, s3_ids


def build_text_lookup(path, wanted_ids=None):
    """Stream ``path`` and return ``entity_id -> (business_name, business_address)``.

    If ``wanted_ids`` is given, only matching rows are kept (bounds memory
    when only a subset of a large Source-2/3 file is needed).
    """
    d = {}
    for chunk in chunks(path, usecols=["entity_id", "business_name", "business_address"]):
        if wanted_ids is not None:
            chunk = chunk[chunk["entity_id"].isin(wanted_ids)]
        d.update(zip(chunk["entity_id"], zip(chunk["business_name"], chunk["business_address"])))
    return d


def mine_substitutions(prefix, s1_text, other_text, name_counter, addr_counter):
    """Stream the ground truth and mine single-token substitutions for one source.

    ``prefix`` is "S2-" or "S3-": only matched ids with that prefix are
    considered on this pass. For each pair, tokenizes name and address; when
    the two records differ by exactly one token on each side, records the
    (token_from_S1, token_from_other) pair in the relevant Counter.
    """
    for chunk in chunks(TRAIN_GROUND_TRUTH, usecols=["source1_entity_id", "matched_entity_ids"]):
        for s1_id, raw in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            for mid in split_ids(raw):
                if not mid.startswith(prefix):
                    continue
                s1_rec = s1_text.get(s1_id)
                other_rec = other_text.get(mid)
                if s1_rec is None or other_rec is None:
                    continue
                for idx, counter in ((0, name_counter), (1, addr_counter)):
                    tok_a = set(tokenize(s1_rec[idx]))
                    tok_b = set(tokenize(other_rec[idx]))
                    diff_a = tok_a - tok_b
                    diff_b = tok_b - tok_a
                    if len(diff_a) == 1 and len(diff_b) == 1:
                        counter[(next(iter(diff_a)), next(iter(diff_b)))] += 1


def write_report(name_counter, addr_counter, path):
    """Write the top substitution pairs for names and addresses to ``path``."""
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for label, counter in (("NAME", name_counter), ("ADDRESS", addr_counter)):
            f.write(f"=== Top {TOP_N} {label} substitutions (token_a -> token_b : count) ===\n")
            for (a, b), n in counter.most_common(TOP_N):
                f.write(f"{a} -> {b} : {n}\n")
            f.write("\n")


def main():
    """Mine train matched pairs for name/address token substitutions and save them."""
    print("Collecting matched S2/S3 ids from ground truth ...")
    matched_s2_ids, matched_s3_ids = collect_matched_ids()
    print(f"  matched S2 ids: {len(matched_s2_ids)}, matched S3 ids: {len(matched_s3_ids)}")

    print("Building Source1 text lookup (full) ...")
    s1_text = build_text_lookup(TRAIN_SOURCE1)

    name_counter, addr_counter = Counter(), Counter()

    print("Mining Source1 <-> Source2 substitutions ...")
    s2_text = build_text_lookup(TRAIN_SOURCE2, wanted_ids=matched_s2_ids)
    mine_substitutions("S2-", s1_text, s2_text, name_counter, addr_counter)
    del s2_text
    gc.collect()

    print("Mining Source1 <-> Source3 substitutions ...")
    s3_text = build_text_lookup(TRAIN_SOURCE3, wanted_ids=matched_s3_ids)
    mine_substitutions("S3-", s1_text, s3_text, name_counter, addr_counter)
    del s3_text, s1_text
    gc.collect()

    print(f"\nTop {TOP_N} NAME substitutions:")
    for (a, b), n in name_counter.most_common(TOP_N):
        print(f"  {a} -> {b} : {n}")

    print(f"\nTop {TOP_N} ADDRESS substitutions:")
    for (a, b), n in addr_counter.most_common(TOP_N):
        print(f"  {a} -> {b} : {n}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "substitutions.txt"
    write_report(name_counter, addr_counter, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
