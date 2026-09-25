"""Step 1 EDA / data-quality checks for the Amazon ML Challenge 2026 business
entity resolution dataset.

Read-only: no external lookups, no model training. Everything here is
computed from the provided train/test TSVs, streamed in chunks so the whole
~2.5 GB dataset never has to sit in memory at once. Run from anywhere; paths
come from ``config.py``.
"""

import csv
import gc
import random
import re
import sys
from collections import Counter

import pandas as pd

# Windows consoles/redirected files default stdout to the cp1252 codepage,
# which cannot encode the Devanagari/accented text found in business_name and
# business_address; force UTF-8 so printing never crashes mid-run.
sys.stdout.reconfigure(encoding="utf-8")

from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
)
from io_utils import split_ids

CHUNK = 300_000
TOKEN_RE = re.compile(r"[a-z0-9]+")
random.seed(0)


def chunks(path, usecols):
    """Yield DataFrame chunks of ``path`` restricted to ``usecols``.

    Centralises the mandated strict read settings (tab-separated, string
    dtype, no NA sniffing, no quote handling) for every streaming pass in
    this script.
    """
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


def count_raw_lines(path):
    """Count raw lines in a file, including the header, without pandas."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def tokenize(text):
    """Lowercase ``text`` and split it into alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# a. row counts
# ---------------------------------------------------------------------------

def check_row_counts(files):
    """Print pandas row counts vs. raw line counts for each file.

    ``files`` is a list of (label, path, id_col) tuples. A file is a pass
    only when the pandas row count equals (raw line count - 1 header row).
    """
    print("\n=== (a) Row counts: pandas vs. raw line counts ===")
    for label, path, id_col in files:
        raw = count_raw_lines(path)
        pandas_rows = sum(len(c) for c in chunks(path, usecols=[id_col]))
        status = "OK" if pandas_rows == raw - 1 else "MISMATCH"
        print(f"  {label}: raw_lines={raw} (incl. header), "
              f"raw_data_rows={raw - 1}, pandas_rows={pandas_rows} [{status}]")


# ---------------------------------------------------------------------------
# b. country counts + unseen-country check
# ---------------------------------------------------------------------------

def country_counts(path):
    """Return a Counter of the ``country`` column values in ``path``."""
    counter = Counter()
    for chunk in chunks(path, usecols=["country"]):
        counter.update(chunk["country"].tolist())
    return counter


def check_countries(train_files, test_files):
    """Print country label counts per file and % of unseen-country test rows.

    Returns the set of country labels observed anywhere in train, so later
    checks can flag countries (e.g. France) that never appeared in training.
    """
    print("\n=== (b) Country label counts ===")
    train_countries = set()
    for label, path in train_files:
        c = country_counts(path)
        train_countries |= set(c.keys())
        print(f"  {label}: {dict(c)}")

    for label, path in test_files:
        c = country_counts(path)
        print(f"  {label}: {dict(c)}")
        if label == "test_source1":
            unseen_rows = sum(n for country, n in c.items() if country not in train_countries)
            total_rows = sum(c.values())
            pct = 100.0 * unseen_rows / total_rows if total_rows else 0.0
            print(f"    -> train countries seen: {sorted(train_countries)}")
            print(f"    -> test_source1 rows with a country unseen in train: "
                  f"{unseen_rows}/{total_rows} ({pct:.2f}%)")
    return train_countries


# ---------------------------------------------------------------------------
# g. empty name/address counts
# ---------------------------------------------------------------------------

def check_empty_fields(all_files):
    """Print counts of empty business_name / business_address per file."""
    print("\n=== (g) Empty name/address counts ===")
    for label, path in all_files:
        empty_name = empty_addr = total = 0
        for chunk in chunks(path, usecols=["business_name", "business_address"]):
            total += len(chunk)
            empty_name += (chunk["business_name"] == "").sum()
            empty_addr += (chunk["business_address"] == "").sum()
        print(f"  {label}: empty_name={empty_name}/{total}, "
              f"empty_address={empty_addr}/{total}")


# ---------------------------------------------------------------------------
# c, d, e (partial), i (sample) — first ground-truth pass, no country lookup
# ---------------------------------------------------------------------------

def gt_pass_one():
    """Stream the ground truth once to compute match-count statistics.

    Returns a dict with: singleton_count, total_entities, matches_per_entity
    (Counter of list-length -> #entities), s2_instances/s3_instances (total
    matched-id occurrences by source), id_owner_count (Counter of matched id
    -> number of distinct Source-1 entities it is matched to, i.e. the
    exclusivity signal), and a reservoir sample of 15 (s1_id, matched_ids)
    rows that have at least one match, for the side-by-side printout.
    """
    singleton_count = 0
    total_entities = 0
    matches_per_entity = Counter()
    s2_instances = s3_instances = 0
    id_owner_count = Counter()
    sample = []
    sample_seen = 0

    for chunk in chunks(TRAIN_GROUND_TRUTH, usecols=["source1_entity_id", "matched_entity_ids"]):
        for s1_id, raw in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            total_entities += 1
            ids = split_ids(raw)
            n = len(ids)
            matches_per_entity[n] += 1
            if n == 0:
                singleton_count += 1
                continue
            distinct_ids = set(ids)
            id_owner_count.update(distinct_ids)
            for mid in distinct_ids:
                if mid.startswith("S2-"):
                    s2_instances += 1
                elif mid.startswith("S3-"):
                    s3_instances += 1
            # reservoir sampling, sample size 15, over non-singleton rows only
            sample_seen += 1
            if len(sample) < 15:
                sample.append((s1_id, ids))
            else:
                j = random.randint(0, sample_seen - 1)
                if j < 15:
                    sample[j] = (s1_id, ids)

    return {
        "singleton_count": singleton_count,
        "total_entities": total_entities,
        "matches_per_entity": matches_per_entity,
        "s2_instances": s2_instances,
        "s3_instances": s3_instances,
        "id_owner_count": id_owner_count,
        "sample": sample,
    }


def print_gt_pass_one(stats):
    """Print the singleton rate, match distribution, and S2/S3 match share."""
    print("\n=== (c) Singleton rate / matches-per-entity / S2 vs S3 share ===")
    total = stats["total_entities"]
    singles = stats["singleton_count"]
    print(f"  total S1 entities (train): {total}")
    print(f"  singleton rate: {singles}/{total} ({100.0 * singles / total:.2f}%)")
    print("  matches-per-entity distribution (n_matches: n_entities):")
    for n in sorted(stats["matches_per_entity"]):
        print(f"    {n}: {stats['matches_per_entity'][n]}")
    s2, s3 = stats["s2_instances"], stats["s3_instances"]
    tot = s2 + s3
    print(f"  matched-id instances: S2={s2} ({100.0 * s2 / tot:.2f}%), "
          f"S3={s3} ({100.0 * s3 / tot:.2f}%)" if tot else "  no matches found")


# ---------------------------------------------------------------------------
# d, e — exclusivity, distractors, missing-from-files
# ---------------------------------------------------------------------------

def build_train_id_country(path):
    """Stream ``path`` and return a dict of ``entity_id -> country``."""
    d = {}
    for chunk in chunks(path, usecols=["entity_id", "country"]):
        d.update(zip(chunk["entity_id"], chunk["country"]))
    return d


def check_exclusivity_and_distractors(id_owner_count, s2_country, s3_country,
                                       s2_total_rows, s3_total_rows):
    """Print exclusivity violations, distractor %, and GT-ids-missing-from-files.

    ``id_owner_count`` maps a matched S2/S3 id to how many distinct S1
    entities it was matched to (from gt_pass_one). Exclusivity is violated
    when that count is > 1.
    """
    print("\n=== (d) Exclusivity: S2/S3 ids matched to more than one S1 ===")
    violations = {mid: n for mid, n in id_owner_count.items() if n > 1}
    print(f"  {len(violations)} S2/S3 id(s) matched to more than one S1 entity "
          f"(out of {len(id_owner_count)} distinct matched ids).")
    if violations:
        example = list(violations.items())[:5]
        print(f"    examples (id: n_owners): {example}")

    print("\n=== (e) Distractors and GT ids missing from files ===")
    matched_s2_ids = {mid for mid in id_owner_count if mid.startswith("S2-")}
    matched_s3_ids = {mid for mid in id_owner_count if mid.startswith("S3-")}

    missing_s2 = matched_s2_ids - s2_country.keys()
    missing_s3 = matched_s3_ids - s3_country.keys()
    print(f"  matched S2 ids not found in train_source2.tsv: {len(missing_s2)}")
    print(f"  matched S3 ids not found in train_source3.tsv: {len(missing_s3)}")

    matched_s2_in_file = matched_s2_ids & s2_country.keys()
    matched_s3_in_file = matched_s3_ids & s3_country.keys()
    distractor_s2 = s2_total_rows - len(matched_s2_in_file)
    distractor_s3 = s3_total_rows - len(matched_s3_in_file)
    print(f"  S2 distractors (no S1 match): {distractor_s2}/{s2_total_rows} "
          f"({100.0 * distractor_s2 / s2_total_rows:.2f}%)")
    print(f"  S3 distractors (no S1 match): {distractor_s3}/{s3_total_rows} "
          f"({100.0 * distractor_s3 / s3_total_rows:.2f}%)")
    return matched_s2_ids, matched_s3_ids


# ---------------------------------------------------------------------------
# f — country mismatches on matched pairs (second ground-truth pass)
# ---------------------------------------------------------------------------

def check_country_mismatch(s1_country, s2_country, s3_country):
    """Stream the ground truth again and count matched pairs whose country differs."""
    print("\n=== (f) Country mismatches on matched pairs ===")
    compared = mismatched = skipped = 0
    for chunk in chunks(TRAIN_GROUND_TRUTH, usecols=["source1_entity_id", "matched_entity_ids"]):
        for s1_id, raw in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            ids = split_ids(raw)
            if not ids:
                continue
            c1 = s1_country.get(s1_id)
            for mid in ids:
                c2 = s2_country.get(mid) if mid.startswith("S2-") else s3_country.get(mid)
                if c1 is None or c2 is None:
                    skipped += 1
                    continue
                compared += 1
                if c1 != c2:
                    mismatched += 1
    pct = 100.0 * mismatched / compared if compared else 0.0
    print(f"  compared pairs: {compared}, mismatched country: {mismatched} ({pct:.2f}%), "
          f"skipped (missing id/country): {skipped}")


# ---------------------------------------------------------------------------
# i — sample matched pairs + France test records
# ---------------------------------------------------------------------------

def build_text_lookup(path, wanted_ids=None):
    """Stream ``path`` and return ``entity_id -> (business_name, business_address)``.

    If ``wanted_ids`` is given, only rows whose id is in that set are kept,
    which keeps memory bounded when scanning a large Source-2/3 file for a
    small subset of matched ids.
    """
    d = {}
    for chunk in chunks(path, usecols=["entity_id", "business_name", "business_address"]):
        if wanted_ids is not None:
            chunk = chunk[chunk["entity_id"].isin(wanted_ids)]
        d.update(zip(chunk["entity_id"], zip(chunk["business_name"], chunk["business_address"])))
    return d


def print_sample_pairs(sample, s1_text, s2_text, s3_text):
    """Print up to 15 sampled matched pairs, S1 record vs. each of its matches."""
    print("\n=== (i) 15 random matched pairs, side by side ===")
    for s1_id, ids in sample:
        name1, addr1 = s1_text.get(s1_id, ("?", "?"))
        print(f"  S1 {s1_id}: {name1!r} | {addr1!r}")
        for mid in ids:
            lookup = s2_text if mid.startswith("S2-") else s3_text
            name2, addr2 = lookup.get(mid, ("<missing>", "<missing>"))
            print(f"    -> {mid}: {name2!r} | {addr2!r}")


def print_france_samples():
    """Stream test_source1.tsv and print the first 8 France records found."""
    print("\n=== (i) 8 France test records ===")
    found = []
    for chunk in chunks(TEST_SOURCE1, usecols=["entity_id", "business_name", "business_address", "country"]):
        fr = chunk[chunk["country"] == "France"]
        if len(fr):
            found.extend(fr.itertuples(index=False))
        if len(found) >= 8:
            break
    for row in found[:8]:
        print(f"  {row.entity_id}: {row.business_name!r} | {row.business_address!r}")


# ---------------------------------------------------------------------------
# h — source2+3 records per source1 record, train vs test
# ---------------------------------------------------------------------------

def check_ratio(label, s1_rows, s2_rows, s3_rows):
    """Print the (Source2 + Source3) rows per Source1 row ratio for one split."""
    ratio = (s2_rows + s3_rows) / s1_rows
    print(f"  {label}: S1={s1_rows}, S2={s2_rows}, S3={s3_rows}, "
          f"(S2+S3)/S1={ratio:.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    """Run every Step 1 EDA check in sequence and print results to stdout."""
    print("ML Challenge 2026 -- Step 1 data checks")

    train_files = [
        ("train_source1", TRAIN_SOURCE1),
        ("train_source2", TRAIN_SOURCE2),
        ("train_source3", TRAIN_SOURCE3),
    ]
    test_files = [
        ("test_source1", TEST_SOURCE1),
        ("test_source2", TEST_SOURCE2),
        ("test_source3", TEST_SOURCE3),
    ]
    all_source_files = train_files + test_files
    row_count_files = (
        [(l, p, "entity_id") for l, p in all_source_files]
        + [("train_ground_truth", TRAIN_GROUND_TRUTH, "source1_entity_id")]
    )

    # a
    check_row_counts(row_count_files)

    # raw row counts, reused for (h) and distractor denominators
    raw_rows = {label: count_raw_lines(path) - 1 for label, path in all_source_files}

    # b
    check_countries(train_files, test_files)

    # g
    check_empty_fields(all_source_files)

    # h
    print("\n=== (h) Source2+3 records per Source1 record ===")
    check_ratio("train", raw_rows["train_source1"], raw_rows["train_source2"], raw_rows["train_source3"])
    check_ratio("test", raw_rows["test_source1"], raw_rows["test_source2"], raw_rows["test_source3"])

    # c, d(part), e(part), i(sample rows)
    gt_stats = gt_pass_one()
    print_gt_pass_one(gt_stats)

    # id -> country lookups needed for d/e/f
    print("\nBuilding id->country lookups for train source1/2/3 ...")
    s1_country = build_train_id_country(TRAIN_SOURCE1)
    s2_country = build_train_id_country(TRAIN_SOURCE2)
    s3_country = build_train_id_country(TRAIN_SOURCE3)

    matched_s2_ids, matched_s3_ids = check_exclusivity_and_distractors(
        gt_stats["id_owner_count"], s2_country, s3_country,
        raw_rows["train_source2"], raw_rows["train_source3"],
    )

    # f
    check_country_mismatch(s1_country, s2_country, s3_country)

    del s2_country, s3_country
    gc.collect()

    # i — text lookups for the sample pairs only need s1 (all) + matched-only s2/s3
    print("\nBuilding text lookups for the sample pairs ...")
    sample_s1_ids = {s1 for s1, _ in gt_stats["sample"]}
    sample_match_ids = {mid for _, ids in gt_stats["sample"] for mid in ids}
    s1_text = build_text_lookup(TRAIN_SOURCE1, wanted_ids=sample_s1_ids)
    s2_text = build_text_lookup(TRAIN_SOURCE2, wanted_ids=sample_match_ids)
    s3_text = build_text_lookup(TRAIN_SOURCE3, wanted_ids=sample_match_ids)
    print_sample_pairs(gt_stats["sample"], s1_text, s2_text, s3_text)

    print_france_samples()

    del s1_country
    gc.collect()

    print("\nStep 1 data checks complete.")


if __name__ == "__main__":
    main()
