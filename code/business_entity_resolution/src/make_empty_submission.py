"""Step 1 safe first submission: an all-empty (all-singleton) baseline.

Writes ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv``
with every Source1 test entity mapped to an empty match/candidate list. This
is a valid, format-passing submission with no matching logic at all — it
exists to prove the pipeline plumbing works before any blocking or modeling
is built.
"""

import csv

import pandas as pd

from config import TEST_SOURCE1, OUTPUT_DIR, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH
from io_utils import write_id_list_tsv

CHUNK = 300_000


def read_test_source1_ids():
    """Stream test_source1.tsv and return the list of entity_ids, in file order."""
    ids = []
    for chunk in pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        usecols=["entity_id"],
        chunksize=CHUNK,
    ):
        ids.extend(chunk["entity_id"].tolist())
    return ids


def main():
    """Write an all-empty matching_results.tsv and candidate_pairs.tsv for test."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    ids = read_test_source1_ids()
    print(f"test_source1 entities: {len(ids)}")

    mapping = {s1_id: [] for s1_id in ids}
    write_id_list_tsv(MATCHING_RESULTS_PATH, "matched_entity_ids", mapping)
    write_id_list_tsv(CANDIDATE_PAIRS_PATH, "candidate_entity_ids", mapping)
    print(f"Wrote {MATCHING_RESULTS_PATH}")
    print(f"Wrote {CANDIDATE_PAIRS_PATH}")


if __name__ == "__main__":
    main()
