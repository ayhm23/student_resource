"""Shared I/O helpers: reading source TSVs, id-list parsing/writing, and the
F0.5 scoring metric used to evaluate matches offline.
"""

import csv

import pandas as pd


def read_tsv(path, usecols=None):
    """Read a challenge TSV file with the required strict, lossless settings.

    ``dtype=str`` and ``keep_default_na=False`` stop pandas from turning
    empty-string fields or literal tokens like "NA" into NaN, and
    ``quoting=csv.QUOTE_NONE`` disables quote handling since commas inside
    addresses are not quoted in these files.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        usecols=usecols,
    )


def split_ids(raw):
    """Split a comma-separated id-list cell into a list of non-empty ids.

    An empty or whitespace-only string (the encoding for "no matches")
    returns an empty list.
    """
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    return raw.split(",")


def write_id_list_tsv(path, header_col, mapping):
    """Write a ``source1_entity_id -> matched/candidate_entity_ids`` TSV.

    ``mapping`` is a dict of ``{source1_id: iterable_of_ids}``. Rows are
    written in ``mapping`` iteration order, one row per source1 id, with
    duplicate ids within a row's list removed while preserving first-seen
    order. Written with plain Python string joins (not pandas) so the exact
    tab/comma/newline formatting the validator expects is guaranteed.
    """
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{header_col}\n")
        for s1_id, ids in mapping.items():
            seen = set()
            deduped = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    deduped.append(i)
            f.write(f"{s1_id}\t{','.join(deduped)}\n")


def f05(pred: set, true: set) -> float:
    """Compute the per-entity F0.5 score exactly as defined by the challenge.

    A singleton (``true`` empty) scores 1.0 if ``pred`` is also empty, else
    0.0. Otherwise F0.5 = 1.25 * P * R / (0.25 * P + R), which is 0.0 whenever
    ``pred`` is empty (recall 0) or shares nothing with ``true``.
    """
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    return (1.25 * precision * recall) / (0.25 * precision + recall)
