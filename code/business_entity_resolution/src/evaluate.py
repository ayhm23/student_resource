"""Step 2 Track 1: macro F0.5 / precision / recall / singleton-accuracy scoring
over a chosen set of S1 ids, with a per-country breakdown.

Reuses ``io_utils.f05`` for the per-entity metric -- it is not redefined here.
Missing predictions/ground truth for an id are treated as an empty set, so an
S1 id that a blocking/candidate step never produced a candidate for still
scores against its TRUE match set (i.e. a blocking miss counts against
recall, it is not simply excluded from the average).
"""

import sys

import pandas as pd

from config import RAW_DIR, TRAIN_GT_WIDE_PARQUET, TRAIN_SAMPLE_S1_PARQUET, raw_parquet_path
from io_utils import f05, split_ids
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


ID_LIST_COLUMNS = ("pred_ids", "matched_entity_ids")


def _to_id_map(obj):
    """Normalize a pred/gt input into a plain ``{s1_id: set(ids)}`` map.

    ``obj`` is either a dict of ``{s1_id: set_of_ids}`` (used as-is) or a
    DataFrame with an ``s1_id`` column plus one of ``ID_LIST_COLUMNS``
    (``pred_ids`` for predictions, ``matched_entity_ids`` for ground truth --
    train_gt_wide's own schema, so it can be passed straight through, and
    also accepted for predictions so a perfect-prediction test can reuse the
    ground truth DataFrame as-is). The ids column holds either a comma-joined
    id-list string per row ("S2-123,S3-456") or an already-iterable of ids.
    """
    if isinstance(obj, dict):
        return {k: set(v) for k, v in obj.items()}
    ids_col = next((c for c in ID_LIST_COLUMNS if c in obj.columns), None)
    if ids_col is None:
        raise ValueError(f"DataFrame must have one of {ID_LIST_COLUMNS}")
    mapping = {}
    for s1_id, raw in zip(obj["s1_id"], obj[ids_col]):
        if isinstance(raw, str):
            mapping[s1_id] = set(split_ids(raw))
        else:
            mapping[s1_id] = set(raw) if raw is not None else set()
    return mapping


def _country_lookup():
    """Read id->country for every train S1 row (only these two columns)."""
    df = pd.read_parquet(raw_parquet_path("train", "S1"), columns=["id", "country"])
    return dict(zip(df["id"], df["country"]))


def score(pred, gt, s1_ids):
    """Compute macro F0.5 and related metrics over ``s1_ids``.

    ``pred`` and ``gt`` each accept a dict ``{s1_id: set(ids)}`` or a
    DataFrame with column ``s1_id`` plus an ids column (``pred_ids`` for
    predictions, ``matched_entity_ids`` for ground truth -- train_gt_wide's
    own schema, so it can be passed straight through). An id in ``s1_ids``
    absent from ``gt`` is a true singleton (empty true set); absent from
    ``pred`` is an empty prediction -- so blocking/candidate misses are
    scored against the real match set, not silently dropped.

    Precision/recall use the same singleton convention as ``f05``: a true
    singleton scores precision=recall=1.0 if predicted empty, else 0.0/0.0
    (never skipped/undefined), so ``mean_precision``/``mean_recall`` are
    directly comparable macro averages over the same population as
    ``macro_f05``.

    Returns a dict with ``macro_f05``, ``f05_by_country`` (dict country ->
    macro F0.5 restricted to that country's ids), ``singleton_accuracy``
    (fraction of true singletons predicted empty), ``mean_precision``, and
    ``mean_recall``.
    """
    pred_map = _to_id_map(pred)
    gt_map = _to_id_map(gt)
    countries = _country_lookup()

    f05_scores = []
    precisions = []
    recalls = []
    singleton_hits = []
    f05_by_country = {}

    for s1_id in s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())

        entity_f05 = f05(pred_set, true_set)
        f05_scores.append(entity_f05)

        if not true_set:
            singleton_hits.append(1.0 if not pred_set else 0.0)
            precision = recall = 1.0 if not pred_set else 0.0
        elif not pred_set:
            precision = recall = 0.0
        else:
            tp = len(pred_set & true_set)
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
        precisions.append(precision)
        recalls.append(recall)

        country = countries.get(s1_id, "UNKNOWN")
        f05_by_country.setdefault(country, []).append(entity_f05)

    n = len(f05_scores)
    return {
        "macro_f05": sum(f05_scores) / n,
        "f05_by_country": {c: sum(v) / len(v) for c, v in f05_by_country.items()},
        "singleton_accuracy": (sum(singleton_hits) / len(singleton_hits)
                                if singleton_hits else float("nan")),
        "mean_precision": sum(precisions) / n,
        "mean_recall": sum(recalls) / n,
    }


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _load_sample_ids():
    """Return the 300k stratified sample's S1 ids as a plain list."""
    df = pd.read_parquet(TRAIN_SAMPLE_S1_PARQUET, columns=["id"])
    return df["id"].tolist()


def test_all_empty_matches_singleton_rate(gt, sample_ids):
    """(a) All-empty predictions on the sample must equal its singleton rate exactly."""
    result = score({}, gt, sample_ids)
    gt_map = _to_id_map(gt)
    singleton_rate = sum(1 for i in sample_ids if not gt_map.get(i, set())) / len(sample_ids)
    passed = result["macro_f05"] == singleton_rate
    print(f"[test a] all-empty predictions: macro_f05={result['macro_f05']:.6f} "
          f"singleton_rate={singleton_rate:.6f} -> {'PASS' if passed else 'FAIL'}")
    return passed, singleton_rate


def test_perfect_predictions_score_one(gt, sample_ids):
    """(b) Predicting the true match set for every id must score exactly 1.0."""
    result = score(gt, gt, sample_ids)
    passed = result["macro_f05"] == 1.0
    print(f"[test b] perfect predictions: macro_f05={result['macro_f05']:.6f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def test_worked_example():
    """(c) Problem-statement worked example must round to 0.714."""
    pred = {1: {"S2-00047", "S2-00193", "S3-00812"}}
    gt = {1: {"S2-00047", "S3-00812"}}
    result = score(pred, gt, [1])
    rounded = round(result["macro_f05"], 3)
    passed = rounded == 0.714
    print(f"[test c] worked example: macro_f05={result['macro_f05']:.6f} rounded={rounded} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    """Run the three unit tests against the real train sample and ground truth."""
    print_sysinfo()

    with stage("load ground truth + sample ids"):
        gt = pd.read_parquet(TRAIN_GT_WIDE_PARQUET)
        sample_ids = _load_sample_ids()

    with stage("test (a) all-empty vs singleton rate"):
        passed_a, singleton_rate = test_all_empty_matches_singleton_rate(gt, sample_ids)

    with stage("test (b) perfect predictions"):
        passed_b = test_perfect_predictions_score_one(gt, sample_ids)

    with stage("test (c) worked example"):
        passed_c = test_worked_example()

    all_passed = passed_a and passed_b and passed_c
    print(f"[summary] singleton_rate={singleton_rate:.6f} all_tests_passed={all_passed}")


if __name__ == "__main__":
    main()
