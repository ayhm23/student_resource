"""Step 2 Track 1: macro F0.5 / precision / recall / singleton-accuracy scoring
over a chosen set of S1 ids, with a per-country breakdown.

Reuses ``io_utils.f05`` for the per-entity metric -- it is not redefined here.
Missing predictions/ground truth for an id are treated as an empty set, so an
S1 id that a blocking/candidate step never produced a candidate for still
scores against its TRUE match set (i.e. a blocking miss counts against
recall, it is not simply excluded from the average).
"""

import sys

import numpy as np
import pandas as pd

from config import TRAIN_GT_LONG_PARQUET, TRAIN_GT_WIDE_PARQUET, TRAIN_SAMPLE_S1_PARQUET, raw_parquet_path
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


def pair_key(match_id, match_source):
    """Integer key for an S2/S3 record: match_id*2 + (source == 'S3'), vectorized."""
    src = np.asarray(match_source)
    return np.asarray(match_id, dtype=np.int64) * 2 + (src == "S3").astype(np.int64)


def score_fast(pred, gt, s1_df):
    """Vectorized equivalent of ``score`` for millions of S1 ids.

    ``pred`` and ``gt``: DataFrames of (s1_id, key) pairs (``key`` from
    ``pair_key``). ``s1_df``: DataFrame (s1_id, country) listing EVERY S1 id to
    score -- ids with no prediction score an empty set, ids absent from ``gt``
    are true singletons -- exactly the conventions of ``score``/``io_utils.f05``.
    """
    s1 = s1_df[["s1_id", "country"]].drop_duplicates("s1_id").set_index("s1_id")
    idx = s1.index
    pred = pred[["s1_id", "key"]].drop_duplicates()
    gt = gt[["s1_id", "key"]]
    n_pred = pred.groupby("s1_id").size().reindex(idx, fill_value=0).to_numpy(dtype=np.float64)
    n_true = gt.groupby("s1_id").size().reindex(idx, fill_value=0).to_numpy(dtype=np.float64)
    tp = (pred.merge(gt, on=["s1_id", "key"]).groupby("s1_id").size()
          .reindex(idx, fill_value=0).to_numpy(dtype=np.float64))

    singleton = n_true == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(singleton, (n_pred == 0).astype(np.float64),
                     np.where(tp > 0, 1.25 * tp / (0.25 * n_true + n_pred), 0.0))
        precision = np.where(singleton, (n_pred == 0).astype(np.float64),
                             np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0))
        recall = np.where(singleton, (n_pred == 0).astype(np.float64), tp / np.maximum(n_true, 1))
    by_country = pd.Series(f, index=idx).groupby(s1["country"].to_numpy()).mean().to_dict()
    return {
        "macro_f05": float(f.mean()),
        "f05_by_country": {k: float(v) for k, v in by_country.items()},
        "singleton_accuracy": float((n_pred[singleton] == 0).mean()) if singleton.any() else float("nan"),
        "mean_precision": float(precision.mean()),
        "mean_recall": float(recall.mean()),
    }


def gt_pairs(s1_ids=None):
    """Train ground truth as (s1_id, key) pairs, optionally restricted to ``s1_ids``."""
    g = pd.read_parquet(TRAIN_GT_LONG_PARQUET)
    if s1_ids is not None:
        g = g[g["s1_id"].isin(set(s1_ids))]
    return pd.DataFrame({"s1_id": g["s1_id"].to_numpy(),
                         "key": pair_key(g["match_id"].to_numpy(), g["match_source"].to_numpy())})


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


def test_fast_scorer_matches_reference(sample_ids):
    """(d) score_fast must reproduce score() on a noisy random prediction set."""
    rng = np.random.default_rng(0)
    ids = sample_ids[:50_000]
    gtp = gt_pairs(ids)
    keep = rng.random(len(gtp)) < 0.7
    noise = pd.DataFrame({"s1_id": rng.choice(ids, 20_000),
                          "key": rng.integers(0, 10_000_000, 20_000, dtype=np.int64)})
    pred = pd.concat([gtp[keep], noise], ignore_index=True)
    countries = _country_lookup()
    s1_df = pd.DataFrame({"s1_id": ids, "country": [countries[i] for i in ids]})
    fast = score_fast(pred, gtp, s1_df)

    def as_map(df):
        m = {}
        for s, k in zip(df["s1_id"], df["key"]):
            m.setdefault(s, set()).add(k)
        return m
    ref = score(as_map(pred), as_map(gtp), ids)
    passed = abs(fast["macro_f05"] - ref["macro_f05"]) < 1e-9 and all(
        abs(fast["f05_by_country"][c] - v) < 1e-9 for c, v in ref["f05_by_country"].items())
    print(f"[test d] fast scorer {fast['macro_f05']:.6f} vs reference {ref['macro_f05']:.6f} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    """Run the unit tests against the real train sample and ground truth."""
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

    with stage("test (d) fast scorer == reference scorer"):
        passed_d = test_fast_scorer_matches_reference(sample_ids)

    all_passed = passed_a and passed_b and passed_c and passed_d
    print(f"[summary] singleton_rate={singleton_rate:.6f} all_tests_passed={all_passed}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
