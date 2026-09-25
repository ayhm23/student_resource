"""Step 1 sanity check of the F0.5 metric implementation in ``io_utils.f05``.

Scores two fake prediction sets against the train ground truth: an all-empty
baseline (should equal the singleton rate) and a perfect baseline (should be
1.0). This is the safety net that catches a bug in the metric itself before
it is used to evaluate anything real.
"""

import csv
import sys

import pandas as pd

from config import TRAIN_GROUND_TRUTH
from io_utils import split_ids, f05

sys.stdout.reconfigure(encoding="utf-8")

CHUNK = 300_000


def score_baseline(predict_fn):
    """Macro-average f05 over train ground truth using ``predict_fn(true_set)``."""
    total = 0.0
    n = 0
    for chunk in pd.read_csv(
        TRAIN_GROUND_TRUTH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        chunksize=CHUNK,
    ):
        for raw in chunk["matched_entity_ids"]:
            true_set = set(split_ids(raw))
            pred_set = predict_fn(true_set)
            total += f05(pred_set, true_set)
            n += 1
    return total / n, n


def main():
    """Run and print the all-empty and perfect-prediction sanity checks."""
    score, n = score_baseline(lambda true_set: set())
    print(f"All-empty predictions on train: macro F0.5 = {score:.6f} over {n} entities "
          f"(should equal the singleton rate).")

    score, n = score_baseline(lambda true_set: set(true_set))
    print(f"Perfect predictions on train:   macro F0.5 = {score:.6f} over {n} entities "
          f"(should be 1.0).")


if __name__ == "__main__":
    main()
