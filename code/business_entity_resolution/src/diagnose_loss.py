"""Where does the macro F0.5 go? Splits the OOF loss into blocking, model and decision parts.

Reads ``train_model.py``'s OOF output (``data/features/oof_stage2.parquet``,
every training candidate with ``label``, ``p2_cal`` and the rule's ``pred``)
and the training S1 population, and reports:

  * actual OOF macro F0.5 (the rule's predictions)
  * oracle-given-candidates: predict exactly the true matches that ARE among
    the candidates -> the ceiling blocking leaves; 1 - this = blocking loss
  * oracle-singleton-aware variants, error counts (FP, FN inside candidates,
    FN never a candidate), all by country and by true-match-set size
"""

import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, FEATURES_DIR, OUTPUT_DIR
from evaluate import gt_pairs, pair_key, score_fast
from perf import print_sysinfo

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def main():
    print_sysinfo()
    oof = pd.read_parquet(FEATURES_DIR / "oof_stage2.parquet")
    s1 = pd.read_parquet(DATA_DIR / "training_s1.parquet")[["s1_id", "country"]]
    gt = gt_pairs(s1["s1_id"])
    src = np.where(oof["src"].to_numpy() == 1, "S3", "S2")
    oof["key"] = pair_key(oof["match_id"].to_numpy(), src)

    lines = []

    def report(name, pred_df):
        m = score_fast(pred_df, gt, s1)
        by = ", ".join(f"{k} {v:.4f}" for k, v in sorted(m["f05_by_country"].items()))
        lines.append(f"{name:55s} macro F0.5 {m['macro_f05']:.4f}  ({by})  P {m['mean_precision']:.4f} "
                     f"R {m['mean_recall']:.4f}")
        return m

    actual = report("actual OOF (tuned rule)", oof.loc[oof["pred"], ["s1_id", "key"]])
    oracle = report("oracle given candidates (blocking ceiling)", oof.loc[oof["label"] == 1, ["s1_id", "key"]])
    lines.append("")
    lines.append(f"LOSS to blocking (1 - ceiling):          {1 - oracle['macro_f05']:.4f}")
    lines.append(f"LOSS to model + decision (ceiling - act): {oracle['macro_f05'] - actual['macro_f05']:.4f}")

    cand = oof[["s1_id", "key"]].drop_duplicates()
    g = gt.merge(cand.assign(in_cand=True), on=["s1_id", "key"], how="left")
    g["in_cand"] = g["in_cand"].fillna(False).astype(bool)
    pred = oof.loc[oof["pred"], ["s1_id", "key", "label"]]
    fp = int((pred["label"] == 0).sum())
    tp = int((pred["label"] == 1).sum())
    fn_in = int(((oof["label"] == 1) & ~oof["pred"]).sum())
    fn_block = int((~g["in_cand"]).sum())
    lines.append("")
    lines.append(f"true pairs {len(g)}: predicted {tp}, missed inside candidates {fn_in}, "
                 f"never a candidate {fn_block} ({100 * fn_block / max(len(g), 1):.2f}%); false positives {fp}")

    n_true = gt.groupby("s1_id").size()
    s1["n_true"] = s1["s1_id"].map(n_true).fillna(0).astype(int)
    all_found = g.groupby("s1_id")["in_cand"].all()
    s1["all_in_cand"] = s1["s1_id"].map(all_found).fillna(True)
    lines.append("")
    lines.append("by true-set size: share of S1 whose true matches are ALL candidates")
    s1["size_bin"] = pd.cut(s1["n_true"], [-1, 0, 1, 3, 6, 10, 1000], labels=["0", "1", "2-3", "4-6", "7-10", "11+"])
    for b, sub in s1.groupby("size_bin", observed=True):
        lines.append(f"  n_true {b:5s}: {len(sub):8d} S1, all-in-candidates {sub['all_in_cand'].mean():.4f}")

    text = "\n".join(lines)
    print(text)
    (OUTPUT_DIR / "LOSS_DIAGNOSIS.md").write_text("```\n" + text + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
