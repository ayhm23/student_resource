"""Track 5 follow-up: persist OOF probabilities and report the 20 worst OOF entities.

Retrains the identical 5-fold LightGBM model (same seed, same data) as
``train_model.py`` -- deterministic, so this reproduces the same OOF values
-- and additionally:
  * saves OOF probabilities to ``data/features/oof_probs.parquet``
  * applies the best decision rule found by ``train_model.py`` (relative,
    t=0.65, alpha=0.7) to build per-S1 predictions from OOF
  * finds the 20 non-singleton sample S1 entities with the worst per-entity
    F0.5 and prints their raw records (S1 + its true matches) alongside what
    was actually predicted, for the STEP2_REPORT.md error-analysis section.
"""

import sys

import pandas as pd

import train_model as tm
from config import FEATURES_DIR, OUTPUT_DIR, TRAIN_GT_LONG_PARQUET, raw_parquet_path
from io_utils import f05, split_ids
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BEST_VARIANT, BEST_T, BEST_ALPHA = "relative", 0.65, 0.7


def raw_text_lookup(ids, split, source):
    """Return {id: (business_name, business_address)} for the given ids from one raw source."""
    path = raw_parquet_path(split, source).as_posix()
    import duckdb
    con = duckdb.connect()
    ids_list = list(ids)
    rows = con.execute(
        f"SELECT id, business_name, business_address FROM '{path}' WHERE id IN "
        f"({','.join(str(i) for i in ids_list)})"
    ).fetchall()
    con.close()
    return {i: (n, a) for i, n, a in rows}


def main():
    """Retrain (deterministic), save OOF probs, and report the 20 worst OOF entities."""
    print_sysinfo()
    with stage("load + train (reproduces train_model.py's OOF exactly)"):
        df = tm.load_train_candidates()
        oof, models, importances = tm.train_cv(df)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = df[["s1_id", "match_id", "other_source", "label", "fold"]].copy()
    oof_df["oof_prob"] = oof
    oof_path = FEATURES_DIR / "oof_probs.parquet"
    oof_df.to_parquet(oof_path)
    print(f"Saved OOF probabilities -> {oof_path}")

    with stage("build OOF predictions with the best decision rule"):
        work = df[["s1_id", "match_id", "other_source"]].copy()
        work["p"] = oof
        excl_work = tm._apply_exclusivity(work)
        best_p = tm._best_p_per_s1(df, oof)
        rel_work = excl_work.join(best_p.rename("s1_best_p"), on="s1_id")
        rel_work = rel_work[rel_work["p"] >= BEST_ALPHA * rel_work["s1_best_p"]]
        pred_map = tm._pred_map_from_work(rel_work, BEST_T)

    gt = pd.read_parquet(TRAIN_GT_LONG_PARQUET)
    gt_map = {}
    for s1_id, sub in gt.groupby("s1_id"):
        gt_map[s1_id] = set(f"{src}-{mid}" for mid, src in zip(sub["match_id"], sub["match_source"]))

    sample_ids = df["s1_id"].unique().tolist()
    scored = []
    for s1_id in sample_ids:
        true_set = gt_map.get(s1_id, set())
        if not true_set:
            continue  # worst-entity report is over non-singletons, per the task spec
        pred_set = pred_map.get(s1_id, set())
        scored.append((s1_id, f05(pred_set, true_set), true_set, pred_set))
    scored.sort(key=lambda x: x[1])
    worst20 = scored[:20]

    print("\n=== 20 worst OOF non-singleton entities (lowest per-entity F0.5) ===")
    all_ids = set()
    for s1_id, _, true_set, pred_set in worst20:
        all_ids.add(s1_id)
        for mid in true_set | pred_set:
            all_ids.add(mid)
    s1_ids_needed = [i for i in all_ids if isinstance(i, int)]
    s2_ids_needed = [int(i.split("-")[1]) for i in all_ids if isinstance(i, str) and i.startswith("S2-")]
    s3_ids_needed = [int(i.split("-")[1]) for i in all_ids if isinstance(i, str) and i.startswith("S3-")]
    s1_text = raw_text_lookup(s1_ids_needed, "train", "S1") if s1_ids_needed else {}
    s2_text = raw_text_lookup(s2_ids_needed, "train", "S2") if s2_ids_needed else {}
    s3_text = raw_text_lookup(s3_ids_needed, "train", "S3") if s3_ids_needed else {}

    def lookup(id_str_or_int):
        if isinstance(id_str_or_int, int):
            return s1_text.get(id_str_or_int, ("?", "?"))
        src, mid = id_str_or_int.split("-")
        mid = int(mid)
        return (s2_text if src == "S2" else s3_text).get(mid, ("<missing>", "<missing>"))

    report_lines = []
    for s1_id, entity_f05, true_set, pred_set in worst20:
        name, addr = lookup(s1_id)
        line = f"\nS1-{s1_id} (F0.5={entity_f05:.3f}): {name!r} | {addr!r}"
        print(line); report_lines.append(line)
        line = f"  TRUE matches ({len(true_set)}):"
        print(line); report_lines.append(line)
        for mid in sorted(true_set):
            n, a = lookup(mid)
            line = f"    {mid}: {n!r} | {a!r}"
            print(line); report_lines.append(line)
        line = f"  PREDICTED ({len(pred_set)}):"
        print(line); report_lines.append(line)
        for mid in sorted(pred_set):
            n, a = lookup(mid)
            tag = "correct" if mid in true_set else "WRONG"
            line = f"    {mid} [{tag}]: {n!r} | {a!r}"
            print(line); report_lines.append(line)

    out_path = OUTPUT_DIR / "worst20_oof_entities.txt"
    out_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
