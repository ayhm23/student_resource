"""Track 5 follow-up: report the 20 worst out-of-fold training entities.

Reads what ``train_model.py`` saved -- ``data/features/oof_stage2.parquet``
(every training candidate row with its OOF probability and the decision rule's
``pred`` flag) and ``data/training_s1.parquet`` -- so nothing is retrained.
Prints the 20 non-singleton training S1 entities with the lowest per-entity
F0.5, with raw text for the S1 record, its true matches, and what was predicted
(including true matches blocking never produced as candidates).
"""

import sys

import duckdb
import pandas as pd

from config import DATA_DIR, FEATURES_DIR, OUTPUT_DIR, TRAIN_GT_LONG_PARQUET, raw_parquet_path
from io_utils import f05
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)


def raw_text_lookup(ids, source):
    """Return {id: (business_name, business_address)} for the given ids from one raw train source."""
    if not ids:
        return {}
    path = raw_parquet_path("train", source).as_posix()
    rows = duckdb.connect().execute(
        f"SELECT id, business_name, business_address FROM '{path}' WHERE id IN "
        f"({','.join(str(int(i)) for i in ids)})").fetchall()
    return {i: (n, a) for i, n, a in rows}


def main():
    print_sysinfo()
    with stage("load OOF predictions + ground truth"):
        oof = pd.read_parquet(FEATURES_DIR / "oof_stage2.parquet")
        training = pd.read_parquet(DATA_DIR / "training_s1.parquet")
        gt = pd.read_parquet(TRAIN_GT_LONG_PARQUET)
        gt = gt[gt["s1_id"].isin(set(training["s1_id"]))]

    src_name = {0: "S2", 1: "S3"}
    pred_map, cand_map = {}, {}
    for s1, mid, src, pred in zip(oof["s1_id"], oof["match_id"], oof["src"], oof["pred"]):
        key = f"{src_name[int(src)]}-{mid}"
        cand_map.setdefault(s1, set()).add(key)
        if pred:
            pred_map.setdefault(s1, set()).add(key)
    gt_map = {}
    for s1, mid, src in zip(gt["s1_id"], gt["match_id"], gt["match_source"]):
        gt_map.setdefault(s1, set()).add(f"{src}-{mid}")

    scored = sorted(((f05(pred_map.get(s1, set()), true), s1, true) for s1, true in gt_map.items()),
                    key=lambda x: x[0])
    worst = scored[:20]

    need = {"S1": set(), "S2": set(), "S3": set()}
    for _, s1, true in worst:
        need["S1"].add(s1)
        for k in true | pred_map.get(s1, set()):
            src, mid = k.split("-")
            need[src].add(int(mid))
    text = {src: raw_text_lookup(sorted(ids), src) for src, ids in need.items()}

    lines = []
    for score, s1, true in worst:
        pred = pred_map.get(s1, set())
        name, addr = text["S1"].get(s1, ("?", "?"))
        lines.append(f"\nS1-{s1} (F0.5={score:.3f}): {name!r} | {addr!r}")
        lines.append(f"  TRUE matches ({len(true)}):")
        for k in sorted(true):
            src, mid = k.split("-")
            tag = "" if k in cand_map.get(s1, set()) else "  [never a candidate: BLOCKING MISS]"
            n, a = text[src].get(int(mid), ("?", "?"))
            lines.append(f"    {k}: {n!r} | {a!r}{tag}")
        lines.append(f"  PREDICTED ({len(pred)}):")
        for k in sorted(pred):
            src, mid = k.split("-")
            n, a = text[src].get(int(mid), ("?", "?"))
            lines.append(f"    {k} [{'correct' if k in true else 'WRONG'}]: {n!r} | {a!r}")
    print("\n".join(lines))
    out_path = OUTPUT_DIR / "worst20_oof_entities.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
