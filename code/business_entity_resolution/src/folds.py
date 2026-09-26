"""Step 2 Track 1: the 300k stratified train-S1 sample used for blocking-recall
measurement (``blocking.py``'s ``measure_recall``), and the 5-fold GroupKFold
assignment over ALL train S1 ids that every later training/evaluation script
should reuse, so folds are consistent across the whole team.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import FOLDS_PARQUET, TRAIN_SAMPLE_S1_PARQUET, raw_parquet_path
from db import connect
from perf import print_sysinfo, stage

sys.stdout.reconfigure(encoding="utf-8")

SAMPLE_SIZE = 300_000
N_FOLDS = 5
SEED = 42


def load_id_country(con):
    """Pull only id/country for all train S1 rows, skipping the large text columns."""
    path = raw_parquet_path("train", "S1").as_posix()
    return con.execute(f"SELECT id, country FROM '{path}'").fetchdf()


def compute_quotas(counts, total_sample):
    """Proportionally allocate total_sample across countries by their row share.

    Largest-remainder rounding makes quotas sum exactly to total_sample; each
    quota is then clamped to that country's available row count so a country
    can never be asked for more ids than it has (defensive -- with only
    US/India in this train set no country actually falls short).
    """
    grand_total = sum(counts.values())
    exact = {c: total_sample * n / grand_total for c, n in counts.items()}
    quotas = {c: int(v) for c, v in exact.items()}
    remainder = total_sample - sum(quotas.values())
    by_frac = sorted(exact, key=lambda c: exact[c] - int(exact[c]), reverse=True)
    for c in by_frac[:remainder]:
        quotas[c] += 1
    return {c: min(q, counts[c]) for c, q in quotas.items()}


def build_stratified_sample(df):
    """Sample SAMPLE_SIZE ids stratified by each country's proportional share (seed 42)."""
    counts = df["country"].value_counts().to_dict()
    quotas = compute_quotas(counts, SAMPLE_SIZE)
    parts = [
        df[df["country"] == country].sample(n=quota, random_state=SEED)
        for country, quota in quotas.items()
    ]
    return pd.concat(parts, ignore_index=True)


def build_folds(df):
    """Assign every train S1 id to one of N_FOLDS GroupKFold folds (seed 42).

    Each row is already its own group (one S1 id per row) so this reduces to
    a balanced random partition; GroupKFold is used anyway per spec so that
    reusing this fold file at a pair level later (multiple pairs sharing one
    S1 id) still guarantees no train/val leakage.
    """
    ids = df["id"].to_numpy()
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = np.empty(len(ids), dtype=np.int8)
    for fold, (_, val_idx) in enumerate(gkf.split(ids, groups=ids)):
        fold_of[val_idx] = fold
    return pd.DataFrame({"id": ids, "fold": fold_of})


def main():
    """Build the 300k stratified sample and the full 5-fold GroupKFold assignment."""
    print_sysinfo()
    con = connect(memory_limit_gb=4, threads=4)

    with stage("load id/country"):
        df = load_id_country(con)
    con.close()

    with stage("build stratified sample"):
        sample = build_stratified_sample(df)
        sample.to_parquet(TRAIN_SAMPLE_S1_PARQUET, index=False)
    print(f"[sample] {len(sample)} rows -> {TRAIN_SAMPLE_S1_PARQUET}")
    print(sample["country"].value_counts())

    with stage("build folds"):
        folds = build_folds(df)
        folds.to_parquet(FOLDS_PARQUET, index=False)
    print(f"[folds] {len(folds)} rows -> {FOLDS_PARQUET}")

    merged = folds.merge(df, on="id")
    print("[folds] per-fold row counts:")
    print(merged.groupby("fold").size())
    print("[folds] per-fold per-country breakdown:")
    print(merged.groupby(["fold", "country"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
