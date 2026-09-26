"""Track 5: LightGBM baseline model, decision-layer tuning, LOCO check, and
the final test submission.

Pipeline (see ``main()``):
  1. Load the labeled train-sample candidate features (Track 4 Phase 2) and
     join in the pre-computed 5-fold assignment (``data/folds.parquet``).
  2. Train a 5-fold LightGBM binary classifier with early stopping; save OOF
     probabilities and per-fold feature importances.
  3. Tune three decision-rule variants on the OOF probabilities against the
     TRUE ground-truth match sets (via ``evaluate.score``, so blocking misses
     count): (a) a global threshold, (b) exclusivity (each S2/S3 kept for
     only its highest-probability S1) + threshold, (c) exclusivity + a
     relative margin against that S1's own best candidate. Grid-searches
     threshold/alpha and reports macro F0.5 (overall, per country),
     precision, recall, singleton accuracy for each.
  4. LOCO: retrain on US-only / India-only train-sample rows, score on the
     other country, using the best decision rule found in step 3.
  5. Apply the 5-fold-averaged model + best decision rule to the full test
     candidate features, write ``output/matching_results.tsv`` and
     ``output/candidate_pairs.tsv``, and validate them.
  6. Append a row to ``output/results_log.csv``.
"""

import subprocess
import sys
import time
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import (CANDIDATE_PAIRS_PATH, FEATURES_DIR, FOLDS_PARQUET,
                     MATCHING_RESULTS_PATH, OUTPUT_DIR, RESULTS_LOG_PATH,
                     ROOT_DIR, TRAIN_GT_WIDE_PARQUET, TRAIN_SAMPLE_S1_PARQUET,
                     raw_parquet_path)
from db import connect
from evaluate import score
from io_utils import write_id_list_tsv
from perf import print_sysinfo, stage

TEST_PREDICT_BATCH_ROWS = 200_000

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SEED = 42
NUMERIC_FEATURES = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaro_winkler", "name_nospace_ratio", "name_nospace_contains", "name_sorted_chars_ratio",
    "name_idf_jaccard", "name_idf_rarest_shared", "name_idf_rarest_unshared",
    "name_skeleton_ratio", "name_skeleton_token_set_ratio", "name_skeleton_jaccard",
    "s1_name_len", "other_name_len", "name_len_absdiff",
    "addr_token_set_ratio", "addr_token_sort_ratio", "addr_num_jaccard", "addr_num_max_equal",
    "addr_num_shared_count", "addr_idf_jaccard", "other_addr_empty",
    "n_keys_hit", "cheap_score", "rank_in_s1", "n_candidates_for_s1",
    "n_competitors", "rank_among_competitors",
]
CATEGORICAL_FEATURES = ["legal_form_match", "addr_state_match", "other_source"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

LGB_PARAMS = dict(
    objective="binary", metric="auc", seed=SEED, verbosity=-1,
    num_leaves=63, learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.9,
    bagging_freq=1, min_data_in_leaf=50,
)
NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 50

_DEVICE_PARAMS_CACHE = {}


def detect_lgb_device():
    """Probe whether this LightGBM build+machine can actually train on GPU; cache the result.

    Tries a tiny real training call with ``device_type='gpu'`` (not just an
    import/flag check, since the standard PyPI wheel is usually built
    CPU-only and only errors out once you actually try to use the GPU tree
    learner) and falls back to plain CPU params if it raises. Prints which
    device won and why, once, so it's obvious from the log which path was
    used -- never silently falls back.
    """
    if "device_params" in _DEVICE_PARAMS_CACHE:
        return _DEVICE_PARAMS_CACHE["device_params"]

    probe_X = pd.DataFrame({"x": np.random.rand(200)})
    probe_y = (probe_X["x"] > 0.5).astype(int)
    probe_set = lgb.Dataset(probe_X, probe_y)
    try:
        lgb.train({"objective": "binary", "verbosity": -1, "device_type": "gpu"},
                   probe_set, num_boost_round=2)
        device_params = {"device_type": "gpu"}
        print("[train_model] GPU detected and usable -- training LightGBM on GPU.")
    except Exception as exc:
        device_params = {"device_type": "cpu"}
        print(f"[train_model] GPU not usable ({type(exc).__name__}: {exc}) -- falling back to CPU. "
              f"(Common cause: the pip-installed lightgbm wheel is CPU-only; a GPU-enabled build "
              f"needs a CUDA/OpenCL-compiled install.)")
    _DEVICE_PARAMS_CACHE["device_params"] = device_params
    return device_params

T_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
ALPHA_GRID = np.round(np.arange(0.50, 1.00, 0.05), 2)


def prep_categoricals(df):
    """Cast the categorical feature columns to pandas 'category' dtype, in place, consistently."""
    df = df.copy()
    for c in CATEGORICAL_FEATURES:
        df[c] = df[c].fillna("missing").astype("category")
    return df


def load_train_candidates():
    """Load the labeled train-sample candidate features joined with fold assignment."""
    df = pd.read_parquet(FEATURES_DIR / "trainsample_candidates_features.parquet")
    folds = pd.read_parquet(FOLDS_PARQUET).rename(columns={"id": "s1_id"})
    df = df.merge(folds, on="s1_id", how="left")
    if df["fold"].isna().any():
        raise ValueError("Some train-sample candidate rows have no fold assignment.")
    df["fold"] = df["fold"].astype(int)
    return prep_categoricals(df)


def train_cv(df, feature_cols=ALL_FEATURES, seed=SEED):
    """5-fold train (using the pre-assigned fold column), returning OOF probs, models, importances."""
    device_params = detect_lgb_device()
    oof = np.full(len(df), np.nan)
    models = []
    importances = []
    folds = sorted(df["fold"].unique())
    for fold_idx, fold in enumerate(folds):
        fold_t0 = time.perf_counter()
        tr = df[df["fold"] != fold]
        va = df[df["fold"] == fold]
        train_set = lgb.Dataset(tr[feature_cols], tr["label"], categorical_feature=CATEGORICAL_FEATURES)
        val_set = lgb.Dataset(va[feature_cols], va["label"], categorical_feature=CATEGORICAL_FEATURES,
                               reference=train_set)
        params = dict(LGB_PARAMS, seed=seed, **device_params)
        model = lgb.train(
            params, train_set, num_boost_round=NUM_BOOST_ROUND, valid_sets=[val_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(period=0)],
        )
        preds = model.predict(va[feature_cols], num_iteration=model.best_iteration)
        oof[df["fold"].values == fold] = preds
        models.append(model)
        importances.append(pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols))
        pct = 100.0 * (fold_idx + 1) / len(folds)
        print(f"  fold {fold_idx + 1}/{len(folds)} done in {time.perf_counter() - fold_t0:.1f}s "
              f"({pct:.0f}% of CV training complete)")
    return oof, models, importances


def _pred_map_from_work(work, t):
    """Threshold an already-prepared (s1_id, match_id, other_source, p[, s1_best_p]) frame into a pred map."""
    kept = work[work["p"] >= t]
    pred_map = {}
    for s1_id, match_id, other_source in zip(kept["s1_id"], kept["match_id"], kept["other_source"]):
        pred_map.setdefault(s1_id, set()).add(f"{other_source}-{match_id}")
    return pred_map


def build_pred_map(df, probs, variant, t, alpha=None):
    """Build a {s1_id: set(prefixed_ids)} prediction map from per-row probabilities.

    ``variant``: 'global' (p>=t), 'exclusive' (each S2/S3 kept only for its
    argmax-probability S1, then p>=t), or 'relative' (exclusive + also
    require p >= alpha * that S1's own best candidate probability). Single-
    shot convenience wrapper (LOCO uses this); the OOF grid search below
    precomputes the expensive exclusivity/best-p steps once instead of
    recomputing them per grid point.
    """
    work = df[["s1_id", "match_id", "other_source"]].copy()
    work["p"] = probs
    if variant in ("exclusive", "relative"):
        work = _apply_exclusivity(work)
    if variant == "relative":
        work = _apply_relative(work, df, probs, alpha)
    return _pred_map_from_work(work, t)


def _apply_exclusivity(work):
    """Keep, per (match_id, other_source) group, only the row with the highest probability."""
    # observed=True: other_source is a pandas 'category' dtype, and pandas'
    # default observed=False groups over every category combination (even
    # ones with zero rows), which makes idxmax raise on the empty groups.
    idx_max = work.groupby(["match_id", "other_source"], observed=True)["p"].idxmax()
    keep_mask = pd.Series(False, index=work.index)
    keep_mask.loc[idx_max] = True
    return work[keep_mask]


def _best_p_per_s1(df, probs):
    """Return a {s1_id: max probability among its own candidates} Series, computed once."""
    tmp = pd.DataFrame({"s1_id": df["s1_id"].values, "p": probs})
    return tmp.groupby("s1_id")["p"].max()


def _apply_relative(work, df, probs, alpha):
    """Filter ``work`` (already exclusivity-applied) to rows meeting the relative-margin rule."""
    best_p = _best_p_per_s1(df, probs)
    work = work.join(best_p.rename("s1_best_p"), on="s1_id")
    return work[work["p"] >= alpha * work["s1_best_p"]]


def tune_decision_rule(df, probs, all_s1_ids, gt):
    """Grid-search t (and alpha) for each decision-rule variant; return (results, best).

    The expensive per-row prep (exclusivity argmax, per-S1 best probability)
    depends only on the model's probabilities, never on t/alpha, so each is
    computed exactly once and reused across the whole grid -- not once per
    grid point, which would make the "relative" variant's 16x10 grid
    recompute an identical full-table groupby 160 times.
    """
    results = []
    base_work = df[["s1_id", "match_id", "other_source"]].copy()
    base_work["p"] = probs

    for t in T_GRID:
        m = score(_pred_map_from_work(base_work, t), gt, all_s1_ids)
        results.append({"variant": "global", "t": t, "alpha": None, **m})

    excl_work = _apply_exclusivity(base_work)
    for t in T_GRID:
        m = score(_pred_map_from_work(excl_work, t), gt, all_s1_ids)
        results.append({"variant": "exclusive", "t": t, "alpha": None, **m})

    best_p = _best_p_per_s1(df, probs)
    rel_base = excl_work.join(best_p.rename("s1_best_p"), on="s1_id")
    for alpha in ALPHA_GRID:
        rel_work = rel_base[rel_base["p"] >= alpha * rel_base["s1_best_p"]]
        for t in T_GRID:
            m = score(_pred_map_from_work(rel_work, t), gt, all_s1_ids)
            results.append({"variant": "relative", "t": t, "alpha": alpha, **m})

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["macro_f05"].idxmax()]
    return results_df, best_row


def print_decision_summary(results_df, best_row):
    """Print the top result per variant and the overall best."""
    print("\nBest result per decision-rule variant:")
    for variant in ("global", "exclusive", "relative"):
        sub = results_df[results_df["variant"] == variant]
        if sub.empty:
            continue
        row = sub.loc[sub["macro_f05"].idxmax()]
        print(f"  {variant:10s} t={row['t']:.2f} alpha={row['alpha']}: "
              f"macro_f05={row['macro_f05']:.4f} precision={row['mean_precision']:.4f} "
              f"recall={row['mean_recall']:.4f} singleton_acc={row['singleton_accuracy']:.4f} "
              f"by_country={row['f05_by_country']}")
    print(f"\nOVERALL BEST: variant={best_row['variant']} t={best_row['t']:.2f} "
          f"alpha={best_row['alpha']}: macro_f05={best_row['macro_f05']:.4f}")


def run_loco(df, best_variant, best_t, best_alpha, all_s1_ids_by_country, gt):
    """Train on one country's train-sample rows, score on the other, with the tuned decision rule."""
    lines = []
    for train_country, test_country in (("US", "India"), ("India", "US")):
        train_rows = df[df["s1_country"] == train_country] if "s1_country" in df.columns else None
        if train_rows is None or train_rows.empty:
            lines.append(f"LOCO {train_country}->{test_country}: SKIPPED (no s1_country column / no rows)")
            continue
        train_set = lgb.Dataset(train_rows[ALL_FEATURES], train_rows["label"],
                                 categorical_feature=CATEGORICAL_FEATURES)
        model = lgb.train(dict(LGB_PARAMS, seed=SEED, **detect_lgb_device()), train_set, num_boost_round=300)

        test_rows = df[df["s1_country"] == test_country]
        test_probs = model.predict(test_rows[ALL_FEATURES])
        pred_map = build_pred_map(test_rows.reset_index(drop=True), test_probs, best_variant, best_t, best_alpha)
        test_s1_ids = all_s1_ids_by_country.get(test_country, [])
        m = score(pred_map, gt, test_s1_ids)
        lines.append(f"LOCO {train_country}->{test_country}: macro_f05={m['macro_f05']:.4f} "
                     f"precision={m['mean_precision']:.4f} recall={m['mean_recall']:.4f} "
                     f"singleton_acc={m['singleton_accuracy']:.4f}")
        print(lines[-1])
    return lines


def append_results_log(description, cv_overall, cv_us, cv_india, loco_us_to_india, loco_india_to_us, notes):
    """Append one row to output/results_log.csv."""
    row = pd.DataFrame([{
        "date": date.today().isoformat(),
        "git_commit": _git_commit(),
        "description": description,
        "cv_overall": round(cv_overall, 4),
        "cv_us": round(cv_us, 4) if cv_us is not None else "",
        "cv_india": round(cv_india, 4) if cv_india is not None else "",
        "loco_us_to_india": round(loco_us_to_india, 4) if loco_us_to_india is not None else "",
        "loco_india_to_us": round(loco_india_to_us, 4) if loco_india_to_us is not None else "",
        "public_lb": "",
        "notes": notes,
    }])
    row.to_csv(RESULTS_LOG_PATH, mode="a", header=False, index=False)


def _git_commit():
    """Return the current short git commit hash, or '' if unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                              capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def write_submission(pred_map, all_test_s1_ids):
    """Write output/matching_results.tsv (every test S1, empty when unpredicted)."""
    mapping = {f"S1-{i}": sorted(pred_map.get(i, set())) for i in all_test_s1_ids}
    write_id_list_tsv(MATCHING_RESULTS_PATH, "matched_entity_ids", mapping)
    print(f"Wrote {MATCHING_RESULTS_PATH}")


def predict_test_probs_chunked(models, train_categories, features_path):
    """Stream the (large) test candidate features through the models in batches.

    Returns a compact DataFrame (s1_id, match_id, other_source, p) -- NOT the
    full feature table -- since the 59.7M-row x 27-column test feature parquet
    (~3 GB on disk) would balloon to well over the available RAM as a pandas
    DataFrame; only these four columns are needed for the decision layer and
    for writing the submission.
    """
    con = connect(memory_limit_gb=3, threads=2)
    # "other_source" is already inside ALL_FEATURES (via CATEGORICAL_FEATURES);
    # dedupe so it isn't selected twice, which would make chunk["other_source"]
    # return a 2-column DataFrame instead of a Series.
    cols = list(dict.fromkeys(["s1_id", "match_id", "other_source"] + ALL_FEATURES))
    result = con.execute(f"SELECT {', '.join(cols)} FROM '{features_path.as_posix()}'")

    id_chunks, source_chunks, prob_chunks = [], [], []
    n_seen = 0
    for batch in result.to_arrow_reader(TEST_PREDICT_BATCH_ROWS):
        chunk = batch.to_pandas()
        for c in CATEGORICAL_FEATURES:
            chunk[c] = chunk[c].fillna("missing").astype("category").cat.set_categories(train_categories[c])
        chunk_probs = np.mean(
            [m.predict(chunk[ALL_FEATURES], num_iteration=m.best_iteration) for m in models], axis=0
        )
        id_chunks.append(chunk[["s1_id", "match_id"]].copy())
        source_chunks.append(chunk["other_source"].astype(str).values)
        prob_chunks.append(chunk_probs)
        n_seen += len(chunk)
        print(f"  predicted {n_seen} test candidate pairs")
    con.close()

    out = pd.concat(id_chunks, ignore_index=True)
    out["other_source"] = np.concatenate(source_chunks)
    out["p"] = np.concatenate(prob_chunks)
    return out


def main():
    """Run the full Track 5 pipeline: train, tune, LOCO, predict test, log."""
    print_sysinfo()

    with stage("load train-sample candidate features + folds"):
        df = load_train_candidates()
    print(f"  {len(df)} labeled candidate pairs, {df['label'].sum()} positive")

    with stage("5-fold LightGBM training"):
        oof, models, importances = train_cv(df)
    auc_oof = float(pd.Series(oof).corr(df["label"], method="spearman"))
    print(f"  OOF spearman corr with label (sanity, not the real metric): {auc_oof:.4f}")

    mean_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    print("\nTop 20 feature importances (mean gain across folds):")
    print(mean_importance.head(20).to_string())

    gt = pd.read_parquet(TRAIN_GT_WIDE_PARQUET)
    all_sample_s1_ids = pd.read_parquet(TRAIN_SAMPLE_S1_PARQUET)["id"].tolist()

    with stage("tune decision rule on OOF"):
        results_df, best_row = tune_decision_rule(df, oof, all_sample_s1_ids, gt)
    print_decision_summary(results_df, best_row)
    best_variant, best_t, best_alpha = best_row["variant"], float(best_row["t"]), \
        (float(best_row["alpha"]) if pd.notna(best_row["alpha"]) else None)

    all_s1_ids_by_country = {}
    sample_country_df = pd.read_parquet(TRAIN_SAMPLE_S1_PARQUET)
    for c, sub in sample_country_df.groupby("country"):
        all_s1_ids_by_country[c] = sub["id"].tolist()

    with stage("LOCO check"):
        loco_lines = run_loco(df, best_variant, best_t, best_alpha, all_s1_ids_by_country, gt)

    cv_us = results_df.loc[results_df["macro_f05"].idxmax(), "f05_by_country"].get("US")
    cv_india = best_row["f05_by_country"].get("India")
    loco_us_to_india = loco_india_to_us = None
    for line in loco_lines:
        if line.startswith("LOCO US->India"):
            loco_us_to_india = float(line.split("macro_f05=")[1].split()[0])
        elif line.startswith("LOCO India->US"):
            loco_india_to_us = float(line.split("macro_f05=")[1].split()[0])

    with stage("predict test candidates + write submission"):
        train_categories = {c: df[c].cat.categories for c in CATEGORICAL_FEATURES}
        work = predict_test_probs_chunked(models, train_categories, FEATURES_DIR / "test_candidates_features.parquet")
        print(f"  {len(work)} total test candidate pairs scored")

        if best_variant in ("exclusive", "relative"):
            excl_work = _apply_exclusivity(work)
        if best_variant == "relative":
            best_p = _best_p_per_s1(work, work["p"].values)
            final_work = excl_work.join(best_p.rename("s1_best_p"), on="s1_id")
            final_work = final_work[final_work["p"] >= best_alpha * final_work["s1_best_p"]]
        elif best_variant == "exclusive":
            final_work = excl_work
        else:
            final_work = work
        pred_map = _pred_map_from_work(final_work, best_t)

        all_test_s1_ids = pd.read_parquet(raw_parquet_path("test", "S1"), columns=["id"])["id"].tolist()
        write_submission(pred_map, all_test_s1_ids)

        cand_map = {}
        for row_s1, row_match, row_source in zip(work["s1_id"], work["match_id"], work["other_source"]):
            cand_map.setdefault(row_s1, []).append(f"{row_source}-{row_match}")
        str_cand_map = {f"S1-{i}": cand_map.get(i, []) for i in all_test_s1_ids}
        write_id_list_tsv(CANDIDATE_PAIRS_PATH, "candidate_entity_ids", str_cand_map)
        print(f"Wrote {CANDIDATE_PAIRS_PATH}")

    with stage("validate submission"):
        result = subprocess.run(
            [sys.executable, "utils/validate_submission.py",
             "--matching", str(MATCHING_RESULTS_PATH), "--candidate", str(CANDIDATE_PAIRS_PATH),
             "--test-dir", "dataset/test"],
            cwd=ROOT_DIR, capture_output=True, text=True,
        )
        print(result.stdout)
        print(result.stderr)

    append_results_log(
        description=f"LightGBM 5-fold, decision={best_variant} t={best_t} alpha={best_alpha}",
        cv_overall=best_row["macro_f05"], cv_us=cv_us, cv_india=cv_india,
        loco_us_to_india=loco_us_to_india, loco_india_to_us=loco_india_to_us,
        notes="blocking recall 90.75% (below 97% target), see BLOCKING_REPORT.md",
    )
    print(f"\nAppended results row to {RESULTS_LOG_PATH}")


if __name__ == "__main__":
    main()
