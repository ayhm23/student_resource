"""Track 5 v3: two-stage LightGBM, calibrated probabilities, and a decision
layer tuned on out-of-fold predictions -- then the test submission.

Pipeline (see ``main()``):
  1. Training S1 set: a seeded random subset of ALL train S1 (every one, if
     the machine's RAM allows -- ``resources.train_max_rows``), including S1
     records blocking found no candidate for, so blocking misses count.
  2. Stage 1: 5-fold LightGBM (pre-assigned folds) on the pair features ->
     out-of-fold probability p1 for training rows; the fold-averaged model
     scores every other train candidate and every test candidate.
  3. Stage-2 features (DuckDB windows over p1 for the WHOLE candidate
     population of each split): how this pair's p1 compares to the S1's other
     candidates and to the other S1 records competing for the same S2/S3
     record. The data plants several businesses at one address (~20% of
     exact-address pairs are not matches); only a model that sees the
     competitors' scores can tell which one a record belongs to.
  4. Stage 2: 5-fold LightGBM on pair features + stage-2 features -> OOF p2;
     cross-fitted isotonic calibration of p2.
  5. Decision layer, tuned on OOF against every training S1's TRUE match set:
     global threshold / exclusivity / relative margin (the old rules) and a
     per-S1 expected-F0.5 optimum (the subset -- possibly empty -- with the
     highest expected F0.5 under the calibrated probabilities, computed
     exactly with Poisson-binomial distributions). Best OOF macro F0.5 wins.
  6. LOCO (US<->India) stage-1 generalization check, test prediction, both
     output files, validation, MODEL_REPORT.md and a results_log row.

Memory: feature matrices are float32 numpy (categoricals as fixed integer
codes; S2/S3 as a 0/1 code, never Python strings) filled batch by batch from
parquet via DuckDB; LightGBM trains on per-fold ``Dataset.subset`` views of one
binned dataset; bulk predictions stream to parquet. Country is never a feature.
"""

import json
import subprocess
import sys
import time
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from config import (CANDIDATE_PAIRS_PATH, DATA_DIR, FEATURES_DIR, FOLDS_PARQUET,
                    MATCHING_RESULTS_PATH, OUTPUT_DIR, RESULTS_LOG_PATH, ROOT_DIR,
                    raw_parquet_path)
from db import connect
from evaluate import gt_pairs
from perf import print_sysinfo, progress_line, stage
from resources import profile

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SEED = 42
NUMERIC_FEATURES = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_jaro_winkler", "name_nospace_ratio", "name_nospace_contains", "name_sorted_chars_ratio",
    "name_idf_jaccard", "name_idf_rarest_shared", "name_idf_rarest_unshared",
    "name_skeleton_ratio", "name_skeleton_token_set_ratio", "name_skeleton_jaccard",
    "name_contain_a_in_b", "name_contain_b_in_a", "name_token_substring_cov",
    "name_unshared_a", "name_unshared_b",
    "s1_name_len", "other_name_len", "name_len_absdiff",
    "addr_token_set_ratio", "addr_token_sort_ratio", "addr_num_jaccard", "addr_num_max_equal",
    "addr_num_shared_count", "addr_num_near_match", "addr_idf_jaccard", "other_addr_empty",
    "n_keys_hit", "cheap_score", "rank_in_s1", "n_candidates_for_s1",
    "n_competitors", "rank_among_competitors",
]
# Fixed integer codes, identical for train and test.
CATEGORICAL_CODES = {
    "legal_form_match": {"missing": 0, "same": 1, "different": 2},
    "addr_state_match": {"missing": 0, "same": 1, "different": 2},
    "other_source": {"S2": 0, "S3": 1},
}
CATEGORICAL_FEATURES = list(CATEGORICAL_CODES)
STAGE1_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
STAGE2_EXTRA = [
    "p1", "p1_s1_max", "p1_gap_s1", "p1_rank_s1", "p1_s1_sum", "p1_s1_n_above_half",
    "p1_comp_other_max", "p1_gap_comp", "p1_rank_comp", "p1_comp_n_above_half",
]
STAGE2_FEATURES = STAGE1_FEATURES + STAGE2_EXTRA
SRC_SQL = "CASE f.other_source WHEN 'S3' THEN 1 ELSE 0 END"

LGB_PARAMS = dict(
    objective="binary", metric="binary_logloss", verbosity=-1,
    num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=1, min_data_in_leaf=100, lambda_l2=1.0, max_bin=255,
)
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 100
LOCO_ROUNDS = 400
PREDICT_CHUNK = 1_000_000
CASCADE_P = 0.005

T_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
ALPHA_GRID = np.round(np.arange(0.40, 1.00, 0.05), 2)
MISS_GRID = (0.0, 0.1, 0.3, 0.6)
EXPF_TOP_M = 8
BETA2 = 0.25
# Rows below this probability can never be picked by any rule in the grid
# (thresholds start at 0.05; the expected-F optimum never adds such a row), so
# they are dropped before the decision layer -- identically for OOF and test.
CANDIDATE_FLOOR = 0.01

TRAINING_S1_PARQUET = DATA_DIR / "training_s1.parquet"
TRAIN_FEATS = FEATURES_DIR / "train_candidates_features.parquet"
TEST_FEATS = FEATURES_DIR / "test_candidates_features.parquet"
OOF_PARQUET = FEATURES_DIR / "oof_stage2.parquet"
MODEL_REPORT = OUTPUT_DIR / "MODEL_REPORT.md"

_DEVICE_PARAMS_CACHE = {}


def detect_lgb_device():
    """Probe (once) whether LightGBM can train on GPU here; fall back to CPU and say why."""
    if "device_params" in _DEVICE_PARAMS_CACHE:
        return _DEVICE_PARAMS_CACHE["device_params"]
    probe_X = np.random.rand(200, 1)
    probe_y = (probe_X[:, 0] > 0.5).astype(int)
    try:
        lgb.train({"objective": "binary", "verbosity": -1, "device_type": "gpu"},
                  lgb.Dataset(probe_X, probe_y), num_boost_round=2)
        device_params = {"device_type": "gpu"}
        print("[train_model] GPU detected and usable -- training LightGBM on GPU.")
    except Exception as exc:
        device_params = {"device_type": "cpu"}
        print(f"[train_model] GPU not usable ({type(exc).__name__}: {str(exc)[:120]}) -- using CPU "
              f"with {profile().lgb_threads} threads.")
    _DEVICE_PARAMS_CACHE["device_params"] = device_params
    return device_params


def lgb_params(n_rows):
    # Larger training sets converge in fewer, bigger steps; keeps a full-2.2M-S1
    # run (tens of millions of rows) within hours instead of a day.
    lr = 0.05 if n_rows < 20_000_000 else 0.1
    return dict(LGB_PARAMS, learning_rate=lr, seed=SEED, num_threads=profile().lgb_threads,
                **detect_lgb_device())


def pair_codes(match_id, src):
    """Integer key per S2/S3 record: match_id*2 + src (0=S2, 1=S3) -- same as evaluate.pair_key."""
    return np.asarray(match_id, dtype=np.int64) * 2 + np.asarray(src, dtype=np.int64)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _feature_sql(col):
    """SQL expression producing one feature as FLOAT (categoricals -> fixed integer codes)."""
    if col in CATEGORICAL_CODES:
        cases = " ".join(f"WHEN '{k}' THEN {v}" for k, v in CATEGORICAL_CODES[col].items())
        return f"CAST(CASE f.{col} {cases} ELSE NULL END AS FLOAT) AS {col}"
    if col in STAGE2_EXTRA:
        return f"CAST(s.{col} AS FLOAT) AS {col}"
    return f"CAST(f.{col} AS FLOAT) AS {col}"


def _from_sql(split, with_stage2):
    feats = (TRAIN_FEATS if split == "train" else TEST_FEATS).as_posix()
    sql = f"FROM '{feats}' f"
    if with_stage2:
        sql += (f" JOIN {split}_s2feat s ON s.s1_id = f.s1_id AND s.match_id = f.match_id "
                f"AND s.src = {SRC_SQL}")
    return sql


def select_training_s1(con):
    """Seeded random subset of all train S1 whose candidate rows fit ``train_max_rows``."""
    max_rows = profile().train_max_rows
    counts = con.execute(f"SELECT s1_id, count(*) AS n FROM '{TRAIN_FEATS.as_posix()}' GROUP BY s1_id").df()
    s1 = pd.read_parquet(raw_parquet_path("train", "S1"), columns=["id", "country"]).rename(columns={"id": "s1_id"})
    folds = pd.read_parquet(FOLDS_PARQUET).rename(columns={"id": "s1_id"})
    s1 = s1.merge(folds, on="s1_id").merge(counts, on="s1_id", how="left")
    s1["n"] = s1["n"].fillna(0).astype(np.int64)
    s1 = s1.sort_values("s1_id").reset_index(drop=True)
    s1 = s1.iloc[np.random.default_rng(SEED).permutation(len(s1))].reset_index(drop=True)
    keep = s1["n"].cumsum() <= max_rows
    chosen = s1[keep][["s1_id", "country", "fold"]].reset_index(drop=True)
    chosen.to_parquet(TRAINING_S1_PARQUET, index=False)
    print(f"[train_model] training S1 set: {len(chosen)}/{len(s1)} train S1 "
          f"({int(s1['n'][keep].sum())} candidate rows; row budget {max_rows} from the machine profile)")
    return chosen


def load_training_matrix(con, features, with_stage2):
    """(keys DataFrame[s1_id, match_id, src], X float32, y int8, fold int8) for training-S1 rows."""
    base = (_from_sql("train", with_stage2)
            + f" JOIN '{TRAINING_S1_PARQUET.as_posix()}' t ON t.s1_id = f.s1_id")
    n = con.execute(f"SELECT count(*) {base}").fetchone()[0]
    select = (["f.s1_id", "f.match_id", f"{SRC_SQL} AS src", "f.label", "t.fold"]
              + [_feature_sql(c) for c in features])
    result = con.execute(f"SELECT {', '.join(select)} {base}")

    X = np.empty((n, len(features)), dtype=np.float32)
    s1_id = np.empty(n, dtype=np.int64)
    match_id = np.empty(n, dtype=np.int64)
    src = np.empty(n, dtype=np.int8)
    y = np.empty(n, dtype=np.int8)
    fold = np.empty(n, dtype=np.int8)
    pos, t0 = 0, time.perf_counter()
    for batch in result.to_arrow_reader(1_000_000):
        m = batch.num_rows
        s1_id[pos:pos + m] = batch.column("s1_id").to_numpy()
        match_id[pos:pos + m] = batch.column("match_id").to_numpy()
        src[pos:pos + m] = batch.column("src").to_numpy()
        y[pos:pos + m] = batch.column("label").to_numpy()
        fold[pos:pos + m] = batch.column("fold").to_numpy()
        for j, c in enumerate(features):
            X[pos:pos + m, j] = batch.column(c).to_numpy(zero_copy_only=False)
        pos += m
        print(f"  {progress_line('load training matrix', pos, n, t0)}")
    keys = pd.DataFrame({"s1_id": s1_id, "match_id": match_id, "src": src})
    return keys, X, y, fold


def iter_split_batches(con, features, split, with_stage2, batch_rows):
    """Yield (keys DataFrame, X float32) over ALL candidate rows of a split."""
    select = ["f.s1_id", "f.match_id", f"{SRC_SQL} AS src"] + [_feature_sql(c) for c in features]
    result = con.execute(f"SELECT {', '.join(select)} {_from_sql(split, with_stage2)}")
    for batch in result.to_arrow_reader(batch_rows):
        X = np.empty((batch.num_rows, len(features)), dtype=np.float32)
        for j, c in enumerate(features):
            X[:, j] = batch.column(c).to_numpy(zero_copy_only=False)
        keys = pd.DataFrame({"s1_id": batch.column("s1_id").to_numpy(),
                             "match_id": batch.column("match_id").to_numpy(),
                             "src": batch.column("src").to_numpy().astype(np.int8)})
        yield keys, X


# ---------------------------------------------------------------------------
# Training / prediction
# ---------------------------------------------------------------------------

def predict_chunked(model, X, idx=None):
    """model.predict over X[idx] in chunks (never copies all of X[idx] at once)."""
    idx = np.arange(len(X)) if idx is None else idx
    out = np.empty(len(idx), dtype=np.float64)
    for lo in range(0, len(idx), PREDICT_CHUNK):
        part = idx[lo:lo + PREDICT_CHUNK]
        out[lo:lo + len(part)] = model.predict(X[part], num_iteration=model.best_iteration,
                                               num_threads=profile().lgb_threads)
    return out


def predict_avg(models, X):
    """Fold-averaged probability, as a cascade: one model scores every row, and only rows it
    gives >= CASCADE_P are re-scored with the full 5-model average.

    Averaging 5 models x ~1.5k trees over 100M+ candidate rows took hours; almost all rows
    are near-certain non-matches that no decision rule can ever select (every rule floors at
    CANDIDATE_FLOOR > CASCADE_P), so their exact averaged value does not matter.
    """
    threads = profile().cores
    p = models[0].predict(X, num_iteration=models[0].best_iteration, num_threads=threads)
    hot = np.flatnonzero(p >= CASCADE_P)
    if len(hot) and len(models) > 1:
        Xh = X[hot]
        p[hot] = np.mean([m.predict(Xh, num_iteration=m.best_iteration, num_threads=threads)
                          for m in models], axis=0)
    return p


def build_dataset(X, y, features):
    cat_idx = [features.index(c) for c in CATEGORICAL_FEATURES]
    return lgb.Dataset(X, label=y, feature_name=features, categorical_feature=cat_idx,
                       params={"max_bin": LGB_PARAMS["max_bin"], "verbosity": -1},
                       free_raw_data=False).construct()


def train_cv(full, X, y, fold, features, label):
    """5-fold LightGBM on subsets of one binned dataset. Returns (oof, models, mean gain importance)."""
    params = lgb_params(len(y))
    oof = np.full(len(y), np.nan, dtype=np.float64)
    models, importances = [], []
    folds = sorted(np.unique(fold))
    t0 = time.perf_counter()
    for i, f in enumerate(folds):
        va_idx = np.flatnonzero(fold == f)
        tr_idx = np.flatnonzero(fold != f)
        model = lgb.train(params, full.subset(tr_idx), num_boost_round=NUM_BOOST_ROUND,
                          valid_sets=[full.subset(va_idx)],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False),
                                     lgb.log_evaluation(period=0)])
        oof[va_idx] = predict_chunked(model, X, va_idx)
        models.append(model)
        importances.append(pd.Series(model.feature_importance(importance_type="gain"), index=features))
        print(f"  [{label}] fold {i + 1}/{len(folds)}: best_iter={model.best_iteration} -- "
              f"{progress_line('CV', i + 1, len(folds), t0)}")
    return oof, models, pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)


def save_models(models, name):
    """Persist fold models (LightGBM text format) to data/models/ for reuse and inspection."""
    out = DATA_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(models):
        m.save_model(str(out / f"{name}_fold{i}.txt"), num_iteration=m.best_iteration)


def write_split_predictions(con, models, features, split, with_stage2, table):
    """Fold-averaged predictions for every candidate row of a split, streamed to parquet -> DuckDB table."""
    feats = (TRAIN_FEATS if split == "train" else TEST_FEATS).as_posix()
    n = con.execute(f"SELECT count(*) FROM '{feats}'").fetchone()[0]
    out_path = FEATURES_DIR / f"{table}.parquet"
    writer, done, t0 = None, 0, time.perf_counter()
    for keys, X in iter_split_batches(con, features, split, with_stage2, profile().predict_batch_rows):
        keys["p"] = predict_avg(models, X)
        t = pa.Table.from_pandas(keys, preserve_index=False)
        writer = writer or pq.ParquetWriter(str(out_path), t.schema)
        writer.write_table(t)
        done += len(keys)
        print(f"  {progress_line(f'predict {split} -> {table}', done, n, t0)}")
    writer.close()
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM '{out_path.as_posix()}'")


def build_stage2_features(con, split, p1_table):
    """Stage-2 features from stage-1 probabilities over the split's full candidate population."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {split}_s2feat AS
        SELECT s1_id, match_id, src, p1,
               p1_s1_max, p1 - p1_s1_max AS p1_gap_s1, p1_rank_s1, p1_s1_sum, p1_s1_n_above_half,
               p1_comp_other_max, p1 - p1_comp_other_max AS p1_gap_comp, p1_rank_comp,
               p1_comp_n_above_half
        FROM (
            SELECT s1_id, match_id, src, p AS p1,
                   max(p) OVER (PARTITION BY s1_id) AS p1_s1_max,
                   row_number() OVER (PARTITION BY s1_id ORDER BY p DESC, match_id, src) AS p1_rank_s1,
                   sum(p) OVER (PARTITION BY s1_id) AS p1_s1_sum,
                   sum((p >= 0.5)::INTEGER) OVER (PARTITION BY s1_id) AS p1_s1_n_above_half,
                   CASE WHEN row_number() OVER w_desc = 1
                        THEN coalesce(nth_value(p, 2) OVER w_all, 0.0)
                        ELSE max(p) OVER (PARTITION BY match_id, src) END AS p1_comp_other_max,
                   row_number() OVER w_desc AS p1_rank_comp,
                   sum((p >= 0.5)::INTEGER) OVER (PARTITION BY match_id, src)
                       - (p >= 0.5)::INTEGER AS p1_comp_n_above_half
            FROM {p1_table}
            WINDOW w_desc AS (PARTITION BY match_id, src ORDER BY p DESC, s1_id),
                   w_all AS (PARTITION BY match_id, src ORDER BY p DESC, s1_id
                             ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
        )
    """)


def cross_fit_isotonic(p, y, fold):
    """Out-of-fold isotonic calibration of OOF probabilities (fit on 4 folds, apply to the 5th)."""
    out = np.empty_like(p)
    for f in np.unique(fold):
        va = fold == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p[~va], y[~va])
        out[va] = iso.predict(p[va])
    return out


# ---------------------------------------------------------------------------
# Decision layer
# ---------------------------------------------------------------------------

class Scorer:
    """Vectorized macro F0.5 over a fixed S1 population (same conventions as evaluate.score)."""

    def __init__(self, s1_df, gt):
        self.s1 = s1_df["s1_id"].to_numpy()
        self.country = s1_df["country"].to_numpy()
        self.pos = pd.Series(np.arange(len(self.s1)), index=self.s1)
        gpos = self.pos.reindex(gt["s1_id"].to_numpy()).to_numpy()
        ok = ~np.isnan(gpos)
        gpos = gpos[ok].astype(np.int64)
        self.n_true = np.bincount(gpos, minlength=len(self.s1)).astype(np.float64)
        self.gt_codes = np.sort(gpos * (1 << 26) + gt["key"].to_numpy()[ok])

    def __call__(self, s1_id, key):
        ppos = self.pos.reindex(s1_id).to_numpy().astype(np.int64)
        codes = ppos * (1 << 26) + key
        n_pred = np.bincount(ppos, minlength=len(self.s1)).astype(np.float64)
        idx = np.searchsorted(self.gt_codes, codes)
        hit = (idx < len(self.gt_codes)) & (self.gt_codes[np.minimum(idx, len(self.gt_codes) - 1)] == codes)
        tp = np.bincount(ppos[hit], minlength=len(self.s1)).astype(np.float64)
        single = self.n_true == 0
        denom = np.maximum(0.25 * self.n_true + n_pred, 1e-12)
        f = np.where(single, (n_pred == 0).astype(np.float64), np.where(tp > 0, 1.25 * tp / denom, 0.0))
        prec = np.where(single, (n_pred == 0).astype(np.float64), np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0))
        rec = np.where(single, (n_pred == 0).astype(np.float64), tp / np.maximum(self.n_true, 1))
        by_c = pd.Series(f).groupby(self.country).mean().to_dict()
        return {"macro_f05": float(f.mean()), "f05_by_country": {k: float(v) for k, v in by_c.items()},
                "singleton_accuracy": float((n_pred[single] == 0).mean()) if single.any() else float("nan"),
                "mean_precision": float(prec.mean()), "mean_recall": float(rec.mean())}


def exclusive_mask(match_key, p):
    """True for the single highest-probability row of each S2/S3 record."""
    order = np.lexsort((-p, match_key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = match_key[order][1:] != match_key[order][:-1]
    mask = np.zeros(len(p), dtype=bool)
    mask[order[first]] = True
    return mask


def s1_best(s1_id, p):
    return pd.Series(p).groupby(s1_id).transform("max").to_numpy()


def _best_k(P, miss):
    """Per-row k in 0..M maximizing expected F0.5 of predicting the top-k (exact Poisson-binomial).

    P: (G, M) probabilities sorted descending (0-padded); miss: (G,) probability
    of a true match outside the scored list (blocking miss / past the top M),
    modelled as one extra Bernoulli.
    """
    G, M = P.shape
    suf = np.empty((M + 1, G, M + 2))
    base = np.zeros((G, M + 2))
    base[:, 0], base[:, 1] = 1.0 - miss, miss
    suf[M] = base
    for k in range(M - 1, -1, -1):
        q = P[:, k:k + 1]
        cur = suf[k + 1] * (1.0 - q)
        cur[:, 1:] += suf[k + 1][:, :-1] * q
        suf[k] = cur
    pre = np.zeros((G, M + 1))
    pre[:, 0] = 1.0
    a = np.arange(M + 1, dtype=float)[:, None]
    b = np.arange(M + 2, dtype=float)[None, :]
    best_val = np.full(G, -1.0)
    best_k = np.zeros(G, dtype=np.int64)
    for k in range(M + 1):
        if k > 0:
            q = P[:, k - 1:k]
            new = pre * (1.0 - q)
            new[:, 1:] += pre[:, :-1] * q
            pre = new
            F = np.where((a >= 1) & (a <= k), (1 + BETA2) * a / (BETA2 * (a + b) + k), 0.0)
        else:
            F = np.zeros((M + 1, M + 2))
            F[0, 0] = 1.0
        ef = np.einsum("gb,gb->g", pre @ F, suf[k])
        better = ef > best_val + 1e-12
        best_val = np.where(better, ef, best_val)
        best_k = np.where(better, k, best_k)
    return best_k


def expected_f_select(s1_id, p, miss_lambda, chunk=200_000):
    """Boolean mask of rows chosen by the per-S1 expected-F0.5 optimum."""
    if len(p) == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((-p, s1_id))
    s_sorted, p_sorted = s1_id[order], p[order]
    new_group = np.r_[True, s_sorted[1:] != s_sorted[:-1]]
    starts = np.flatnonzero(new_group)
    group = np.cumsum(new_group) - 1
    rank = np.arange(len(order)) - starts[group]
    G = len(starts)
    P = np.zeros((G, EXPF_TOP_M))
    top = rank < EXPF_TOP_M
    P[group[top], rank[top]] = p_sorted[top]
    rest = np.bincount(group[~top], weights=p_sorted[~top], minlength=G)
    miss = np.clip(miss_lambda + rest, 0.0, 1.0)
    k_star = np.empty(G, dtype=np.int64)
    for lo in range(0, G, chunk):
        k_star[lo:lo + chunk] = _best_k(P[lo:lo + chunk], miss[lo:lo + chunk])
    mask = np.zeros(len(p), dtype=bool)
    mask[order[rank < k_star[group]]] = True
    return mask


def apply_rule(rule, s1_id, match_key, p):
    """Row mask of predicted matches for one decision rule."""
    keep = p >= CANDIDATE_FLOOR
    if rule["exclusive"]:
        keep &= exclusive_mask(match_key, p)
    if rule["kind"] == "threshold":
        mask = keep & (p >= rule["t"])
        if rule.get("alpha") is not None:
            mask &= p >= rule["alpha"] * s1_best(s1_id, np.where(keep, p, 0.0))
        return mask
    idx = np.flatnonzero(keep)
    mask = np.zeros(len(p), dtype=bool)
    mask[idx[expected_f_select(s1_id[idx], p[idx], rule["miss"])]] = True
    return mask


def tune_decision_rule(s1_id, match_key, probs, scorer):
    """Evaluate every rule candidate on OOF (``probs``: {"raw": p, "cal": p}); return (results, best rule)."""
    results = []
    for name, p in probs.items():
        floor = p >= CANDIDATE_FLOOR
        excl = floor & exclusive_mask(match_key, p)
        best = s1_best(s1_id, np.where(excl, p, 0.0))
        for exclusive, base in ((False, floor), (True, excl)):
            for t in T_GRID:
                m = base & (p >= t)
                results.append({"kind": "threshold", "probs": name, "exclusive": exclusive, "t": float(t),
                                "alpha": None, "miss": None, **scorer(s1_id[m], match_key[m])})
        for alpha in ALPHA_GRID:
            rel = excl & (p >= alpha * best)
            for t in T_GRID:
                m = rel & (p >= t)
                results.append({"kind": "threshold", "probs": name, "exclusive": True, "t": float(t),
                                "alpha": float(alpha), "miss": None, **scorer(s1_id[m], match_key[m])})
        for exclusive, base in ((False, floor), (True, excl)):
            idx = np.flatnonzero(base)
            for miss in MISS_GRID:
                m = idx[expected_f_select(s1_id[idx], p[idx], miss)]
                results.append({"kind": "expected_f", "probs": name, "exclusive": exclusive, "t": None,
                                "alpha": None, "miss": miss, **scorer(s1_id[m], match_key[m])})
        print(f"  evaluated {len(results)} rule candidates so far (probs={name})")
    df = pd.DataFrame(results)
    best_row = df.loc[df["macro_f05"].idxmax()]
    rule = {k: best_row[k] for k in ("kind", "probs", "exclusive", "t", "alpha", "miss")}
    rule = {k: (None if (v is None or (isinstance(v, float) and np.isnan(v))) else
                (bool(v) if k == "exclusive" else v)) for k, v in rule.items()}
    return df, rule


def rule_str(rule):
    if rule["kind"] == "expected_f":
        return f"expected-F0.5 optimum (probs={rule['probs']}, exclusive={rule['exclusive']}, miss={rule['miss']})"
    alpha = rule.get("alpha")
    alpha_ok = alpha is not None and not (isinstance(alpha, float) and np.isnan(alpha))
    return (f"threshold t={rule['t']} (probs={rule['probs']}, exclusive={rule['exclusive']}"
            + (f", alpha={alpha})" if alpha_ok else ")"))


# ---------------------------------------------------------------------------
# LOCO and outputs
# ---------------------------------------------------------------------------

def run_loco(full, X, keys, y, training_s1, gt):
    """Train stage 1 on one country, score the other (exclusive, t=0.5): a generalization check."""
    lines = []
    country_of = dict(zip(training_s1["s1_id"], training_s1["country"]))
    country = keys["s1_id"].map(country_of).to_numpy()
    mkey = pair_codes(keys["match_id"], keys["src"])
    for tr_c, te_c in (("US", "India"), ("India", "US")):
        tr = np.flatnonzero(country == tr_c)
        te = np.flatnonzero(country == te_c)
        if not len(tr) or not len(te):
            lines.append(f"LOCO {tr_c}->{te_c}: SKIPPED (no rows)")
            continue
        model = lgb.train(lgb_params(len(tr)), full.subset(tr), num_boost_round=LOCO_ROUNDS)
        model.best_iteration = 0
        p = predict_chunked(model, X, te)
        s1_te = keys["s1_id"].to_numpy()[te]
        mask = exclusive_mask(mkey[te], p) & (p >= 0.5)
        pop = training_s1[training_s1["country"] == te_c]
        m = Scorer(pop, gt)(s1_te[mask], mkey[te][mask])
        lines.append(f"LOCO {tr_c}->{te_c}: macro_f05={m['macro_f05']:.4f} precision={m['mean_precision']:.4f} "
                     f"recall={m['mean_recall']:.4f} singleton_acc={m['singleton_accuracy']:.4f}")
        print(lines[-1])
    return lines


def write_id_tsv(con, pairs_table, path, header_col, order_col):
    """One row per test S1 (all of them) with ids aggregated from a (s1_id, match_id, src) table."""
    s1_path = raw_parquet_path("test", "S1").as_posix()
    result = con.execute(f"""
        SELECT s.id, coalesce(string_agg(
                   CASE c.src WHEN 1 THEN 'S3-' ELSE 'S2-' END || CAST(c.match_id AS VARCHAR), ','
                   ORDER BY c.{order_col}), '') AS ids
        FROM '{s1_path}' s LEFT JOIN {pairs_table} c ON c.s1_id = s.id
        GROUP BY s.id ORDER BY s.id
    """)
    n = 0
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(f"source1_entity_id\t{header_col}\n")
        for batch in result.to_arrow_reader(500_000):
            ids = batch.column("id").to_numpy()
            lists = batch.column("ids").to_pylist()
            fh.write("".join(f"S1-{i}\t{l}\n" for i, l in zip(ids, lists)))
            n += len(ids)
    print(f"Wrote {path} ({n} rows)")


def append_results_log(description, best_row, loco_lines, notes):
    loco = {}
    for line in loco_lines:
        if "macro_f05=" in line:
            loco[line.split(":")[0]] = float(line.split("macro_f05=")[1].split()[0])
    by_c = best_row["f05_by_country"]
    row = pd.DataFrame([{
        "date": date.today().isoformat(), "git_commit": _git_commit(), "description": description,
        "cv_overall": round(best_row["macro_f05"], 4),
        "cv_us": round(by_c.get("US", float("nan")), 4), "cv_india": round(by_c.get("India", float("nan")), 4),
        "loco_us_to_india": loco.get("LOCO US->India", ""), "loco_india_to_us": loco.get("LOCO India->US", ""),
        "public_lb": "", "notes": notes,
    }])
    row.to_csv(RESULTS_LOG_PATH, mode="a", header=not RESULTS_LOG_PATH.exists(), index=False)


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def main():
    """Train both stages, tune the decision rule, predict test, write + validate the submission."""
    print_sysinfo()
    prof = profile()
    con = connect(role="light")

    with stage("select training S1 set"):
        training_s1 = select_training_s1(con)
    gt = gt_pairs(training_s1["s1_id"])
    scorer = Scorer(training_s1, gt)

    # ---- stage 1 ----
    with stage("load stage-1 training matrix"):
        keys, X, y, fold = load_training_matrix(con, STAGE1_FEATURES, with_stage2=False)
    print(f"  {len(y)} training rows, {int(y.sum())} positive, {X.nbytes / 2**30:.2f} GB float32")
    with stage("stage 1: 5-fold LightGBM"):
        full = build_dataset(X, y, STAGE1_FEATURES)
        p1_oof, models1, imp1 = train_cv(full, X, y, fold, STAGE1_FEATURES, "stage1")
        save_models(models1, "stage1")
    with stage("LOCO check (stage 1)"):
        loco_lines = run_loco(full, X, keys, y, training_s1, gt)
    del full, X

    with stage("stage 1: score all train + test candidates"):
        con.execute(f"SET memory_limit='{prof.duckdb_heavy_gb}GB'")
        con.execute(f"SET threads={prof.duckdb_threads_heavy}")
        write_split_predictions(con, models1, STAGE1_FEATURES, "train", False, "train_p1_all")
        con.register("_oof1", keys.assign(p=p1_oof))
        con.execute("""
            CREATE OR REPLACE TABLE train_p1 AS
            SELECT a.s1_id, a.match_id, a.src, coalesce(o.p, a.p) AS p
            FROM train_p1_all a LEFT JOIN _oof1 o
              ON o.s1_id = a.s1_id AND o.match_id = a.match_id AND o.src = a.src
        """)
        con.unregister("_oof1")
        con.execute("DROP TABLE train_p1_all")
        write_split_predictions(con, models1, STAGE1_FEATURES, "test", False, "test_p1")
    with stage("stage-2 features (full train + test populations)"):
        build_stage2_features(con, "train", "train_p1")
        build_stage2_features(con, "test", "test_p1")
        con.execute(f"SET memory_limit='{prof.duckdb_light_gb}GB'")
        con.execute(f"SET threads={prof.duckdb_threads_light}")
    del keys

    # ---- stage 2 ----
    with stage("load stage-2 training matrix"):
        keys, X, y, fold = load_training_matrix(con, STAGE2_FEATURES, with_stage2=True)
    with stage("stage 2: 5-fold LightGBM"):
        full = build_dataset(X, y, STAGE2_FEATURES)
        p2_oof, models2, imp2 = train_cv(full, X, y, fold, STAGE2_FEATURES, "stage2")
        save_models(models2, "stage2")
    del full, X
    p2_cal = cross_fit_isotonic(p2_oof, y, fold)
    iso_full = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p2_oof, y)

    s1_id = keys["s1_id"].to_numpy()
    mkey = pair_codes(keys["match_id"], keys["src"])
    with stage("tune decision rule on OOF (every training S1, blocking misses included)"):
        results, rule = tune_decision_rule(s1_id, mkey, {"raw": p2_oof, "cal": p2_cal}, scorer)
        stage1_ref = tune_decision_rule(s1_id, mkey, {"raw": p1_oof_aligned(keys, con)}, scorer)[0]
    best_row = results.loc[results["macro_f05"].idxmax()]
    s1_best_row = stage1_ref.loc[stage1_ref["macro_f05"].idxmax()]
    print(f"\nBEST RULE: {rule_str(rule)} -> OOF macro F0.5 {best_row['macro_f05']:.4f} "
          f"{best_row['f05_by_country']}")
    print(f"  (stage 1 alone, best rule: {s1_best_row['macro_f05']:.4f} -- stage 2 adds "
          f"{best_row['macro_f05'] - s1_best_row['macro_f05']:+.4f})")
    top = results.sort_values("macro_f05", ascending=False)
    for kind in ("threshold", "expected_f"):
        r = top[top["kind"] == kind].iloc[0]
        print(f"  best {kind}: {rule_str(r.to_dict())} -> {r['macro_f05']:.4f}")

    p_rule = p2_cal if rule["probs"] == "cal" else p2_oof
    keys.assign(label=y, fold=fold, p2=p2_oof, p2_cal=p2_cal,
                pred=apply_rule(rule, s1_id, mkey, p_rule)).to_parquet(OOF_PARQUET, index=False)
    (DATA_DIR / "decision_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")

    # ---- test ----
    with stage("stage 2: score test candidates + apply the decision rule"):
        n_test = con.execute(f"SELECT count(*) FROM '{TEST_FEATS.as_posix()}'").fetchone()[0]
        parts, done, t0 = [], 0, time.perf_counter()
        for tkeys, Xt in iter_split_batches(con, STAGE2_FEATURES, "test", True, prof.predict_batch_rows):
            p2 = predict_avg(models2, Xt)
            tkeys["p"] = iso_full.predict(p2) if rule["probs"] == "cal" else p2
            parts.append(tkeys[tkeys["p"] >= CANDIDATE_FLOOR])
            done += len(tkeys)
            print(f"  {progress_line('predict test (stage 2)', done, n_test, t0)}")
        test = pd.concat(parts, ignore_index=True)
        tmask = apply_rule(rule, test["s1_id"].to_numpy(), pair_codes(test["match_id"], test["src"]),
                           test["p"].to_numpy())
        preds = test[tmask].reset_index(drop=True)
        print(f"  {len(preds)} predicted matches for {preds['s1_id'].nunique()} test S1 "
              f"({len(test)} of {n_test} candidate pairs above the {CANDIDATE_FLOOR} floor)")

    with stage("write + validate submission"):
        con.register("_preds", preds)
        con.execute("CREATE OR REPLACE TABLE test_predictions AS SELECT * FROM _preds")
        con.unregister("_preds")
        con.execute("""CREATE OR REPLACE VIEW v_test_scored_src AS
                       SELECT s1_id, match_id, CASE match_source WHEN 'S3' THEN 1 ELSE 0 END AS src, rank_in_s1
                       FROM test_scored""")
        write_id_tsv(con, "test_predictions", MATCHING_RESULTS_PATH, "matched_entity_ids", "p DESC")
        write_id_tsv(con, "v_test_scored_src", CANDIDATE_PAIRS_PATH, "candidate_entity_ids", "rank_in_s1")
        result = subprocess.run(
            [sys.executable, "utils/validate_submission.py", "--matching", str(MATCHING_RESULTS_PATH),
             "--candidate", str(CANDIDATE_PAIRS_PATH), "--test-dir", "dataset/test"],
            cwd=ROOT_DIR, capture_output=True, text=True)
        validation = (result.stdout + result.stderr).strip()
        print(validation)

    report = [
        f"# Model report ({date.today().isoformat()})", "",
        f"Machine profile: budget {prof.budget_gb} GB RAM, {prof.cores} cores; blocking caps "
        f"{prof.cap_default}/{prof.cap_strict}, top_k {prof.top_k}; training row budget {prof.train_max_rows}.", "",
        "## Result", "",
        f"- Training S1: {len(training_s1)} of 2.2M train S1; training candidate rows: {len(y)}",
        f"- Decision rule: {rule_str(rule)}",
        f"- OOF macro F0.5 (stage 2): **{best_row['macro_f05']:.4f}** by country "
        f"{json.dumps({k: round(v, 4) for k, v in best_row['f05_by_country'].items()})}; precision "
        f"{best_row['mean_precision']:.4f}, recall {best_row['mean_recall']:.4f}, singleton accuracy "
        f"{best_row['singleton_accuracy']:.4f}",
        f"- Stage 1 alone (best rule): {s1_best_row['macro_f05']:.4f}",
        "", "## Top decision rules (OOF)", "", "| rule | macro F0.5 | precision | recall |", "|---|---|---|---|",
    ]
    for _, r in top.head(15).iterrows():
        report.append(f"| {rule_str(r.to_dict())} | {r['macro_f05']:.4f} | {r['mean_precision']:.4f} | "
                      f"{r['mean_recall']:.4f} |")
    report += ["", "## LOCO (stage 1, exclusive t=0.5)", ""] + [f"- {line}" for line in loco_lines]
    report += ["", "## Stage-1 feature importance (gain, top 25)", "", "```", imp1.head(25).to_string(), "```",
               "", "## Stage-2 feature importance (gain, top 25)", "", "```", imp2.head(25).to_string(), "```",
               "", "## Submission validation", "", "```", validation, "```"]
    MODEL_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {MODEL_REPORT}")

    append_results_log(f"2-stage LightGBM, {rule_str(rule)}", best_row, loco_lines,
                       f"training S1 {len(training_s1)}, caps {prof.cap_default}/{prof.cap_strict}, "
                       f"top_k {prof.top_k}; see MODEL_REPORT.md / BLOCKING_REPORT.md")
    con.close()


def p1_oof_aligned(keys, con):
    """Stage-1 OOF probabilities for the stage-2 training rows (same row order as ``keys``)."""
    con.register("_k", keys[["s1_id", "match_id", "src"]].assign(_row=np.arange(len(keys))))
    p = con.execute("""
        SELECT t.p FROM _k k JOIN train_p1 t
          ON t.s1_id = k.s1_id AND t.match_id = k.match_id AND t.src = k.src
        ORDER BY k._row
    """).fetchnumpy()["p"]
    con.unregister("_k")
    return np.asarray(p, dtype=np.float64)


if __name__ == "__main__":
    main()
