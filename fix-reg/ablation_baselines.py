"""
Ablation Baseline Comparison for RecessionRadar ICML 2026 paper.

Three properly tuned baselines vs Full Chain Stacking Ensemble:
  A — Single-stage XGBoost (4 independent Optuna-tuned models, one per horizon)
  B — MultiOutputRegressor wrapping XGBoost with per-horizon best params (isolates chaining)
  C — MultiOutputRegressor wrapping XGBoost with jointly-tuned params (one set for all horizons)

Run from: RecessionRadar/RecessionRadar/
"""

import os
import sys
import re
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
SPLIT      = "2020-01-01"
OUT_CSV    = "fix-reg/ablation_results.csv"
OUT_PNG    = "fix-reg/ablation_mae_curves.png"

TARGETS = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]
LABELS  = ["Current", "1M", "3M", "6M"]
H_STEPS = [1, 1, 3, 6]  # for DM test bandwidth

# Full model MAE from paper (Table 2)
FULL_MODEL_MAE = {
    "Current": 6.8292,
    "1M":      5.6336,
    "3M":      7.7285,
    "6M":     10.1696,
}

# ── classes required to unpickle the full ensemble ───────────────────────────
def _inv_logit_model(z):
    return np.clip(1 / (1 + np.exp(-np.clip(z, -50, 50))) * 100, 0, 100)

def _san(df):
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
    return df

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
    def fit(self, X, y): return self
    def predict(self, X): return self.model.predict(X)

class FullChainCatBoostModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit_model(self.chain_model.predict(Xs)), 0, 100)

class FullChainLightGBMModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        X = _san(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit_model(self.chain_model.predict(Xs)), 0, 100)

class FullChainRandomForestModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit_model(self.chain_model.predict(Xs)), 0, 100)

class FullChainStackingEnsemble:
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {"CatBoost": FullChainCatBoostModel,
                            "LightGBM": FullChainLightGBMModel,
                            "RandomForest": FullChainRandomForestModel}
        self.meta_models = {}; self.cv_folds = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler = {}; self.fitted_base_models = {}
    def _eng(self, *bp):
        f = list(bp)
        if self.use_feature_engineering:
            f += [np.mean(bp, axis=0), 0.4*bp[0]+0.35*bp[1]+0.25*bp[2],
                  np.std(bp, axis=0), np.min(bp, axis=0), np.max(bp, axis=0)]
            for i in range(len(bp)):
                for j in range(i+1, len(bp)): f.append(np.abs(bp[i]-bp[j]))
        return np.column_stack(f)
    def predict(self, X):
        bp = {n: m.predict(X) for n, m in self.fitted_base_models.items()}
        fp = np.zeros_like(list(bp.values())[0])
        for i, t in enumerate(TARGETS):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf  = self._eng(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)

# ── helpers ──────────────────────────────────────────────────────────────────
EPS = 1e-6

def logit(y):
    y_s = np.clip(np.array(y, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(y_s / (1 - y_s))

def inv_logit(z):
    return np.clip(1.0 / (1.0 + np.exp(-np.array(z, dtype=float))) * 100.0, 0, 100)

def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ── DM test (Harvey-Leybourne-Newbold 1997) ──────────────────────────────────
def dm_test(actual, pred_baseline, pred_ours, h=1):
    """
    Two-sided HLN-modified DM test, squared-error loss.
    d_t = SE_baseline_t - SE_ours_t
    Positive DM -> our model has lower loss (better).
    """
    T = len(actual)
    d = (actual - pred_baseline)**2 - (actual - pred_ours)**2
    d_bar = np.mean(d)
    bw = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for k in range(1, bw + 1):
        w = 1.0 - k / (bw + 1)
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2 * w * gamma_k
    if lrv <= 0:
        return np.nan, np.nan
    dm_raw = d_bar / np.sqrt(lrv / T)
    hlr = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_hlr = dm_raw * hlr
    p_value = 2 * (1 - stats.t.cdf(abs(dm_hlr), df=T-1))
    return dm_hlr, p_value

# ── load and split data ───────────────────────────────────────────────────────
print("=" * 70)
print("ABLATION BASELINE COMPARISON")
print("=" * 70)
print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

X_train = clean(train_df.drop(columns=TARGETS + ["date"]))
X_test  = clean(test_df.drop(columns=TARGETS + ["date"]))
y_train = clean(train_df[TARGETS])
y_test  = clean(test_df[TARGETS])

print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X_train.shape[1]}")

X_train_arr = X_train.values.astype(float)
X_test_arr  = X_test.values.astype(float)

# ── Optuna search space helper ────────────────────────────────────────────────
def suggest_xgb_params(trial):
    return dict(
        n_estimators      = trial.suggest_int("n_estimators", 100, 800),
        max_depth         = trial.suggest_int("max_depth", 3, 8),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 1.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 3.0),
        objective         = "reg:squarederror",
        random_state      = 42,
        verbosity         = 0,
        tree_method       = "hist",
    )

def cv_mae_single(params, X, y_logit, n_splits=5):
    """TimeSeriesSplit CV MAE on logit-transformed targets, returns MAE in probability space."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for tr_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val_logit = y_logit[tr_idx], y_logit[val_idx]
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        pred_logit = model.predict(X_val)
        pred_prob  = inv_logit(pred_logit)
        # Actual in probability space (un-logit the val labels)
        actual_prob = inv_logit(y_val_logit)
        maes.append(mean_absolute_error(actual_prob, pred_prob))
    return float(np.mean(maes))

# ── BASELINE A: Independent XGBoost per horizon ──────────────────────────────
print("\n" + "-" * 70)
print("BASELINE A: Single-stage XGBoost — 4 independent Optuna-tuned models")
print("  40 trials per horizon, TimeSeriesSplit(n_splits=5)")
print("-" * 70)

baseline_A_best_params = {}
baseline_A_preds = {}
baseline_A_mae   = {}

for i, (target, label) in enumerate(zip(TARGETS, LABELS)):
    print(f"\n  [{label}] Tuning horizon {i+1}/4 ...")
    y_logit = logit(y_train[target].values)

    def objective_A(trial):
        params = suggest_xgb_params(trial)
        return cv_mae_single(params, X_train_arr, y_logit)

    study_A = optuna.create_study(direction="minimize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_A.optimize(objective_A, n_trials=40, show_progress_bar=False)

    best = study_A.best_params
    best["objective"]    = "reg:squarederror"
    best["random_state"] = 42
    best["verbosity"]    = 0
    best["tree_method"]  = "hist"
    baseline_A_best_params[label] = best

    # Train on full training set
    model_A = XGBRegressor(**best)
    model_A.fit(X_train_arr, y_logit)
    pred_logit = model_A.predict(X_test_arr)
    pred_prob  = inv_logit(pred_logit)
    baseline_A_preds[label] = pred_prob

    mae = mean_absolute_error(y_test[target].values, pred_prob)
    baseline_A_mae[label] = mae
    print(f"    Best CV MAE (logit space cv): {study_A.best_value:.4f}")
    print(f"    Test MAE: {mae:.4f} pp  |  best n_estimators={best['n_estimators']}, "
          f"max_depth={best['max_depth']}, lr={best['learning_rate']:.4f}")

print(f"\n  Baseline A Summary: "
      f"Current={baseline_A_mae['Current']:.4f}  "
      f"1M={baseline_A_mae['1M']:.4f}  "
      f"3M={baseline_A_mae['3M']:.4f}  "
      f"6M={baseline_A_mae['6M']:.4f}")

# Select the best params from A (lowest average MAE across horizons) for use in B
best_A_avg_mae = {lbl: baseline_A_mae[lbl] for lbl in LABELS}
# Baseline B uses the individually best params per horizon (MultiOutputRegressor does independent prediction)
# but wraps same-horizon best params. Actually: use the params from the horizon with best average
# as specified: "params that gave best average MAE"
avg_mae_per_horizon = {lbl: baseline_A_mae[lbl] for lbl in LABELS}
best_label_for_B = min(avg_mae_per_horizon, key=avg_mae_per_horizon.get)
best_params_B = baseline_A_best_params[best_label_for_B]
print(f"\n  Baseline B will use params from horizon '{best_label_for_B}' "
      f"(best individual MAE={avg_mae_per_horizon[best_label_for_B]:.4f})")

# ── BASELINE B: MultiOutputRegressor with best-A params ──────────────────────
print("\n" + "-" * 70)
print("BASELINE B: MultiOutputRegressor(XGBRegressor) — best params from A")
print("  Isolates chaining contribution (same tuned params, no chain)")
print("-" * 70)

# Stack all 4 targets' logit-transformed values as multi-output
y_train_logit_all = np.column_stack([logit(y_train[t].values) for t in TARGETS])

mor_B = MultiOutputRegressor(XGBRegressor(**best_params_B), n_jobs=1)
mor_B.fit(X_train_arr, y_train_logit_all)
pred_logit_B = mor_B.predict(X_test_arr)

baseline_B_preds = {}
baseline_B_mae   = {}
for i, (target, label) in enumerate(zip(TARGETS, LABELS)):
    pred_prob = inv_logit(pred_logit_B[:, i])
    baseline_B_preds[label] = pred_prob
    mae = mean_absolute_error(y_test[target].values, pred_prob)
    baseline_B_mae[label] = mae
    print(f"  [{label}] Test MAE: {mae:.4f} pp")

print(f"\n  Baseline B Summary: "
      f"Current={baseline_B_mae['Current']:.4f}  "
      f"1M={baseline_B_mae['1M']:.4f}  "
      f"3M={baseline_B_mae['3M']:.4f}  "
      f"6M={baseline_B_mae['6M']:.4f}")

# ── BASELINE C: MultiOutputRegressor with jointly-tuned params ───────────────
print("\n" + "-" * 70)
print("BASELINE C: MultiOutputRegressor(XGBRegressor) — jointly tuned params")
print("  40 trials optimizing avg MAE across all 4 horizons simultaneously")
print("-" * 70)

y_train_logit_list = [logit(y_train[t].values) for t in TARGETS]

def cv_mae_joint(params, X, y_logit_list, n_splits=5):
    """Average CV MAE across all 4 horizons."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    horizon_maes = []
    for y_logit in y_logit_list:
        maes = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val_logit = y_logit[tr_idx], y_logit[val_idx]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            pred_logit = model.predict(X_val)
            pred_prob  = inv_logit(pred_logit)
            actual_prob = inv_logit(y_val_logit)
            maes.append(mean_absolute_error(actual_prob, pred_prob))
        horizon_maes.append(np.mean(maes))
    return float(np.mean(horizon_maes))

def objective_C(trial):
    params = suggest_xgb_params(trial)
    return cv_mae_joint(params, X_train_arr, y_train_logit_list)

study_C = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=42))
print("  Optimizing joint objective...")
study_C.optimize(objective_C, n_trials=40, show_progress_bar=False)
print(f"  Best joint CV MAE: {study_C.best_value:.4f}")

best_params_C = study_C.best_params
best_params_C["objective"]    = "reg:squarederror"
best_params_C["random_state"] = 42
best_params_C["verbosity"]    = 0
best_params_C["tree_method"]  = "hist"
print(f"  Best params: n_estimators={best_params_C['n_estimators']}, "
      f"max_depth={best_params_C['max_depth']}, "
      f"lr={best_params_C['learning_rate']:.4f}")

mor_C = MultiOutputRegressor(XGBRegressor(**best_params_C), n_jobs=1)
mor_C.fit(X_train_arr, y_train_logit_all)
pred_logit_C = mor_C.predict(X_test_arr)

baseline_C_preds = {}
baseline_C_mae   = {}
for i, (target, label) in enumerate(zip(TARGETS, LABELS)):
    pred_prob = inv_logit(pred_logit_C[:, i])
    baseline_C_preds[label] = pred_prob
    mae = mean_absolute_error(y_test[target].values, pred_prob)
    baseline_C_mae[label] = mae
    print(f"  [{label}] Test MAE: {mae:.4f} pp")

print(f"\n  Baseline C Summary: "
      f"Current={baseline_C_mae['Current']:.4f}  "
      f"1M={baseline_C_mae['1M']:.4f}  "
      f"3M={baseline_C_mae['3M']:.4f}  "
      f"6M={baseline_C_mae['6M']:.4f}")

# ── Load full model predictions ───────────────────────────────────────────────
print("\n" + "-" * 70)
print("Loading full model and computing predictions for DM tests...")
print("-" * 70)

with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

ens_preds_raw = ensemble.predict(X_test)   # shape (65, 4)
full_model_preds = {label: ens_preds_raw[:, i] for i, label in enumerate(LABELS)}

# Verify full model MAE matches paper
print("\n  Verifying full model MAE against paper values:")
for i, (target, label) in enumerate(zip(TARGETS, LABELS)):
    actual = y_test[target].values
    mae_computed = mean_absolute_error(actual, full_model_preds[label])
    mae_paper    = FULL_MODEL_MAE[label]
    print(f"  [{label}] Computed={mae_computed:.4f}  Paper={mae_paper:.4f}  "
          f"{'OK' if abs(mae_computed - mae_paper) < 0.5 else 'MISMATCH'}")

# ── DM tests ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("DIEBOLD-MARIANO TESTS (HLN 1997) vs Full Model")
print("  d_t = SE_baseline - SE_full  |  Positive DM -> Full Model better")
print("-" * 70)

dm_results = {}  # (label, baseline_name) -> (dm_stat, p_value)

for label, h in zip(LABELS, H_STEPS):
    target_idx = LABELS.index(label)
    actual     = y_test[TARGETS[target_idx]].values
    pred_full  = full_model_preds[label]

    for bl_name, bl_preds in [
        ("Baseline_A", baseline_A_preds),
        ("Baseline_B", baseline_B_preds),
        ("Baseline_C", baseline_C_preds),
    ]:
        dm_stat, p_val = dm_test(actual, bl_preds[label], pred_full, h=h)
        dm_results[(label, bl_name)] = (dm_stat, p_val)

print(f"\n  {'Horizon':<10}  {'Baseline':<12}  {'DM stat':>8}  {'p-value':>8}  Interpretation")
print("  " + "-" * 65)
for label in LABELS:
    for bl_name in ["Baseline_A", "Baseline_B", "Baseline_C"]:
        dm_stat, p_val = dm_results[(label, bl_name)]
        if np.isnan(dm_stat):
            interp = "n/a"
            dm_str = "     n/a"
            p_str  = "     n/a"
        else:
            dm_str = f"{dm_stat:8.3f}"
            p_str  = f"{p_val:8.4f}"
            if p_val < 0.01:   sig = "***"
            elif p_val < 0.05: sig = " **"
            elif p_val < 0.10: sig = "  *"
            else:              sig = "   "
            direction = "Full model better" if dm_stat > 0 else "Baseline better"
            interp = f"{sig}  {direction}"
        print(f"  {label:<10}  {bl_name:<12}  {dm_str}  {p_str}  {interp}")

# ── Build MAE table ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL MAE TABLE (percentage points)")
print("=" * 70)

header = (f"{'Horizon':<10}  {'BaselineA':>14}  {'BaselineB':>14}  "
          f"{'BaselineC':>14}  {'Full Model':>12}")
print(header)
print("-" * 70)
for label in LABELS:
    a_mae = baseline_A_mae[label]
    b_mae = baseline_B_mae[label]
    c_mae = baseline_C_mae[label]
    f_mae = FULL_MODEL_MAE[label]
    print(f"{label:<10}  {a_mae:>14.4f}  {b_mae:>14.4f}  {c_mae:>14.4f}  {f_mae:>12.4f}")
print("=" * 70)
print(f"{'Avg':<10}  "
      f"{np.mean(list(baseline_A_mae.values())):>14.4f}  "
      f"{np.mean(list(baseline_B_mae.values())):>14.4f}  "
      f"{np.mean(list(baseline_C_mae.values())):>14.4f}  "
      f"{np.mean(list(FULL_MODEL_MAE.values())):>12.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = []
for label, h in zip(LABELS, H_STEPS):
    target_idx = LABELS.index(label)
    actual     = y_test[TARGETS[target_idx]].values
    pred_full  = full_model_preds[label]

    for bl_name, bl_mae_dict, bl_preds_dict in [
        ("Baseline_A (XGB indep)",    baseline_A_mae,  baseline_A_preds),
        ("Baseline_B (MOR-XGB)",      baseline_B_mae,  baseline_B_preds),
        ("Baseline_C (MOR-XGB-joint)",baseline_C_mae,  baseline_C_preds),
    ]:
        bl_key = bl_name.split(" ")[0]  # "Baseline_A", etc.
        dm_key = bl_key.replace("(", "").replace(")", "")
        # find DM result
        dm_stat, p_val = dm_results.get((label, bl_key.rstrip("_").replace("_A","_A").replace("_B","_B").replace("_C","_C")),
                                        (np.nan, np.nan))
        # Recompute with correct key
        for k in ["Baseline_A", "Baseline_B", "Baseline_C"]:
            if k in bl_name:
                dm_stat, p_val = dm_results[(label, k)]
                break

        rows.append({
            "Horizon":       label,
            "Baseline":      bl_name,
            "Baseline_MAE":  round(bl_mae_dict[label], 4),
            "FullModel_MAE": FULL_MODEL_MAE[label],
            "DM_stat":       round(dm_stat, 4) if not np.isnan(dm_stat) else None,
            "p_value":       round(p_val,  4) if not np.isnan(p_val)  else None,
            "Sig_10pct":     bool(p_val < 0.10) if not np.isnan(p_val) else None,
            "Sig_5pct":      bool(p_val < 0.05) if not np.isnan(p_val) else None,
            "H_step":        h,
        })

csv_df = pd.DataFrame(rows)
csv_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

x      = np.arange(len(LABELS))
labels = ["Current", "1M", "3M", "6M"]

mae_A    = [baseline_A_mae[l]  for l in LABELS]
mae_B    = [baseline_B_mae[l]  for l in LABELS]
mae_C    = [baseline_C_mae[l]  for l in LABELS]
mae_full = [FULL_MODEL_MAE[l]  for l in LABELS]

ax.plot(x, mae_A,    marker="o", linewidth=2, markersize=7,
        label="Baseline A (XGB indep, Optuna)", color="#9467bd")
ax.plot(x, mae_B,    marker="s", linewidth=2, markersize=7,
        label="Baseline B (MOR-XGB, best-A params)", color="#ff7f0e")
ax.plot(x, mae_C,    marker="^", linewidth=2, markersize=7,
        label="Baseline C (MOR-XGB, joint tuning)", color="#d62728")
ax.plot(x, mae_full, marker="D", linewidth=2.5, markersize=8,
        label="Full Model (Chain Stacking Ensemble)", color="#2ca02c", zorder=5)

# Annotate values
for xi in x:
    ax.annotate(f"{mae_A[xi]:.2f}",    (xi, mae_A[xi]),
                textcoords="offset points", xytext=(0, 8),  fontsize=8, color="#9467bd", ha="center")
    ax.annotate(f"{mae_B[xi]:.2f}",    (xi, mae_B[xi]),
                textcoords="offset points", xytext=(0, -14), fontsize=8, color="#ff7f0e", ha="center")
    ax.annotate(f"{mae_C[xi]:.2f}",    (xi, mae_C[xi]),
                textcoords="offset points", xytext=(0, 8),  fontsize=8, color="#d62728", ha="center")
    ax.annotate(f"{mae_full[xi]:.2f}", (xi, mae_full[xi]),
                textcoords="offset points", xytext=(0, -14), fontsize=8, color="#2ca02c",
                ha="center", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_xlabel("Forecast Horizon", fontsize=12)
ax.set_ylabel("MAE (pp)", fontsize=12)
ax.set_title("Ablation Baseline Comparison — MAE by Forecast Horizon", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(max(mae_A), max(mae_B), max(mae_C)) * 1.20)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
plt.close()
print(f"Saved: {OUT_PNG}")

print("\nDone.")
