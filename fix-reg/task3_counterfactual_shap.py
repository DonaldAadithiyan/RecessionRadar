"""
TASK 3 — Counterfactual SHAP on Stress Test Scenarios
Compares feature attributions between baseline and interest rate shock scenarios
for the 6M horizon model, revealing the monetary transmission mechanism.

The five scenarios replicate those from scenario_test.py:
  -100bps, -50bps, Baseline, +50bps, +100bps applied to 3M and 1Y rates.
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# ── Class definitions for unpickling ─────────────────────────
recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS = ["Current", "1M", "3M", "6M"]

eps = 1e-8
def safe_logit(y):
    return np.log(np.clip(np.clip(y,0,100)/100, eps, 1-eps) /
                  (1 - np.clip(np.clip(y,0,100)/100, eps, 1-eps)))
def safe_inv_logit(z):
    return np.clip(1/(1+np.exp(-np.clip(z,-50,50)))*100, 0, 100)
def sanitize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+','_',c) for c in df.columns]
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
    def __init__(self): self.chain_model = None; self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainLightGBMModel:
    def __init__(self): self.chain_model = None; self.scaler = None
    def predict(self, X):
        X = sanitize_columns(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainRandomForestModel:
    def __init__(self): self.chain_model = None; self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainStackingEnsemble:
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {'CatBoost': FullChainCatBoostModel,
                            'LightGBM': FullChainLightGBMModel,
                            'RandomForest': FullChainRandomForestModel}
        self.meta_models = {}; self.cv_folds = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler = {}; self.fitted_base_models = {}
    def _engineer_meta_features(self, *bp):
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
        for i, t in enumerate(recession_targets):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf = self._engineer_meta_features(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)

# ── Config ───────────────────────────────────────────────────
DATA_PATH    = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH   = "fix-reg/models/full_chain_stacking.pkl"
OUT_DIR      = "fix-reg/task3_outputs"
SPLIT        = "2020-01-01"
RATE_FEATURES = ["3_months_rate", "1_year_rate"]
DIFF_FEATURES = ["3_months_rate_diff3"]
SHOCKS_BPS   = [-100, -50, 0, 50, 100]
SHOCK_VALS   = [s / 100 for s in SHOCKS_BPS]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data and build scenario inputs (replicate scenario_test.py) ──
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def clean(d):
    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

test_df = df[df["date"] >= SPLIT].copy()
X_test_all = clean(test_df.drop(columns=recession_targets + ["date"]))

# Baseline = last observation (May 2025)
baseline_row = X_test_all.iloc[[-1]].copy()
baseline_date = test_df["date"].iloc[-1]
print(f"  Baseline date: {baseline_date.strftime('%Y-%m-%d')}")
print(f"  3M rate: {baseline_row['3_months_rate'].values[0]:.3f}%")
print(f"  1Y rate: {baseline_row['1_year_rate'].values[0]:.3f}%")

# Build 5 scenario rows
scenario_rows = []
for shock_val, shock_bps in zip(SHOCK_VALS, SHOCKS_BPS):
    row = baseline_row.copy()
    for feat in RATE_FEATURES:
        if feat in row.columns:
            row[feat] = row[feat] + shock_val
    for feat in DIFF_FEATURES:
        if feat in row.columns:
            row[feat] = row[feat] + shock_val
    scenario_rows.append(row)
    spread = row['10_year_rate'].values[0] - row['3_months_rate'].values[0]
    print(f"  {shock_bps:+4d}bps → 3M={row['3_months_rate'].values[0]:.3f}%  "
          f"10Y-3M spread={spread*100:.0f}bps")

# ── Load model ────────────────────────────────────────────────
print("\nLoading model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

# ── Get ensemble predictions for all scenarios ────────────────
print("Getting ensemble predictions for all scenarios...")
scenario_preds = []
for row, shock_bps in zip(scenario_rows, SHOCKS_BPS):
    pred = ensemble.predict(row)[0]   # shape (4,) = [Current, 1M, 3M, 6M]
    scenario_preds.append(pred)
    print(f"  {shock_bps:+4d}bps: " +
          "  ".join(f"{l}:{p:.2f}%" for l, p in zip(LABELS, pred)))

preds_arr   = np.array(scenario_preds)   # (5, 4)
baseline_idx = SHOCKS_BPS.index(0)
baseline_pred_6m = preds_arr[baseline_idx, 3]

# ── Build SHAP input for 6M estimator ─────────────────────────
# The 6M estimator (chain.estimators_[3]) receives:
#   scaled_X + chain_pred_from_est0 + chain_pred_from_est1 + chain_pred_from_est2
# We need to reconstruct this augmented input for each scenario.

cb_model = ensemble.fitted_base_models["CatBoost"]
chain    = cb_model.chain_model
scaler   = cb_model.scaler
orig_feature_names = list(X_test_all.columns)

def build_chain_aug_input(raw_row, chain_obj, scaler_obj):
    """Build the augmented input for 6M (estimators_[3]) from raw unscaled row."""
    X_s = pd.DataFrame(scaler_obj.transform(raw_row),
                       columns=raw_row.columns, index=raw_row.index)
    X_aug = X_s.values.copy()
    for est in chain_obj.estimators_[:3]:   # first 3 estimators feed into 6M
        pred_i = est.predict(X_aug).reshape(-1, 1)
        X_aug  = np.hstack([X_aug, pred_i])
    return X_aug   # shape (1, n_orig + 3)

print("\nBuilding augmented 6M inputs for each scenario...")
aug_inputs = []
for row, shock_bps in zip(scenario_rows, SHOCKS_BPS):
    aug = build_chain_aug_input(row, chain, scaler)
    aug_inputs.append(aug[0])   # flatten to 1D
    print(f"  {shock_bps:+4d}bps: aug_input shape {aug.shape}")

aug_inputs = np.array(aug_inputs)   # (5, n_aug)

# ── SHAP on 6M estimator for all 5 scenarios ─────────────────
print("\nComputing SHAP values on 6M estimator for all 5 scenario inputs...")
est_6m    = chain.estimators_[3]
explainer = shap.TreeExplainer(est_6m)

n_orig = len(orig_feature_names)
shap_per_scenario = []
for i, (aug_inp, shock_bps) in enumerate(zip(aug_inputs, SHOCKS_BPS)):
    sv = explainer.shap_values(aug_inp.reshape(1, -1))[0]   # (n_aug,)
    sv_orig = sv[:n_orig]   # only original feature SHAP values
    shap_per_scenario.append(sv_orig)
    print(f"  {shock_bps:+4d}bps: SHAP computed, top feature: "
          f"{orig_feature_names[np.abs(sv_orig).argmax()]} "
          f"({sv_orig[np.abs(sv_orig).argmax()]:.4f})")

shap_per_scenario = np.array(shap_per_scenario)   # (5, n_orig)

# Baseline SHAP vector
baseline_shap = shap_per_scenario[baseline_idx]

# ── Attribution differences: scenario − baseline ──────────────
# Focus on -100bps and +100bps
results = {}
for shock_bps in [-100, 100]:
    s_idx    = SHOCKS_BPS.index(shock_bps)
    delta    = shap_per_scenario[s_idx] - baseline_shap   # (n_orig,)
    abs_delta = np.abs(delta)
    top8_idx  = np.argsort(abs_delta)[::-1][:8]

    top8_rows = []
    for fidx in top8_idx:
        top8_rows.append({
            "Feature":         orig_feature_names[fidx],
            "SHAP_Baseline":   round(float(baseline_shap[fidx]), 4),
            f"SHAP_{shock_bps}bps": round(float(shap_per_scenario[s_idx, fidx]), 4),
            "Difference":      round(float(delta[fidx]), 4),
            "Direction":       "TOWARD recession" if delta[fidx] > 0 else "AWAY from recession",
        })
    results[shock_bps] = {
        "top8_df":  pd.DataFrame(top8_rows),
        "delta":    delta,
        "shap_vec": shap_per_scenario[s_idx],
        "pred_6m":  preds_arr[s_idx, 3],
    }

# ── Key number: % of probability increase explained by top 3 ──
# Using -100bps scenario (which maps to the +16.68pp figure in the task)
neg100_pred = results[-100]["pred_6m"]
neg100_delta_total = neg100_pred - baseline_pred_6m
neg100_top3_shap_sum = results[-100]["delta"][
    np.argsort(np.abs(results[-100]["delta"]))[::-1][:3]
].sum()

print(f"\n--- -100bps scenario ---")
print(f"Baseline 6M prediction:  {baseline_pred_6m:.4f}%")
print(f"-100bps 6M prediction:   {neg100_pred:.4f}%")
print(f"Total probability change: {neg100_delta_total:+.4f}pp")

# Note: the task references +16.68pp for +100bps easing, which matches -100bps easing
# In the model convention from scenario_test.py, lower rates = more recession risk
if abs(neg100_delta_total) > 0.01:
    top3_pct = (neg100_top3_shap_sum / neg100_delta_total) * 100
    print(f"Top-3 SHAP difference sum: {neg100_top3_shap_sum:+.4f}")
    print(f"% of change explained by top 3: {top3_pct:.1f}%")
else:
    top3_pct = float("nan")
    print("Probability change near zero — % calculation not meaningful")

# Also check +100bps for comparison
pos100_pred = results[100]["pred_6m"]
pos100_delta_total = pos100_pred - baseline_pred_6m
print(f"\n--- +100bps scenario ---")
print(f"+100bps 6M prediction:    {pos100_pred:.4f}%")
print(f"Total probability change: {pos100_delta_total:+.4f}pp")

# ── Save attribution tables ───────────────────────────────────
for shock_bps in [-100, 100]:
    df_out = results[shock_bps]["top8_df"]
    fname  = f"{OUT_DIR}/attribution_change_{shock_bps}bps.csv"
    df_out.to_csv(fname, index=False)
    print(f"\nAttribution table {shock_bps}bps → {fname}")
    print(df_out.to_string(index=False))

# Save summary numbers
summary_nums = {
    "baseline_6m_pred":        round(float(baseline_pred_6m), 4),
    "neg100bps_6m_pred":       round(float(neg100_pred), 4),
    "neg100bps_delta_pp":      round(float(neg100_delta_total), 4),
    "pos100bps_6m_pred":       round(float(pos100_pred), 4),
    "pos100bps_delta_pp":      round(float(pos100_delta_total), 4),
    "neg100bps_top3_shap_pct": round(float(top3_pct), 2) if not np.isnan(top3_pct) else None,
    "baseline_date":           baseline_date.strftime("%Y-%m-%d"),
}
import json
with open(f"{OUT_DIR}/task3_summary_numbers.json", "w") as f:
    json.dump(summary_nums, f, indent=2)
print(f"\nSummary numbers → {OUT_DIR}/task3_summary_numbers.json")
print(json.dumps(summary_nums, indent=2))

# ── Figure D — Horizontal bar chart for -100bps ──────────────
top8_df = results[-100]["top8_df"]
feat_labels = [f.replace("_", " ") for f in top8_df["Feature"]]
diffs   = top8_df["Difference"].values
colors  = ["#c62828" if d > 0 else "#1565c0" for d in diffs]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = range(len(feat_labels))
bars  = ax.barh(y_pos, diffs, color=colors, edgecolor="white", linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(feat_labels, fontsize=9)
ax.axvline(x=0, color="#333", linewidth=1.5)
ax.set_xlabel("SHAP Attribution Change (scenario − baseline)", fontsize=11)
ax.set_title(
    "Figure D — Feature Attribution Change: −100bps Easing vs Baseline (6M Horizon)\n"
    "Red = pushes toward recession  |  Blue = pushes away from recession",
    fontsize=11, fontweight="bold"
)
for bar, d in zip(bars, diffs):
    offset = 0.001 if d >= 0 else -0.001
    ax.text(d + offset, bar.get_y() + bar.get_height()/2,
            f"{d:+.4f}", va="center",
            ha="left" if d >= 0 else "right",
            fontsize=8.5, fontweight="bold",
            color="#c62828" if d > 0 else "#1565c0")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.invert_yaxis()   # most important at top
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/figure_D_counterfactual_shap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure D saved → {OUT_DIR}/figure_D_counterfactual_shap.png")
print("\nTASK 3 COMPLETE")
