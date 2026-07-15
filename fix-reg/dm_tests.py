"""
Diebold-Mariano tests: our ensemble vs each baseline, all 4 horizons.
Loss differential: d_t = SE_baseline_t - SE_ours_t  (squared error)
Positive DM stat  → our model has lower loss (better)
Negative DM stat  → baseline has lower loss (better)
Uses Harvey-Leybourne-Newbold (1997) small-sample modification, t(T-1) distribution.
Bandwidth set to h-1 (optimal for h-step-ahead forecasts, NW Bartlett kernel).

Run from: RecessionRadar/RecessionRadar/
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multioutput import RegressorChain
from xgboost import DMatrix, train as xgb_train
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────
DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
SPLIT      = "2020-01-01"

TARGETS = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]
LABELS  = ["Current", "1-Month", "3-Month", "6-Month"]
# Forecast step-ahead length per horizon (for NW bandwidth)
H_STEPS = [1, 1, 3, 6]

# ── classes required for unpickling the ensemble ────────────────
eps = 1e-8
def _inv_logit(z):
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
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainLightGBMModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        X = _san(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainRandomForestModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)

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

# ── helpers ─────────────────────────────────────────────────────
EPS = 1e-6
def logit(y):
    y_s = np.clip(np.array(y, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(y_s / (1 - y_s))
def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-np.array(z, dtype=float))) * 100.0
def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ── DM test (Harvey-Leybourne-Newbold 1997) ─────────────────────
def dm_test(actual, pred_baseline, pred_ours, h=1):
    """
    Two-sided HLN-modified Diebold-Mariano test using squared-error loss.
    d_t = SE_baseline_t - SE_ours_t
    Returns (DM_stat, p_value).
    Positive DM → our model is better; negative DM → baseline is better.
    """
    T = len(actual)
    e1 = actual - pred_baseline   # baseline errors
    e2 = actual - pred_ours        # our errors

    d = e1**2 - e2**2              # loss differential
    d_bar = np.mean(d)

    # Long-run variance via Bartlett NW with bandwidth h-1
    bw = max(h - 1, 0)
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for k in range(1, bw + 1):
        w = 1.0 - k / (bw + 1)    # Bartlett weight
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        lrv += 2 * w * gamma_k

    if lrv <= 0:
        return np.nan, np.nan

    dm_raw = d_bar / np.sqrt(lrv / T)

    # HLN small-sample correction: scale factor × t(T-1) critical values
    hlr = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    dm_hlr = dm_raw * hlr

    p_value = 2 * (1 - stats.t.cdf(abs(dm_hlr), df=T-1))
    return dm_hlr, p_value

# ── load data ────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

X_train = clean(train_df.drop(columns=TARGETS + ["date"]))
X_test  = clean(test_df.drop(columns=TARGETS + ["date"]))
y_train = clean(train_df[TARGETS])
y_test  = clean(test_df[TARGETS])       # NaN tails ffilled → matches baselines.py

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── baseline 1: naive mean ───────────────────────────────────────
print("Building Naive Mean baseline...")
naive_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    naive_preds[lbl] = np.full(len(y_test), y_train[t].mean())

# ── baseline 2: probit (yield curve) ────────────────────────────
print("Building Probit (Yield Curve) baseline...")
spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1, 1)
probit_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    reg = LinearRegression()
    reg.fit(spread_train, logit(y_train[t].values))
    probit_preds[lbl] = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)

# ── baseline 3: single-stage XGBoost ────────────────────────────
print("Building Single-Stage XGBoost baseline...")
XGB_PARAMS = {
    "objective": "reg:squarederror", "max_depth": 5, "eta": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9, "seed": 42, "verbosity": 0,
}
sxgb_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    dtrain = DMatrix(X_train.values, label=logit(y_train[t].values))
    dtest  = DMatrix(X_test.values)
    model  = xgb_train(XGB_PARAMS, dtrain, num_boost_round=500)
    sxgb_preds[lbl] = np.clip(inv_logit(model.predict(dtest)), 0, 100)

# ── ensemble predictions ─────────────────────────────────────────
print("Loading ensemble and running predictions...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)
ens_preds_raw = ensemble.predict(X_test)   # shape (65, 4)
ens_preds = {lbl: ens_preds_raw[:, i] for i, lbl in enumerate(LABELS)}

# ── run DM tests ────────────────────────────────────────────────
print("\nRunning DM tests...\n")

baselines = {
    "Naive Mean":           naive_preds,
    "Probit (YC)":          probit_preds,
    "Single-Stage XGB":     sxgb_preds,
}

results = {}   # results[(horizon, baseline)] = (dm, p)

header = f"{'Horizon':<10}  {'Baseline':<20}  {'DM stat':>8}  {'p-value':>8}  {'Signif':>6}  Direction"
print(header)
print("-" * len(header))

for lbl, h in zip(LABELS, H_STEPS):
    actual = y_test[TARGETS[LABELS.index(lbl)]].values
    our    = ens_preds[lbl]
    for bl_name, bl_dict in baselines.items():
        dm, p = dm_test(actual, bl_dict[lbl], our, h=h)
        results[(lbl, bl_name)] = (dm, p)

        if np.isnan(dm):
            sig, direction = "  n/a", "n/a"
        elif p < 0.01:
            sig = " ***"
        elif p < 0.05:
            sig = "  **"
        elif p < 0.10:
            sig = "   *"
        else:
            sig = "    "

        if not np.isnan(dm):
            direction = "Ours better" if dm > 0 else "Baseline better"
        else:
            direction = "n/a"

        print(f"{lbl:<10}  {bl_name:<20}  {dm:>8.3f}  {p:>8.4f}  {sig}  {direction}")
    print()

# ── print clean table for paper ─────────────────────────────────
print("\n" + "=" * 80)
print("DIEBOLD-MARIANO TEST RESULTS")
print("Loss: squared error  |  d_t = SE_baseline - SE_ours  |  HLN (1997) modified")
print("Positive DM → our model has lower squared error (better)")
print("Negative DM → baseline has lower squared error (better)")
print("Significance: *** p<0.01  ** p<0.05  * p<0.10  (two-sided t(T-1), T=65)")
print("=" * 80)
print(f"\n{'Horizon':<10}  {'vs Naive Mean':^22}  {'vs Probit (YC)':^22}  {'vs Single XGB':^22}")
print(f"{'':10}  {'DM stat':>8}  {'p-val':>6}  {'sig':>3}  {'DM stat':>8}  {'p-val':>6}  {'sig':>3}  {'DM stat':>8}  {'p-val':>6}  {'sig':>3}")
print("-" * 80)

for lbl in LABELS:
    row = f"{lbl:<10}"
    for bl_name in ["Naive Mean", "Probit (YC)", "Single-Stage XGB"]:
        dm, p = results[(lbl, bl_name)]
        if np.isnan(dm):
            row += f"  {'n/a':>8}  {'n/a':>6}  {'':>3}"
        else:
            if p < 0.01:   sig = "***"
            elif p < 0.05: sig = " **"
            elif p < 0.10: sig = "  *"
            else:           sig = "   "
            row += f"  {dm:>8.3f}  {p:>6.4f}  {sig}"
    print(row)

print("=" * 80)
print("\nNote: bandwidth h for NW estimator set to h-1 per optimal theory for h-step forecasts.")
print("Current & 1M: h=1 (bandwidth 0, no autocorrelation correction).")
print("3M: h=3 (bandwidth 2).  6M: h=6 (bandwidth 5).")

# ── save results to CSV ──────────────────────────────────────────
rows = []
for lbl in LABELS:
    for bl_name, bl_key in [("Naive Mean", "Naive Mean"),
                             ("Probit (YC)", "Probit (YC)"),
                             ("Single-Stage XGB", "Single-Stage XGB")]:
        dm, p = results[(lbl, bl_key)]
        rows.append({
            "Horizon":   lbl,
            "Baseline":  bl_name,
            "DM_stat":   round(dm, 4) if not np.isnan(dm) else None,
            "p_value":   round(p,  4) if not np.isnan(p)  else None,
            "Significant_10pct": (p < 0.10) if not np.isnan(p) else None,
            "Significant_5pct":  (p < 0.05) if not np.isnan(p) else None,
        })

out_csv = "fix-reg/dm_test_results.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"\n✓ Results saved to {out_csv}")
