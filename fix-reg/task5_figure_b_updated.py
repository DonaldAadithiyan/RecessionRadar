"""
Regenerate Figure B using 40% calibration (Nov 1998 – Dec 2019).
Saves figure_B_updated_40pct.png into task1_outputs/ alongside the original.
Same visual style as Figure B; band will be visibly wider and more honest.
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS       = ["Current", "1M", "3M", "6M"]
DATA_PATH    = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH   = "fix-reg/models/full_chain_stacking.pkl"
OUT_DIR      = "fix-reg/task1_outputs"
SPLIT        = "2020-01-01"
CAL_FRAC     = 0.40
GAMMA        = 0.005
ALPHA_TARGET = 0.10

eps = 1e-8
def safe_logit(y):
    return np.log(np.clip(np.clip(y,0,100)/100,eps,1-eps) /
                  (1-np.clip(np.clip(y,0,100)/100,eps,1-eps)))
def safe_inv_logit(z):
    return np.clip(1/(1+np.exp(-np.clip(z,-50,50)))*100, 0, 100)
def sanitize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+','_',c) for c in df.columns]
    return df

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {}; self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds; self.model = None
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

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def clean(d):
    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

cal_size = int(len(train_df) * CAL_FRAC)
cal_df   = train_df.iloc[-cal_size:].copy()

X_cal  = clean(cal_df.drop(columns=recession_targets + ["date"]))
y_cal  = cal_df[recession_targets].values
X_test = clean(test_df.drop(columns=recession_targets + ["date"]))
y_test = test_df[recession_targets].values
dates_test = test_df["date"].values

print(f"Calibration: {cal_size} rows  "
      f"({cal_df['date'].min().date()} → {cal_df['date'].max().date()})")

with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

preds_cal  = ensemble.predict(X_cal)
preds_test = ensemble.predict(X_test)

# ── ACI — 40% calibration, same Gibbs & Candès update ────────
results = {}
for h_idx, h_label in enumerate(LABELS):
    cal_mask   = ~np.isnan(y_cal[:, h_idx])
    cal_scores = np.abs(preds_cal[cal_mask, h_idx] - y_cal[cal_mask, h_idx])

    alpha = ALPHA_TARGET
    lowers, uppers, covered, alphas = [], [], [], []

    for t in range(len(y_test)):
        q  = np.quantile(cal_scores, np.clip(1 - alpha, 0.0, 1.0))
        lo = np.clip(preds_test[t, h_idx] - q, 0, 100)
        hi = np.clip(preds_test[t, h_idx] + q, 0, 100)
        lowers.append(lo); uppers.append(hi); alphas.append(alpha)

        y_true = y_test[t, h_idx]
        if np.isnan(y_true):
            covered.append(np.nan)
        else:
            inside  = float(lo <= y_true <= hi)
            covered.append(inside)
            alpha = np.clip(alpha + GAMMA * (ALPHA_TARGET - (1 - inside)), 0.001, 0.999)

    lowers  = np.maximum(np.array(lowers), 0.0)
    uppers  = np.minimum(np.array(uppers), 100.0)
    covered = np.array(covered)
    cov_rate = np.nanmean(covered) * 100
    results[h_label] = {
        "lowers": lowers, "uppers": uppers, "covered": covered,
        "preds": preds_test[:, h_idx], "actuals": y_test[:, h_idx],
        "mean_width": (uppers - lowers).mean(), "coverage_rate": cov_rate,
    }
    print(f"  {h_label}: width={results[h_label]['mean_width']:.2f}pp  "
          f"coverage={cov_rate:.1f}%")

# ── Figure B (updated, 40% calibration) ───────────────────────
r6m      = results["6M"]
dates_dt = pd.to_datetime(dates_test)
valid_6m = ~np.isnan(r6m["actuals"])

# Identify covered vs missed points (for scatter colouring)
covered_mask = valid_6m & (r6m["covered"] == 1.0)
missed_mask  = valid_6m & (r6m["covered"] == 0.0)

fig, ax = plt.subplots(figsize=(13, 6))

# ACI band — identical colour/alpha to original Figure B
ax.fill_between(dates_dt, r6m["lowers"], r6m["uppers"],
                alpha=0.30, color="#2196F3", label="ACI 90% interval (40% cal)", zorder=2)

# Actual line
ax.plot(dates_dt[valid_6m], r6m["actuals"][valid_6m],
        color="#1a1a2e", linewidth=2.5, label="Actual recession prob (6M)", zorder=5)

# Predicted dashed line
ax.plot(dates_dt, r6m["preds"],
        color="#e15759", linewidth=1.8, linestyle="--",
        label="Predicted (6M horizon)", zorder=4, alpha=0.85)

# Covered / missed scatter dots
ax.scatter(dates_dt[covered_mask], r6m["actuals"][covered_mask],
           s=36, color="#2ca02c", zorder=6, label="Covered", marker="o")
ax.scatter(dates_dt[missed_mask],  r6m["actuals"][missed_mask],
           s=50, color="#d62728", zorder=6, label="Missed",   marker="X")

# Event verticals — identical to original
covid_date   = pd.Timestamp("2020-03-01")
tighten_date = pd.Timestamp("2022-03-01")
ax.axvline(covid_date,   color="#c62828", linewidth=2.2, linestyle="--",
           alpha=0.9, label="Mar 2020 (COVID)")
ax.axvline(tighten_date, color="#e65100", linewidth=2.2, linestyle="--",
           alpha=0.9, label="Mar 2022 (Fed tightening)")

ylim = ax.get_ylim()
ax.text(covid_date,   ylim[1]*0.98, " COVID\n Mar 2020",
        color="#c62828", fontsize=9, va="top", fontweight="bold")
ax.text(tighten_date, ylim[1]*0.98, " Fed Tightening\n Mar 2022",
        color="#e65100", fontsize=9, va="top", fontweight="bold")

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Recession Probability (%)", fontsize=12)
ax.set_title(
    f"Figure B (Updated) — 6M Horizon ACI Uncertainty Band (2020–2025)\n"
    f"40% calibration (Nov 1998 – Dec 2019, 254 obs, 16 recession months)  |  "
    f"Coverage = {r6m['coverage_rate']:.1f}%  |  Mean width = {r6m['mean_width']:.2f} pp",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9.5, loc="upper right", ncol=2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
plt.tight_layout()

out_path = f"{OUT_DIR}/figure_B_updated_40pct.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure B (updated) saved → {out_path}")

# ── Width comparison: original vs updated ─────────────────────
orig_widths = {"Current": 2.49, "1M": 2.41, "3M": 5.14, "6M": 12.34}
print("\nWidth comparison (original 20% vs updated 40%):")
print(f"{'Horizon':<10} {'Original':>10} {'Updated':>10} {'Change':>10}")
for h in LABELS:
    ow = orig_widths[h]
    nw = results[h]["mean_width"]
    print(f"{h:<10} {ow:>10.2f} {nw:>10.2f} {nw-ow:>+10.2f}")
