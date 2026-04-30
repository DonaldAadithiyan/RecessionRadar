"""
Generate recession.png and figure 02.png
Run from: RecessionRadar/RecessionRadar/
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

# ── Paths (relative to RecessionRadar/RecessionRadar/) ─────────
DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
OUT_DIR    = "fix-reg"
SPLIT      = "2020-01-01"

# ── Required classes for unpickling ───────────────────────────
recession_targets = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]
LABELS  = ["Current", "1M", "3M", "6M"]
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red

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
def clean(d):
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.ffill().bfill().fillna(0)

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
            mf  = self._engineer_meta_features(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)


# ══════════════════════════════════════════════════════════════
# LOAD DATA + MODEL
# ══════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

test_df  = df[df["date"] >= SPLIT].copy()
X_test   = clean(test_df.drop(columns=recession_targets + ["date"]))
y_test   = test_df[recession_targets].copy()
dates    = test_df["date"].values           # datetime64 array for x-axis

print(f"  Test rows: {len(test_df)}  ({pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()})")

print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

print("Running predictions...")
preds = ensemble.predict(X_test)           # shape (n, 4), already in 0-100 %
print("  Prediction range check:")
for i, lbl in enumerate(LABELS):
    print(f"    {lbl}: [{preds[:,i].min():.2f}, {preds[:,i].max():.2f}]")


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — recession.png
# ══════════════════════════════════════════════════════════════
print("\nGenerating recession.png...")

covid_date   = pd.Timestamp("2020-03-01")
fed_date     = pd.Timestamp("2022-03-01")

fig, axes = plt.subplots(4, 2, figsize=(10, 8))
fig.suptitle("Recession Probability Predictions — All Horizons",
             fontsize=12, fontweight="bold", y=1.01)

for i, (lbl, col) in enumerate(zip(LABELS, COLOURS)):
    actual = y_test[recession_targets[i]].values.astype(float)
    pred   = preds[:, i]

    # ── time series (left column) ──────────────────────────────
    ax_ts = axes[i, 0]
    ax_ts.plot(dates, actual, color="black",  linewidth=1.5, label="Actual")
    ax_ts.plot(dates, pred,   color=col, linewidth=1.5,
               linestyle="--", label="Predicted")

    ax_ts.axvline(np.datetime64(covid_date), color="red",    linestyle="--",
                  linewidth=1, alpha=0.8, label="COVID (Mar 2020)")
    ax_ts.axvline(np.datetime64(fed_date),   color="orange", linestyle="--",
                  linewidth=1, alpha=0.8, label="Fed Tightening (Mar 2022)")

    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    ax_ts.set_ylabel("Probability (%)", fontsize=9)
    ax_ts.set_title(f"{lbl} Horizon — Time Series", fontsize=9, fontweight="bold")
    ax_ts.set_ylim(bottom=0)
    ax_ts.tick_params(labelsize=9)
    ax_ts.legend(fontsize=7, loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # ── scatter (right column) ─────────────────────────────────
    ax_sc = axes[i, 1]

    # only plot where actual is not NaN
    mask = ~np.isnan(actual)
    act_m, pred_m = actual[mask], pred[mask]

    # clip to 0-30 so sparse high-recession points don't blank the plot
    act_clip  = np.clip(act_m,  0, 30)
    pred_clip = np.clip(pred_m, 0, 30)
    ax_sc.scatter(act_clip, pred_clip, alpha=0.6, s=20, color=col)
    ax_sc.plot([0, 30], [0, 30], "r-", linewidth=1.2)   # diagonal ref line
    ax_sc.set_xlim(0, 30)
    ax_sc.set_ylim(0, 30)

    # MAE on ffill-filled full array — matches clean_data() used in training notebook
    actual_filled = pd.Series(actual).ffill().bfill().values
    mae = mean_absolute_error(actual_filled, pred)
    ax_sc.set_title(f"{lbl} — MAE = {mae:.2f}%", fontsize=9, fontweight="bold")
    ax_sc.set_xlabel("Actual (%)", fontsize=9)
    ax_sc.set_ylabel("Predicted (%)", fontsize=9)
    ax_sc.tick_params(labelsize=9)
    ax_sc.grid(True, alpha=0.3)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "recession.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {out1}")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — figure 02.png  (pipeline architecture diagram)
# ══════════════════════════════════════════════════════════════
print("\nGenerating figure 02.png...")

fig, ax = plt.subplots(figsize=(5, 7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# helpers ------------------------------------------------------------------
def box(ax, x, y, w, h, label, fc, tc="white", fs=8, bold=False, text2=None):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.015",
                           facecolor=fc, edgecolor="#555555", linewidth=0.8)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    cy = y + h/2 + (0.012 if text2 else 0)
    ax.text(x + w/2, cy, label,
            ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, fontfamily="sans-serif")
    if text2:
        ax.text(x + w/2, y + h/2 - 0.012, text2,
                ha="center", va="center", fontsize=fs-1,
                color=tc, fontfamily="sans-serif")

def arrow(ax, x0, y0, x1, y1, label=None):
    ap = FancyArrowPatch((x0, y0), (x1, y1),
                         arrowstyle="->",
                         color="#333333",
                         linewidth=1.5,
                         connectionstyle="arc3,rad=0.0",
                         mutation_scale=10)
    ax.add_patch(ap)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.05, my, label, ha="left", va="center",
                fontsize=7.5, color="#333333")

# ── LAYER 1 — INPUT (top) ─────────────────────────────────────
INPUT_Y  = 0.80
INPUT_H  = 0.16

patch_in = FancyBboxPatch((0.04, INPUT_Y), 0.92, INPUT_H,
                           boxstyle="round,pad=0.015",
                           facecolor="#dce8f5", edgecolor="#555555", linewidth=0.8)
ax.add_patch(patch_in)
ax.text(0.50, INPUT_Y + INPUT_H - 0.025, "INPUT INDICATORS",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#1a1a2e", fontfamily="sans-serif")

# two columns of indicators
left_items  = ["1yr rate", "6M rate", "CPI", "10yr rate",
               "Share Price", "CSI", "OECD CLI", "GDP per capita"]
right_items = ["3M rate", "INDPRO", "Unemployment Rate", "PPI"]

left_text  = "\n".join(left_items)
right_text = "\n".join(right_items)

ax.text(0.17, INPUT_Y + 0.075, left_text,
        ha="center", va="center", fontsize=7.2,
        color="#1a1a2e", fontfamily="sans-serif", linespacing=1.4)
ax.text(0.70, INPUT_Y + 0.075, right_text,
        ha="center", va="center", fontsize=7.2,
        color="#1a1a2e", fontfamily="sans-serif", linespacing=1.4)

# label left/right columns
ax.text(0.17, INPUT_Y + INPUT_H - 0.042, "Prophet indicators",
        ha="center", va="center", fontsize=7, color="#4a4a6a",
        style="italic")
ax.text(0.70, INPUT_Y + INPUT_H - 0.042, "ARIMA indicators",
        ha="center", va="center", fontsize=7, color="#4a4a6a",
        style="italic")

# ── LAYER 2 — STAGE 1 ─────────────────────────────────────────
S1_Y  = 0.55
S1_H  = 0.10

# Prophet box (left half)
box(ax, 0.04, S1_Y, 0.43, S1_H, "PROPHET", "#2a9d8f", fs=8.5, bold=True)
# ARIMA box (right half)
box(ax, 0.53, S1_Y, 0.43, S1_H, "ARIMA",   "#e76f51", fs=8.5, bold=True)

# arrows: input → prophet, input → arima
arrow(ax, 0.25, INPUT_Y, 0.25, S1_Y + S1_H)
arrow(ax, 0.75, INPUT_Y, 0.75, S1_Y + S1_H)

# XGBoost Residual Correction (full width)
XGB_Y = 0.40
XGB_H = 0.09
box(ax, 0.04, XGB_Y, 0.92, XGB_H,
    "XGBOOST RESIDUAL CORRECTION", "#264653", fs=8.5, bold=True)

# arrows: prophet → xgb, arima → xgb
arrow(ax, 0.25, S1_Y, 0.25, XGB_Y + XGB_H)
arrow(ax, 0.75, S1_Y, 0.75, XGB_Y + XGB_H)

# arrow from XGBoost down with label
arrow(ax, 0.50, XGB_Y, 0.50, 0.31)
ax.text(0.53, 0.355, "12 Forecasted\nIndicators →",
        ha="left", va="center", fontsize=7, color="#333333")

# ── LAYER 3 — STAGE 2 ─────────────────────────────────────────
S2_Y  = 0.22
S2_H  = 0.09
BW    = 0.28   # each box width

box(ax, 0.04, S2_Y, BW,  S2_H, "CATBOOST",      "#457b9d", fs=7.8, bold=True)
box(ax, 0.36, S2_Y, BW,  S2_H, "LIGHTGBM",      "#2a9d8f", fs=7.8, bold=True)
box(ax, 0.68, S2_Y, BW,  S2_H, "RANDOM\nFOREST","#e9c46a", fs=7.8, bold=True)

# arrows from xgb to each stage-2 model
arrow(ax, 0.50, 0.40, 0.18, S2_Y + S2_H)
arrow(ax, 0.50, 0.40, 0.50, S2_Y + S2_H)
arrow(ax, 0.50, 0.40, 0.82, S2_Y + S2_H)

# ElasticNet meta-learner
ELAS_Y = 0.10
ELAS_H = 0.08
box(ax, 0.04, ELAS_Y, 0.92, ELAS_H,
    "ELASTICNET META-LEARNER", "#264653", fs=8.5, bold=True)

# arrows: each s2 model → elasticnet
arrow(ax, 0.18, S2_Y, 0.18, ELAS_Y + ELAS_H)
arrow(ax, 0.50, S2_Y, 0.50, ELAS_Y + ELAS_H)
arrow(ax, 0.82, S2_Y, 0.82, ELAS_Y + ELAS_H)

# Recession Probabilities output
REC_Y = 0.005
REC_H = 0.075
box(ax, 0.04, REC_Y, 0.92, REC_H,
    "RECESSION PROBABILITIES", "#e63946", fs=9, bold=True)
arrow(ax, 0.50, ELAS_Y, 0.50, REC_Y + REC_H)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "figure 02.png")
plt.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"  Saved → {out2}")
print("\nDone.")
