"""
Yield Curve Magnitude Test
--------------------------
Splits the 65 post-2020 test observations into:
  - LOUD:      |10Y - 3M spread| > 100 bps  (strong signal regime)
  - AMBIGUOUS: |10Y - 3M spread| <= 100 bps (weak signal regime)

Then compares MAE of our Two-Stage Ensemble vs Probit baseline in each bin.
Expected: Probit wins in LOUD months (yield curve dominates).
          Our framework wins (or narrows gap) in AMBIGUOUS months.
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# REQUIRED CLASS DEFINITIONS (to unpickle the saved ensemble)
# These must match exactly what was used in ensemble.ipynb
# ─────────────────────────────────────────────────────────────

recession_targets = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]

epsilon = 1e-8

def safe_logit_transform(y):
    y_clipped = np.clip(y, 0, 100)
    y_scaled = np.clip(y_clipped / 100.0, epsilon, 1 - epsilon)
    return np.log(y_scaled / (1 - y_scaled))

def safe_inv_logit_transform(y_logit):
    y_logit_clipped = np.clip(y_logit, -50, 50)
    return np.clip(1 / (1 + np.exp(-y_logit_clipped)) * 100, 0, 100)

def sanitize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {
            "objective": "regression", "metric": "rmse", "max_depth": 8,
            "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 30, "reg_alpha": 0.3, "reg_lambda": 0.3,
            "seed": 42, "verbose": -1,
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y):
        val_size = max(10, int(0.1 * len(X)))
        dtrain = lgb.Dataset(X[:-val_size], label=y[:-val_size])
        dval   = lgb.Dataset(X[-val_size:], label=y[-val_size:], reference=dtrain)
        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

class FullChainCatBoostModel:
    def __init__(self):
        self.chain_model = None
        self.scaler = None

    def fit(self, X_train, y_train):
        self.scaler = RobustScaler()
        Xs = pd.DataFrame(self.scaler.fit_transform(X_train),
                          columns=X_train.columns, index=X_train.index)
        yt = safe_logit_transform(y_train[recession_targets].values)
        base = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6,
                                  l2_leaf_reg=3, subsample=0.8, random_seed=42,
                                  loss_function="RMSE", verbose=False)
        self.chain_model = RegressorChain(base, order=[0, 1, 2, 3])
        self.chain_model.fit(Xs, yt)

    def predict(self, X_test):
        Xs = pd.DataFrame(self.scaler.transform(X_test),
                          columns=X_test.columns, index=X_test.index)
        return np.clip(safe_inv_logit_transform(self.chain_model.predict(Xs)), 0, 100)

class FullChainLightGBMModel:
    def __init__(self):
        self.chain_model = None
        self.scaler = None
        self.lgb_params = {
            "objective": "regression", "metric": "rmse", "max_depth": 8,
            "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 30, "reg_alpha": 0.3, "reg_lambda": 0.3,
            "seed": 42, "verbose": -1,
        }

    def fit(self, X_train, y_train):
        X_train = sanitize_columns(X_train)
        self.scaler = RobustScaler()
        Xs = pd.DataFrame(self.scaler.fit_transform(X_train),
                          columns=X_train.columns, index=X_train.index)
        yt = safe_logit_transform(y_train[recession_targets].values)
        self.chain_model = RegressorChain(
            LGBMWrapper(params=self.lgb_params, num_boost_round=500),
            order=[0, 1, 2, 3]
        )
        self.chain_model.fit(Xs, yt)

    def predict(self, X_test):
        X_test = sanitize_columns(X_test)
        Xs = pd.DataFrame(self.scaler.transform(X_test),
                          columns=X_test.columns, index=X_test.index)
        return np.clip(safe_inv_logit_transform(self.chain_model.predict(Xs)), 0, 100)

class FullChainRandomForestModel:
    def __init__(self):
        self.chain_model = None
        self.scaler = None

    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        Xs = pd.DataFrame(self.scaler.fit_transform(X_train),
                          columns=X_train.columns, index=X_train.index)
        yt = safe_logit_transform(y_train[recession_targets].values)
        rf = RandomForestRegressor(n_estimators=500, max_depth=12,
                                   min_samples_split=10, min_samples_leaf=5,
                                   max_features=0.8, max_samples=0.8,
                                   random_state=42, n_jobs=-1)
        self.chain_model = RegressorChain(base_estimator=rf, order=[0, 1, 2, 3])
        self.chain_model.fit(Xs, yt)

    def predict(self, X_test):
        Xs = pd.DataFrame(self.scaler.transform(X_test),
                          columns=X_test.columns, index=X_test.index)
        return np.clip(safe_inv_logit_transform(self.chain_model.predict(Xs)), 0, 100)

class FullChainStackingEnsemble:
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {
            'CatBoost':     FullChainCatBoostModel,
            'LightGBM':     FullChainLightGBMModel,
            'RandomForest': FullChainRandomForestModel,
        }
        self.meta_models      = {}
        self.cv_folds         = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler      = {}
        self.fitted_base_models = {}

    def _engineer_meta_features(self, *base_preds):
        features = list(base_preds)
        if self.use_feature_engineering:
            features.append(np.mean(base_preds, axis=0))
            features.append(0.4*base_preds[0] + 0.35*base_preds[1] + 0.25*base_preds[2])
            features.append(np.std(base_preds, axis=0))
            features.append(np.min(base_preds, axis=0))
            features.append(np.max(base_preds, axis=0))
            for i in range(len(base_preds)):
                for j in range(i+1, len(base_preds)):
                    features.append(np.abs(base_preds[i] - base_preds[j]))
        return np.column_stack(features)

    def predict(self, X_test):
        base_predictions = {name: model.predict(X_test)
                            for name, model in self.fitted_base_models.items()}
        final_predictions = np.zeros_like(list(base_predictions.values())[0])
        for i, target in enumerate(recession_targets):
            base_preds_for_target = [base_predictions[name][:, i]
                                     for name in self.base_models.keys()]
            meta_features = self._engineer_meta_features(*base_preds_for_target)
            meta_features_scaled = self.meta_scaler[target].transform(meta_features)
            final_predictions[:, i] = self.meta_models[target].predict(meta_features_scaled)
        return np.clip(final_predictions, 0, 100)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
DATA_PATH  = "../data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "models/full_chain_stacking.pkl"
PDF_PATH   = "yield_curve_test_report.pdf"
SPLIT      = "2020-01-01"
LABELS     = ["Current", "1-Month", "3-Month", "6-Month"]
LOUD_THRESHOLD = 1.0   # percentage points (100 bps)

COLORS = {
    "Ours (Two-Stage)":     "#2ca02c",
    "Probit (Yield Curve)": "#ff7f0e",
}

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

def clean(d):
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.ffill().bfill().fillna(0)

X_train = clean(train_df.drop(columns=recession_targets + ["date"]))
X_test  = clean(test_df.drop(columns=recession_targets + ["date"]))
y_train = clean(train_df[recession_targets])
y_test  = clean(test_df[recession_targets])

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# Load saved ensemble → per-observation predictions
# ─────────────────────────────────────────────────────────────
print("Loading saved ensemble model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

ours_preds = ensemble.predict(X_test)   # shape (65, 4)
ours_df = pd.DataFrame(ours_preds, columns=recession_targets, index=X_test.index)
print(f"  Ensemble predictions shape: {ours_preds.shape}")

# Sanity check — overall MAE should match paper numbers
EPS = 1e-6
def logit(y):
    return np.log(np.clip(y/100, EPS, 1-EPS) / (1 - np.clip(y/100, EPS, 1-EPS)))
def inv_logit(z):
    return np.clip(1/(1+np.exp(-z))*100, 0, 100)

for t, lbl in zip(recession_targets, LABELS):
    mae = mean_absolute_error(y_test[t], ours_df[t])
    print(f"  {lbl:10s}  MAE={mae:.4f}  (paper: {[6.83,5.63,7.73,10.17][LABELS.index(lbl)]})")

# ─────────────────────────────────────────────────────────────
# Probit (yield curve) predictions
# ─────────────────────────────────────────────────────────────
print("\nComputing probit predictions...")
spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1, 1)

probit_preds = {}
for t, lbl in zip(recession_targets, LABELS):
    reg = LinearRegression()
    reg.fit(spread_train, logit(y_train[t].values))
    probit_preds[lbl] = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)

# ─────────────────────────────────────────────────────────────
# Yield curve magnitude bins
# ─────────────────────────────────────────────────────────────
print("\nComputing yield curve spread and binning test observations...")
spread_vals = (X_test["10_year_rate"] - X_test["3_months_rate"]).values
abs_spread  = np.abs(spread_vals)

loud_mask      = abs_spread > LOUD_THRESHOLD
ambiguous_mask = ~loud_mask

n_loud      = loud_mask.sum()
n_ambiguous = ambiguous_mask.sum()

print(f"  LOUD (|spread| > {LOUD_THRESHOLD*100:.0f} bps): {n_loud} months")
print(f"  AMBIGUOUS (|spread| ≤ {LOUD_THRESHOLD*100:.0f} bps): {n_ambiguous} months")
print(f"  Spread range: {spread_vals.min()*100:.1f} to {spread_vals.max()*100:.1f} bps")

# ─────────────────────────────────────────────────────────────
# MAE by bin
# ─────────────────────────────────────────────────────────────
print("\nComputing MAE by regime bin...")

results = {}
for t, lbl in zip(recession_targets, LABELS):
    actual        = y_test[t].values
    ours_pred     = ours_df[t].values
    probit_pred   = probit_preds[lbl]

    mae_ours_loud      = mean_absolute_error(actual[loud_mask],      ours_pred[loud_mask])      if n_loud > 0      else np.nan
    mae_ours_ambig     = mean_absolute_error(actual[ambiguous_mask],  ours_pred[ambiguous_mask]) if n_ambiguous > 0 else np.nan
    mae_probit_loud    = mean_absolute_error(actual[loud_mask],      probit_pred[loud_mask])    if n_loud > 0      else np.nan
    mae_probit_ambig   = mean_absolute_error(actual[ambiguous_mask],  probit_pred[ambiguous_mask]) if n_ambiguous > 0 else np.nan

    results[lbl] = {
        "mae_ours_loud":    mae_ours_loud,
        "mae_ours_ambig":   mae_ours_ambig,
        "mae_probit_loud":  mae_probit_loud,
        "mae_probit_ambig": mae_probit_ambig,
        "winner_loud":    "Probit" if mae_probit_loud  < mae_ours_loud  else "Ours",
        "winner_ambig":   "Probit" if mae_probit_ambig < mae_ours_ambig else "Ours",
        "actual":         actual,
        "ours_pred":      ours_pred,
        "probit_pred":    probit_pred,
    }

# ─────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 78)
print(f"{'':30} {'LOUD (>100bps)':^22} {'AMBIGUOUS (≤100bps)':^22}")
print(f"{'Target':30} {'Ours MAE':>10} {'Probit MAE':>10} {'Winner':>10}  {'Ours MAE':>8} {'Probit MAE':>10} {'Winner':>8}")
print("-" * 78)
for lbl in LABELS:
    r = results[lbl]
    print(
        f"{lbl:30} {r['mae_ours_loud']:>10.2f} {r['mae_probit_loud']:>10.2f} {r['winner_loud']:>10}"
        f"  {r['mae_ours_ambig']:>8.2f} {r['mae_probit_ambig']:>10.2f} {r['winner_ambig']:>8}"
    )
print("=" * 78)

# ─────────────────────────────────────────────────────────────
# Generate PDF
# ─────────────────────────────────────────────────────────────
print(f"\nGenerating PDF: {PDF_PATH} ...")
dates      = test_df["date"].values
dates_loud = dates[loud_mask]
dates_ambig = dates[ambiguous_mask]

with PdfPages(PDF_PATH) as pdf:

    # ── PAGE 1: Title & Setup ────────────────────────────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("#f8f9fa")
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()

    hdr = mpatches.FancyBboxPatch((0, 0.88), 1, 0.12, boxstyle="square,pad=0",
        facecolor="#1a1a2e", transform=ax.transAxes)
    ax.add_patch(hdr)
    ax.text(0.5, 0.94, "Yield Curve Magnitude Test", ha="center", va="center",
            fontsize=20, fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.5, 0.895, "Does the probit advantage depend on yield curve signal strength?",
            ha="center", va="center", fontsize=11, color="#aaa", transform=ax.transAxes)

    ax.text(0.5, 0.84, "Hypothesis", ha="center", fontsize=13, fontweight="bold",
            color="#1a1a2e", transform=ax.transAxes)

    hyp_text = (
        "If the probit's advantage is regime-specific — driven by the unusually loud yield curve\n"
        "signal of the 2022–2023 Federal Reserve tightening cycle — then:\n\n"
        "   • In LOUD months (|spread| > 100 bps): Probit wins — signal dominates\n"
        "   • In AMBIGUOUS months (|spread| ≤ 100 bps): Our framework wins — multi-signal matters\n\n"
        "If our framework loses in AMBIGUOUS months too, that is also a valid finding:\n"
        "it suggests the multi-indicator complexity hurts in low-signal environments."
    )
    ax.text(0.5, 0.73, hyp_text, ha="center", va="top", fontsize=10.5,
            color="#333", transform=ax.transAxes, linespacing=1.7)

    # Stats boxes
    box_data = [
        (0.08, f"{n_loud}\nLOUD months\n(|spread| > 100 bps)", "#c62828"),
        (0.38, f"{n_ambiguous}\nAMBIGUOUS months\n(|spread| ≤ 100 bps)", "#1565c0"),
        (0.68, f"{n_loud/65*100:.0f}%\nof test set is\nin LOUD regime", "#4a148c"),
    ]
    for bx, txt, col in box_data:
        rect = mpatches.FancyBboxPatch((bx, 0.30), 0.25, 0.18,
            boxstyle="round,pad=0.01", facecolor=col, alpha=0.12,
            edgecolor=col, linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(rect)
        lines = txt.split("\n")
        ax.text(bx+0.125, 0.42, lines[0], ha="center", va="center",
                fontsize=18, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(bx+0.125, 0.355, "\n".join(lines[1:]), ha="center", va="center",
                fontsize=9, color=col, transform=ax.transAxes, linespacing=1.4)

    spread_range_txt = (
        f"Test spread range: {spread_vals.min()*100:.0f} bps to {spread_vals.max()*100:.0f} bps  "
        f"| Mean |spread|: {abs_spread.mean()*100:.0f} bps  "
        f"| Threshold: ±100 bps"
    )
    ax.text(0.5, 0.23, spread_range_txt, ha="center", fontsize=9,
            color="#555", transform=ax.transAxes, style="italic")

    ax.text(0.5, 0.05,
            "Test set: Jan 2020 – May 2025 (65 observations) | Spread = 10Y Treasury − 3M Treasury",
            ha="center", fontsize=8.5, color="#888", transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 2: Spread time-series with regime shading ───────
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 8.5),
                                          gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("white")
    fig.suptitle("10Y–3M Yield Curve Spread — Test Period with Regime Labels",
                 fontsize=14, fontweight="bold", color="#1a1a2e")

    # Spread over time
    ax_top.plot(dates, spread_vals * 100, color="#1a1a2e", linewidth=1.8, label="10Y−3M spread (bps)")
    ax_top.axhline(y=100,  color="#c62828", linestyle="--", linewidth=1, alpha=0.7, label="+100 bps threshold")
    ax_top.axhline(y=-100, color="#c62828", linestyle="--", linewidth=1, alpha=0.7, label="−100 bps threshold")
    ax_top.axhline(y=0,    color="#888",    linestyle=":",  linewidth=0.8)
    ax_top.fill_between(dates, spread_vals*100, 0,
                         where=loud_mask, alpha=0.15, color="#c62828", label="LOUD regime")
    ax_top.fill_between(dates, spread_vals*100, 0,
                         where=ambiguous_mask, alpha=0.15, color="#1565c0", label="AMBIGUOUS regime")
    ax_top.set_ylabel("Spread (bps)", fontsize=10)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Bin indicator
    regime_vals = np.where(loud_mask, 1, 0)
    ax_bot.bar(dates, np.ones(len(dates)), color=["#c62828" if l else "#1565c0" for l in loud_mask],
               alpha=0.7, width=20)
    ax_bot.set_yticks([]); ax_bot.set_ylabel("Regime", fontsize=9)
    ax_bot.set_ylim(0, 1.2)
    loud_patch = mpatches.Patch(color="#c62828", alpha=0.7, label=f"LOUD ({n_loud} months)")
    ambig_patch = mpatches.Patch(color="#1565c0", alpha=0.7, label=f"AMBIGUOUS ({n_ambiguous} months)")
    ax_bot.legend(handles=[loud_patch, ambig_patch], fontsize=9, loc="upper right")
    ax_bot.spines["top"].set_visible(False); ax_bot.spines["right"].set_visible(False)
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 3: MAE results table ────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    ax.text(0.5, 0.97, "MAE by Regime — Full Results Table", ha="center", va="top",
            fontsize=17, fontweight="bold", color="#1a1a2e", transform=ax.transAxes)
    ax.text(0.5, 0.93,
            f"LOUD: |spread| > 100 bps ({n_loud} months)   |   "
            f"AMBIGUOUS: |spread| ≤ 100 bps ({n_ambiguous} months)",
            ha="center", va="top", fontsize=10.5, color="#555",
            transform=ax.transAxes, style="italic")

    col_labels = [
        "Target",
        f"Ours\nLOUD", f"Probit\nLOUD", "Winner\nLOUD",
        f"Ours\nAMBIG", f"Probit\nAMBIG", "Winner\nAMBIG",
    ]
    table_data = []
    for lbl in LABELS:
        r = results[lbl]
        w_loud  = "✓ Ours"  if r["winner_loud"]  == "Ours" else "✓ Probit"
        w_ambig = "✓ Ours"  if r["winner_ambig"] == "Ours" else "✓ Probit"
        table_data.append([
            lbl,
            f"{r['mae_ours_loud']:.2f}",   f"{r['mae_probit_loud']:.2f}",   w_loud,
            f"{r['mae_ours_ambig']:.2f}",  f"{r['mae_probit_ambig']:.2f}",  w_ambig,
        ])

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.3, 3.2)

    hdr_cols   = ["#1a1a2e", "#c62828", "#c62828", "#c62828", "#1565c0", "#1565c0", "#1565c0"]
    for j, col in enumerate(hdr_cols):
        tbl[(0, j)].set_facecolor(col)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(LABELS)+1):
        r = results[LABELS[i-1]]
        # LOUD columns
        for j in [1, 2]:
            tbl[(i, j)].set_facecolor("#fff5f5")
        # Winner LOUD
        tbl[(i, 3)].set_facecolor("#fff5f5")
        if r["winner_loud"] == "Ours":
            tbl[(i, 3)].set_text_props(color="#2ca02c", fontweight="bold")
            tbl[(i, 1)].set_text_props(color="#2ca02c", fontweight="bold")
        else:
            tbl[(i, 3)].set_text_props(color="#ff7f0e", fontweight="bold")
            tbl[(i, 2)].set_text_props(color="#ff7f0e", fontweight="bold")
        # AMBIG columns
        for j in [4, 5]:
            tbl[(i, j)].set_facecolor("#f0f4ff")
        tbl[(i, 6)].set_facecolor("#f0f4ff")
        if r["winner_ambig"] == "Ours":
            tbl[(i, 6)].set_text_props(color="#2ca02c", fontweight="bold")
            tbl[(i, 4)].set_text_props(color="#2ca02c", fontweight="bold")
        else:
            tbl[(i, 6)].set_text_props(color="#ff7f0e", fontweight="bold")
            tbl[(i, 5)].set_text_props(color="#ff7f0e", fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 4: Bar chart grouped by regime ──────────────────
    fig, axes_arr = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("MAE by Regime — Grouped Bar Charts", fontsize=14,
                 fontweight="bold", color="#1a1a2e", y=0.98)

    bar_w = 0.3
    x = np.array([0, 1])
    tick_labels = ["LOUD\n(>100bps)", "AMBIGUOUS\n(≤100bps)"]

    for ax, lbl in zip(axes_arr.flat, LABELS):
        r = results[lbl]
        ours_vals   = [r["mae_ours_loud"],   r["mae_ours_ambig"]]
        probit_vals = [r["mae_probit_loud"], r["mae_probit_ambig"]]

        b1 = ax.bar(x - bar_w/2, ours_vals,   bar_w, color=COLORS["Ours (Two-Stage)"],
                    label="Ours", edgecolor="white", linewidth=1)
        b2 = ax.bar(x + bar_w/2, probit_vals, bar_w, color=COLORS["Probit (Yield Curve)"],
                    label="Probit", edgecolor="white", linewidth=1)

        for bar, v in [(b, val) for b, val in zip(list(b1)+list(b2),
                                                    ours_vals+probit_vals)]:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_title(f"{lbl}", fontsize=11, fontweight="bold", color="#1a1a2e")
        ax.set_xticks(x); ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_ylabel("MAE (pp)", fontsize=9)
        ax.set_ylim(0, max(ours_vals+probit_vals)*1.3)
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 5: Prediction traces by regime ──────────────────
    for regime_name, mask, regime_dates, col in [
        ("LOUD Months (|spread| > 100 bps)", loud_mask,      dates_loud,  "#c62828"),
        ("AMBIGUOUS Months (|spread| ≤ 100 bps)", ambiguous_mask, dates_ambig, "#1565c0"),
    ]:
        if mask.sum() == 0:
            continue
        fig, axes_arr = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        fig.suptitle(f"Predictions vs Actual — {regime_name}",
                     fontsize=13, fontweight="bold", color="#1a1a2e", y=0.99)

        for ax, t, lbl in zip(axes_arr.flat, recession_targets, LABELS):
            r = results[lbl]
            actual      = r["actual"][mask]
            ours_pred   = r["ours_pred"][mask]
            probit_pred = r["probit_pred"][mask]
            mae_o = mean_absolute_error(actual, ours_pred)
            mae_p = mean_absolute_error(actual, probit_pred)

            ax.plot(regime_dates, actual,      color="#2c3e50", lw=2,   label="Actual", zorder=5)
            ax.plot(regime_dates, ours_pred,   color=COLORS["Ours (Two-Stage)"],
                    lw=1.5, linestyle="-",  label=f"Ours (MAE={mae_o:.2f})")
            ax.plot(regime_dates, probit_pred, color=COLORS["Probit (Yield Curve)"],
                    lw=1.5, linestyle="--", label=f"Probit (MAE={mae_p:.2f})")

            winner = "Ours" if mae_o < mae_p else "Probit"
            win_col = COLORS["Ours (Two-Stage)"] if winner == "Ours" else COLORS["Probit (Yield Curve)"]
            ax.set_title(f"{lbl}  —  Winner: {winner}", fontsize=10,
                         fontweight="bold", color=win_col)
            ax.set_ylabel("Recession Prob (%)", fontsize=8)
            ax.legend(fontsize=7); ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    # ── PAGE 7 (or last): Interpretation & paper text ────────
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_axis_off()

    hdr = mpatches.FancyBboxPatch((0,0.91),1,0.09, boxstyle="square,pad=0",
        facecolor="#1a1a2e", transform=ax.transAxes)
    ax.add_patch(hdr)
    ax.text(0.5, 0.955, "Interpretation & Paper-Ready Text",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="white", transform=ax.transAxes)

    # Determine what the results actually show
    ours_wins_ambig = sum(1 for lbl in LABELS if results[lbl]["winner_ambig"] == "Ours")
    probit_wins_loud = sum(1 for lbl in LABELS if results[lbl]["winner_loud"] == "Probit")

    if ours_wins_ambig >= 3 and probit_wins_loud >= 3:
        finding = "HYPOTHESIS CONFIRMED"
        finding_col = "#2e7d32"
        finding_detail = (
            f"Probit wins in {probit_wins_loud}/4 targets during LOUD months.\n"
            f"Our framework wins in {ours_wins_ambig}/4 targets during AMBIGUOUS months.\n"
            "The probit advantage is regime-specific — it disappears when the yield curve signal is weak."
        )
    elif ours_wins_ambig <= 1:
        finding = "HYPOTHESIS NOT CONFIRMED — valid research finding"
        finding_col = "#e65100"
        finding_detail = (
            f"Our framework loses in AMBIGUOUS months too ({4-ours_wins_ambig}/4 targets).\n"
            "This suggests multi-indicator complexity does not help — and may hurt — in low-signal environments.\n"
            "This is an honest, interesting finding worth reporting directly."
        )
    else:
        finding = "MIXED RESULT"
        finding_col = "#1565c0"
        finding_detail = (
            f"Our framework wins {ours_wins_ambig}/4 targets in AMBIGUOUS months.\n"
            f"Probit wins {probit_wins_loud}/4 targets in LOUD months.\n"
            "Partial confirmation — the regime effect exists but is not uniform across all horizons."
        )

    pill = mpatches.FancyBboxPatch((0.1, 0.81), 0.8, 0.07,
        boxstyle="round,pad=0.01", facecolor=finding_col, alpha=0.12,
        edgecolor=finding_col, linewidth=2, transform=ax.transAxes)
    ax.add_patch(pill)
    ax.text(0.5, 0.845, finding, ha="center", va="center", fontsize=13,
            fontweight="bold", color=finding_col, transform=ax.transAxes)
    ax.text(0.5, 0.785, finding_detail, ha="center", va="top", fontsize=9.5,
            color="#333", transform=ax.transAxes, linespacing=1.6)

    paper_text_loud  = "\n".join([f"  {lbl}: Ours={results[lbl]['mae_ours_loud']:.2f}, Probit={results[lbl]['mae_probit_loud']:.2f}" for lbl in LABELS])
    paper_text_ambig = "\n".join([f"  {lbl}: Ours={results[lbl]['mae_ours_ambig']:.2f}, Probit={results[lbl]['mae_probit_ambig']:.2f}" for lbl in LABELS])

    sections = [
        ("#c62828", f"LOUD months ({n_loud} obs, |spread|>100bps):", paper_text_loud),
        ("#1565c0", f"AMBIGUOUS months ({n_ambiguous} obs, |spread|≤100bps):", paper_text_ambig),
    ]
    y = 0.68
    for col, title, body in sections:
        ax.text(0.04, y, title, fontsize=10, fontweight="bold",
                color=col, transform=ax.transAxes)
        y -= 0.03
        ax.text(0.06, y, body, fontsize=9, color="#444",
                transform=ax.transAxes, va="top", linespacing=1.5)
        y -= 0.15

    ax.text(0.04, y, "Suggested sentence for Section 4.4 (Limitations):", fontsize=10,
            fontweight="bold", color="#1a1a2e", transform=ax.transAxes)
    y -= 0.04
    paper_sentence = (
        f"\"To test whether the probit's advantage is regime-specific, we partition the 65 test\n"
        f"observations by yield curve signal strength: LOUD months (|10Y–3M spread| > 100 bps,\n"
        f"n={n_loud}) and AMBIGUOUS months (|spread| ≤ 100 bps, n={n_ambiguous}). "
        + ("In LOUD months, the probit\n"
           "achieves lower MAE on all four horizons, consistent with the strong contemporaneous\n"
           "yield curve signal. In AMBIGUOUS months, our framework achieves lower MAE on "
           f"{ours_wins_ambig}/4 horizons,\n"
           "supporting the argument that multi-signal integration provides value when no single\n"
           "indicator dominates.\""
           if ours_wins_ambig >= 2 else
           "Even in AMBIGUOUS\nmonths, the probit remains competitive, suggesting the multi-signal complexity\n"
           "does not provide a measurable advantage in this test window. We report this finding\n"
           "directly as a limitation and direction for future work with extended test periods.\"")
    )
    ax.text(0.04, y, paper_sentence, fontsize=8.5, color="#1565c0",
            transform=ax.transAxes, va="top", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#1565c0", alpha=0.4))

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    d = pdf.infodict()
    d["Title"] = "Yield Curve Magnitude Test — RecessionRadar"

print(f"\n✓  PDF saved: {os.path.abspath(PDF_PATH)}")
print("Done.")
