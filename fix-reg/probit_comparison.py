"""
probit_comparison.py
Honest probit comparison for the RecessionRadar ICML 2026 paper.

Run from: RecessionRadar/RecessionRadar/
  python fix-reg/probit_comparison.py

Outputs (written to fix-reg/):
  probit_comparison.csv
  figure_probit_vs_shap.png
  capability_matrix.png
"""

import os
import sys
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─────────────────────────────────────────────────────────────────────────────
# Required classes for unpickling FullChainStackingEnsemble
# ─────────────────────────────────────────────────────────────────────────────

def _inv_logit(z):
    return np.clip(1 / (1 + np.exp(-np.clip(z, -50, 50))) * 100, 0, 100)

def _san(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
    return df


class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X)


class FullChainCatBoostModel:
    def __init__(self):
        self.chain_model = self.scaler = None

    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainLightGBMModel:
    def __init__(self):
        self.chain_model = self.scaler = None

    def predict(self, X):
        X = _san(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainRandomForestModel:
    def __init__(self):
        self.chain_model = self.scaler = None

    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainStackingEnsemble:
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {
            'CatBoost': FullChainCatBoostModel,
            'LightGBM': FullChainLightGBMModel,
            'RandomForest': FullChainRandomForestModel,
        }
        self.meta_models = {}
        self.cv_folds = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler = {}
        self.fitted_base_models = {}

    def _eng(self, *bp):
        f = list(bp)
        if self.use_feature_engineering:
            f += [
                np.mean(bp, axis=0),
                0.4 * bp[0] + 0.35 * bp[1] + 0.25 * bp[2],
                np.std(bp, axis=0),
                np.min(bp, axis=0),
                np.max(bp, axis=0),
            ]
            for i in range(len(bp)):
                for j in range(i + 1, len(bp)):
                    f.append(np.abs(bp[i] - bp[j]))
        return np.column_stack(f)

    def predict(self, X):
        bp = {n: m.predict(X) for n, m in self.fitted_base_models.items()}
        fp = np.zeros_like(list(bp.values())[0])
        for i, t in enumerate([
            'recession_probability',
            '1_month_recession_probability',
            '3_month_recession_probability',
            '6_month_recession_probability',
        ]):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf = self._eng(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data", "fix", "feature_selected_reg_full.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "full_chain_stacking.pkl")
SHAP_PATH  = os.path.join(SCRIPT_DIR, "task2_outputs", "shap_matrix_all_features.csv")
OUT_CSV    = os.path.join(SCRIPT_DIR, "probit_comparison.csv")
OUT_FIG1   = os.path.join(SCRIPT_DIR, "figure_probit_vs_shap.png")
OUT_FIG2   = os.path.join(SCRIPT_DIR, "capability_matrix.png")

SPLIT   = "2020-01-01"
TARGETS = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]
LABELS  = ["Current", "1M", "3M", "6M"]
H_STEPS = [1, 1, 3, 6]

PAPER_OURS_MAE = {
    "Current": 6.8292,
    "1M":      5.6336,
    "3M":      7.7285,
    "6M":      10.1696,
}

N_BOOT = 5000
SEED   = 42
EPS    = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)


def logit(y):
    y_s = np.clip(np.array(y, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(y_s / (1 - y_s))


def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-np.array(z, dtype=float))) * 100.0


def dm_test(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano test with HLN 1997 correction.
    d_t = SE_pred1 - SE_pred2; positive DM -> pred1 is worse (pred2 is better).
    """
    d = (actual - pred1) ** 2 - (actual - pred2) ** 2
    d_bar = np.mean(d)
    bw = max(h - 1, 0)
    lrv = np.var(d, ddof=0)
    for k in range(1, bw + 1):
        w = 1 - k / (bw + 1)
        lrv += 2 * w * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
    if lrv <= 0:
        return np.nan, np.nan
    n = len(d)
    dm = d_bar / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p  = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return dm, p


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

X_train = clean(train_df.drop(columns=TARGETS + ["date"]))
X_test  = clean(test_df.drop(columns=TARGETS + ["date"]))
y_train = clean(train_df[TARGETS])
y_test  = test_df[TARGETS].ffill()   # ffill before computing MAE, consistent with paper

print(f"  Train: {len(X_train)} | Test: {len(X_test)} | Features: {X_train.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# Probit baseline (yield-curve spread -> logit -> LinearRegression -> inv_logit)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[Probit] Yield-curve spread baseline...")
spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1, 1)

probit_preds = {}
probit_coef  = {}
for t, lbl in zip(TARGETS, LABELS):
    y_tr_logit = logit(y_train[t].values)
    reg = LinearRegression()
    reg.fit(spread_train, y_tr_logit)
    preds = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)
    mae   = mean_absolute_error(y_test[t].values, preds)
    probit_preds[lbl] = preds
    probit_coef[lbl]  = {"coef": reg.coef_[0], "intercept": reg.intercept_}
    print(f"  {lbl:8s}  coef={reg.coef_[0]:+.4f}  intercept={reg.intercept_:+.4f}  MAE={mae:.4f}")

# The probit uses ONE spread feature; coefficient should be the same across horizons
# (each horizon is fit independently, so coefficients may differ slightly — we show
# the average for the annotation)
mean_coef = np.mean([probit_coef[l]["coef"] for l in LABELS])
print(f"\n  Mean probit coefficient (all horizons): {mean_coef:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Load ensemble and get predictions
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading ensemble model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

print("  Running ensemble predictions on test set...")
ens_preds_arr = ensemble.predict(X_test)   # shape (65, 4)
ens_preds = {lbl: ens_preds_arr[:, i] for i, lbl in enumerate(LABELS)}


# ─────────────────────────────────────────────────────────────────────────────
# Compute per-horizon stats: MAE, Bootstrap CI, DM test, Cohen's d
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing statistics...")
rng = np.random.default_rng(SEED)

rows = []
for (t, lbl, h) in zip(TARGETS, LABELS, H_STEPS):
    actual  = y_test[t].values
    p_probit = probit_preds[lbl]
    p_ours   = ens_preds[lbl]

    mae_probit = mean_absolute_error(actual, p_probit)
    mae_ours   = mean_absolute_error(actual, p_ours)
    mae_diff   = mae_probit - mae_ours   # positive = probit is worse (higher MAE) = we're better

    # Bootstrap CI on mae_diff (N_boot=5000, seed=42)
    n = len(actual)
    boot_diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_diffs[b] = (
            mean_absolute_error(actual[idx], p_probit[idx])
            - mean_absolute_error(actual[idx], p_ours[idx])
        )
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # DM test: d_t = SE_probit - SE_ours; positive DM -> probit worse -> our model better
    dm_stat, p_val = dm_test(actual, p_probit, p_ours, h=h)

    # Cohen's d on absolute error differentials
    # d_t = |e_probit| - |e_ours|; positive = probit has higher error (we're better)
    d_t    = np.abs(actual - p_probit) - np.abs(actual - p_ours)
    cohens = float(np.mean(d_t) / np.std(d_t, ddof=0)) if np.std(d_t, ddof=0) > 0 else np.nan

    print(
        f"  {lbl:8s}  Probit MAE={mae_probit:.4f}  Ours MAE={mae_ours:.4f}  "
        f"diff={mae_diff:+.4f}  CI=[{ci_lower:.4f},{ci_upper:.4f}]  "
        f"DM={dm_stat:.4f}  p={p_val:.4f}  d={cohens:.4f}"
    )

    rows.append({
        "Horizon":            lbl,
        "Probit_MAE":         round(mae_probit, 4),
        "Ours_MAE":           round(mae_ours, 4),
        "MAE_diff":           round(mae_diff, 4),
        "Bootstrap_CI_lower": round(ci_lower, 4),
        "Bootstrap_CI_upper": round(ci_upper, 4),
        "DM_stat":            round(dm_stat, 4),
        "p_value":            round(p_val, 4),
        "Cohens_d":           round(cohens, 4),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_CSV, index=False)
print(f"\n[Output 1] Saved: {OUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Output 2: figure_probit_vs_shap.png
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding figure_probit_vs_shap.png ...")

shap_df = pd.read_csv(SHAP_PATH)
shap_df = shap_df.set_index("Feature")
shap_df["mean_shap"] = shap_df[["Current", "1M", "3M", "6M"]].mean(axis=1)
top10 = shap_df.nlargest(10, "mean_shap")
top10_vals = top10[["Current", "1M", "3M", "6M"]].values  # (10, 4)

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
fig.suptitle(
    "Probit vs. Ensemble: Feature Attribution Comparison",
    fontsize=13, fontweight="bold", y=1.01
)

# ── Left panel: Probit "feature importance" ──────────────────────────────────
# Each horizon gets a horizontal bar with the probit coefficient for that horizon
bar_labels  = [f"Yield Curve Spread\n({lbl})" for lbl in LABELS]
bar_values  = [abs(probit_coef[lbl]["coef"]) for lbl in LABELS]
bar_colors  = ["#9e9e9e"] * 4
bar_y       = np.arange(len(LABELS))

ax_left.barh(bar_y, bar_values, color=bar_colors, edgecolor="white", height=0.5)
ax_left.set_yticks(bar_y)
ax_left.set_yticklabels(bar_labels, fontsize=9)
ax_left.set_xlabel("|β| (logit coefficient)", fontsize=9)
ax_left.set_title("Probit (YC): Single Feature,\nNo Horizon Differentiation", fontsize=10, fontweight="bold")
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)
ax_left.invert_yaxis()

# Annotate with the mean coefficient value
ax_left.text(
    0.98, 0.04,
    f"Same feature at all horizons\nβ ≈ {mean_coef:.4f} (mean across horizons)",
    ha="right", va="bottom",
    transform=ax_left.transAxes,
    fontsize=8, color="#555",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#cccccc")
)

# ── Right panel: SHAP heatmap ─────────────────────────────────────────────────
feat_names = [name[:30] for name in top10.index.tolist()]  # truncate long names
horizons   = ["Current", "1M", "3M", "6M"]

im = ax_right.imshow(top10_vals, aspect="auto", cmap="YlOrRd", interpolation="nearest")
ax_right.set_xticks(np.arange(4))
ax_right.set_xticklabels(horizons, fontsize=9)
ax_right.set_yticks(np.arange(10))
ax_right.set_yticklabels(feat_names, fontsize=8)
ax_right.set_title(
    "Our Model: Horizon-Conditional\nFeature Attribution (SHAP)",
    fontsize=10, fontweight="bold"
)

# Annotate cells with values
for i in range(10):
    for j in range(4):
        val = top10_vals[i, j]
        ax_right.text(
            j, i, f"{val:.2f}",
            ha="center", va="center",
            fontsize=7,
            color="black" if val < top10_vals.max() * 0.7 else "white"
        )

plt.colorbar(im, ax=ax_right, shrink=0.8, label="Mean |SHAP value|")

plt.tight_layout()
fig.savefig(OUT_FIG1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Output 2] Saved: {OUT_FIG1}")


# ─────────────────────────────────────────────────────────────────────────────
# Output 3: capability_matrix.png
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding capability_matrix.png ...")

# Table content
# Each cell: (label, justification text)
capabilities = [
    "Uncertainty Intervals",
    "Horizon-Conditional XAI",
    "Scenario Simulation",
    "Multi-Signal Integration",
    "Online Recalibration",
]

# (label, sub-text)
probit_cells = [
    ("No",      "No distributional\noutput; point estimate only"),
    ("No",      "Single fixed feature;\nno per-horizon attribution"),
    ("No",      "Cannot substitute\nindicator paths"),
    ("No",      "1 feature: 10yr–3mo\nspread only"),
    ("Partial", "Refit needed;\nno adaptive mechanism"),
]
xgb_cells = [
    ("No",      "Point estimates only;\nno conformal coverage"),
    ("Partial", "Feature importance only;\nno horizon conditioning"),
    ("No",      "Black-box: cannot\nsubstitute indicator paths"),
    ("Yes",     "All 46 features\nused jointly"),
    ("Partial", "Refit needed;\nno adaptive α"),
]
ours_cells = [
    ("Yes",     "ACI conformal bands;\nadaptive coverage guarantee"),
    ("Yes",     "SHAP per horizon;\nconditioned on chain outputs"),
    ("Yes",     "Substitute indicator paths\ninto Stage 2 directly"),
    ("Yes",     "46 features: macro,\nfinancial, sentiment"),
    ("Yes",     "ACI adaptive α;\nno refit required"),
]

COLOR_MAP = {
    "Yes":     "#c8e6c9",
    "No":      "#ffcdd2",
    "Partial": "#fff9c4",
}
TEXT_COLOR = {
    "Yes":     "#1b5e20",
    "No":      "#b71c1c",
    "Partial": "#f57f17",
}

n_rows = len(capabilities)
n_cols = 4  # Capability | Probit | Single-Stage XGB | Ours

col_headers = ["Capability", "Probit\n(Yield Curve)", "Single-Stage XGB", "Two-Stage Ensemble\n(Ours)"]
col_w = [0.28, 0.20, 0.20, 0.27]  # fractional widths (sum ≈ 0.95)

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor("white")
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

fig.suptitle(
    "Capability Audit: Probit vs. Single-Stage XGB vs. Two-Stage Ensemble",
    fontsize=12, fontweight="bold", y=0.97
)

# Layout
left_margin = 0.02
top          = 0.88
header_h     = 0.09
row_h        = (top - header_h - 0.03) / n_rows
col_starts   = [left_margin]
for w in col_w[:-1]:
    col_starts.append(col_starts[-1] + w)
col_starts_arr = np.array(col_starts)
col_w_arr      = np.array(col_w)

# Header row
header_color = "#1a1a2e"
for j, (hdr, cx, cw) in enumerate(zip(col_headers, col_starts_arr, col_w_arr)):
    rect = mpatches.FancyBboxPatch(
        (cx + 0.005, top - header_h + 0.005),
        cw - 0.01, header_h - 0.01,
        boxstyle="round,pad=0.005",
        facecolor=header_color, edgecolor="white", linewidth=1.5,
        transform=ax.transAxes, zorder=2
    )
    ax.add_patch(rect)
    ax.text(
        cx + cw / 2, top - header_h / 2,
        hdr, ha="center", va="center",
        fontsize=9, fontweight="bold", color="white",
        transform=ax.transAxes, zorder=3
    )

# Data rows
all_row_cells = [
    [capabilities[i], probit_cells[i], xgb_cells[i], ours_cells[i]]
    for i in range(n_rows)
]

for i, row_data in enumerate(all_row_cells):
    y_top = top - header_h - i * row_h
    bg = "#f9f9f9" if i % 2 == 0 else "white"

    for j, cell in enumerate(row_data):
        cx = col_starts_arr[j]
        cw = col_w_arr[j]
        cy = y_top - row_h

        if j == 0:
            # Capability label column
            rect = mpatches.FancyBboxPatch(
                (cx + 0.005, cy + 0.005), cw - 0.01, row_h - 0.01,
                boxstyle="round,pad=0.005",
                facecolor=bg, edgecolor="#e0e0e0", linewidth=0.8,
                transform=ax.transAxes, zorder=2
            )
            ax.add_patch(rect)
            ax.text(
                cx + cw / 2, cy + row_h / 2,
                cell, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#212121",
                transform=ax.transAxes, zorder=3
            )
        else:
            label, justification = cell
            cell_color = COLOR_MAP[label]
            txt_color  = TEXT_COLOR[label]

            rect = mpatches.FancyBboxPatch(
                (cx + 0.005, cy + 0.005), cw - 0.01, row_h - 0.01,
                boxstyle="round,pad=0.005",
                facecolor=cell_color, edgecolor="#cccccc", linewidth=0.8,
                transform=ax.transAxes, zorder=2
            )
            ax.add_patch(rect)

            # Main label (Yes/No/Partial)
            ax.text(
                cx + cw / 2, cy + row_h * 0.65,
                label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=txt_color,
                transform=ax.transAxes, zorder=3
            )
            # Sub-text justification
            ax.text(
                cx + cw / 2, cy + row_h * 0.25,
                justification, ha="center", va="center",
                fontsize=6.5, color="#444444",
                transform=ax.transAxes, zorder=3,
                linespacing=1.3
            )

# Legend
legend_y  = 0.03
legend_x  = 0.30
for label, color, tc in [
    ("Yes — fully supported",     "#c8e6c9", "#1b5e20"),
    ("Partial — limited support", "#fff9c4", "#f57f17"),
    ("No — not supported",        "#ffcdd2", "#b71c1c"),
]:
    rect = mpatches.FancyBboxPatch(
        (legend_x, legend_y - 0.015), 0.015, 0.025,
        boxstyle="round,pad=0.002",
        facecolor=color, edgecolor="#cccccc", linewidth=0.8,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    ax.text(legend_x + 0.02, legend_y - 0.002, label, fontsize=7.5, color="#333",
            transform=ax.transAxes, va="center")
    legend_x += 0.23

fig.savefig(OUT_FIG2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[Output 3] Saved: {OUT_FIG2}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY — Probit vs. Ensemble")
print("=" * 70)
print(f"{'Horizon':>8}  {'Probit MAE':>10}  {'Ours MAE':>10}  {'MAE_diff':>9}  {'DM stat':>8}  {'p-value':>8}")
print("-" * 70)
for r in rows:
    print(
        f"  {r['Horizon']:>6}  {r['Probit_MAE']:>10.4f}  {r['Ours_MAE']:>10.4f}  "
        f"{r['MAE_diff']:>+9.4f}  {r['DM_stat']:>8.4f}  {r['p_value']:>8.4f}"
    )
print("=" * 70)
print(
    "\nNote: MAE_diff = Probit_MAE - Ours_MAE\n"
    "  Negative = probit is better (lower MAE)\n"
    "  Positive = our model is better\n"
    "\nDM_stat > 0 means probit has higher squared errors (our model better).\n"
)

# Confirm output files
print("\nOutput files:")
for path in [OUT_CSV, OUT_FIG1, OUT_FIG2]:
    exists = os.path.isfile(path)
    print(f"  {'[OK]' if exists else '[MISSING]'}  {path}")

print("\nDone.")
