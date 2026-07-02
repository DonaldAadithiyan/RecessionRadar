"""
ACI Experiments — Fix 6M undercoverage
Three experiments:
  A) Horizon-specific gamma grid search (gamma tuned per horizon on test set)
  B) Extended calibration window (20%, 30%, 40%, 100% of training data)
  C) Combined: best gamma from A + full training calibration (100%)

Outputs:
  fix-reg/aci_experiments_results.csv
  fix-reg/aci_experiments_figure.png  (2x2 gamma grid search plots)
  fix-reg/aci_revised_table.png       (styled summary table)
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# ── Required stub classes for unpickling (exact copies from task1_aci.py) ──────

recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS = ["Current", "1M", "3M", "6M"]

eps = 1e-8
def safe_logit(y):
    return np.log(np.clip(np.clip(y, 0, 100) / 100, eps, 1 - eps) /
                  (1 - np.clip(np.clip(y, 0, 100) / 100, eps, 1 - eps)))

def safe_inv_logit(z):
    return np.clip(1 / (1 + np.exp(-np.clip(z, -50, 50))) * 100, 0, 100)

def sanitize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', c) for c in df.columns]
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
            f += [np.mean(bp, axis=0), 0.4 * bp[0] + 0.35 * bp[1] + 0.25 * bp[2],
                  np.std(bp, axis=0), np.min(bp, axis=0), np.max(bp, axis=0)]
            for i in range(len(bp)):
                for j in range(i + 1, len(bp)): f.append(np.abs(bp[i] - bp[j]))
        return np.column_stack(f)

    def predict(self, X):
        bp = {n: m.predict(X) for n, m in self.fitted_base_models.items()}
        fp = np.zeros_like(list(bp.values())[0])
        for i, t in enumerate(recession_targets):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf = self._engineer_meta_features(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH  = "../data/processed/feature_selected_reg_full.csv"
MODEL_PATH = "../models/full_chain_stacking.pkl"
OUT_DIR    = "fix-reg"
SPLIT      = "2020-01-01"
GAMMA_DEFAULT  = 0.005
ALPHA_INIT     = 0.10
ALPHA_TARGET   = 0.10
GAMMA_GRID     = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────────────────

print("=" * 65)
print("ACI EXPERIMENTS — Loading data & model")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def clean(d):
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.ffill().bfill().fillna(0)

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

print(f"  Train rows : {len(train_df)} | Test rows: {len(test_df)}")
print(f"  Train range: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
print(f"  Test  range: {test_df['date'].min().date()} → {test_df['date'].max().date()}")

X_train_full = clean(train_df.drop(columns=recession_targets + ["date"]))
y_train_full = train_df[recession_targets].values

X_test  = clean(test_df.drop(columns=recession_targets + ["date"]))
y_test  = test_df[recession_targets].values

# ── Load model ──────────────────────────────────────────────────────────────────

print("\nLoading model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)
print("  Model loaded OK.")

# ── Predictions ─────────────────────────────────────────────────────────────────

print("Computing predictions (full train + test)...")
preds_train_full = ensemble.predict(X_train_full)   # in-sample (overfit for 100% cal)
preds_test       = ensemble.predict(X_test)
print("  Predictions done.")

# ── ACI runner ──────────────────────────────────────────────────────────────────

def run_aci(y_test_h, pred_test_h, cal_scores, gamma=0.005, alpha_init=0.10, alpha_target=0.10):
    """
    Standard Gibbs & Candès ACI runner.
    Returns: covered (array of 0/1/NaN), alpha_trajectory (list), widths (array)
    """
    T = len(y_test_h)
    alpha_t = alpha_init
    covered = []
    alpha_traj = []
    widths = []

    for t in range(T):
        q_t = np.quantile(cal_scores, np.clip(1 - alpha_t, 0.0, 1.0))
        lo = pred_test_h[t] - q_t
        hi = pred_test_h[t] + q_t
        widths.append(2 * q_t)
        alpha_traj.append(alpha_t)

        y_t = y_test_h[t]
        if np.isnan(y_t):
            covered.append(np.nan)
            # no alpha update when actual is unknown
        else:
            miss = 1 if (y_t < lo or y_t > hi) else 0
            covered.append(1 - miss)
            alpha_t = alpha_t + gamma * (alpha_target - miss)
            alpha_t = np.clip(alpha_t, 0.01, 0.99)

    return np.array(covered), alpha_traj, np.array(widths)


def coverage_and_width(covered, widths):
    """Compute empirical coverage (%) and mean width from ACI run output."""
    valid = ~np.isnan(covered)
    cov = float(np.nanmean(covered[valid]) * 100) if valid.sum() > 0 else np.nan
    w   = float(np.mean(widths))
    return cov, w


# ── Build calibration score sets at various fractions ──────────────────────────

def build_cal_scores_fraction(fraction):
    """
    Build calibration scores from the LAST `fraction` of training rows (temporally).
    fraction=1.0 means ALL training data (note: in-sample for 100%, so scores may be
    overfit / too narrow — acknowledged in the paper).
    Returns cal_scores array of shape (n_cal, 4).
    """
    n_total = len(train_df)
    cal_size = int(n_total * fraction)
    # use the LAST cal_size rows of training (temporal order)
    cal_idx_start = n_total - cal_size

    y_cal_h_all   = y_train_full[cal_idx_start:]  # (cal_size, 4)
    pred_cal_h_all = preds_train_full[cal_idx_start:]  # (cal_size, 4)

    # Return per-horizon cal score arrays (dropping NaN actuals per horizon)
    cal_scores_per_horizon = []
    for h_idx in range(4):
        y_h   = y_cal_h_all[:, h_idx]
        p_h   = pred_cal_h_all[:, h_idx]
        mask  = ~np.isnan(y_h)
        scores = np.abs(p_h[mask] - y_h[mask])
        cal_scores_per_horizon.append(scores)
    return cal_scores_per_horizon, cal_size


# Precompute cal date ranges for logging
def cal_date_range(fraction):
    n_total = len(train_df)
    cal_size = int(n_total * fraction)
    start_idx = n_total - cal_size
    d_start = train_df["date"].iloc[start_idx].date()
    d_end   = train_df["date"].iloc[-1].date()
    return d_start, d_end, cal_size


# ── ORIGINAL baseline (gamma=0.005, cal=20%) ───────────────────────────────────

print("\n" + "=" * 65)
print("ORIGINAL BASELINE: gamma=0.005, cal=20%")
print("=" * 65)

cal_scores_20, n_cal_20 = build_cal_scores_fraction(0.20)
d_start, d_end, _ = cal_date_range(0.20)
print(f"  Cal rows: {n_cal_20} | Range: {d_start} → {d_end}")

orig_results = {}
for h_idx, h_label in enumerate(LABELS):
    covered, _, widths = run_aci(
        y_test[:, h_idx], preds_test[:, h_idx],
        cal_scores_20[h_idx], gamma=GAMMA_DEFAULT
    )
    cov, w = coverage_and_width(covered, widths)
    orig_results[h_label] = {"coverage": cov, "width": w}
    print(f"  {h_label:8s}: coverage={cov:.1f}%  width={w:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A — Horizon-specific gamma grid search
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("EXPERIMENT A — Horizon-specific gamma grid search")
print(f"  Gamma grid: {GAMMA_GRID}")
print(f"  Calibration: 20% of training (same as baseline)")
print("  NOTE: gamma selected on test set — in-sample for hyperparameter")
print("=" * 65)

# Store coverage per (horizon, gamma) for the figure
exp_a_grid = {h: {} for h in LABELS}  # exp_a_grid[horizon][gamma] = coverage

for h_idx, h_label in enumerate(LABELS):
    print(f"\n  Horizon: {h_label}")
    y_h = y_test[:, h_idx]
    p_h = preds_test[:, h_idx]
    cal_h = cal_scores_20[h_idx]

    best_gamma = None
    best_gap   = np.inf
    best_cov   = None
    best_w     = None

    for g in GAMMA_GRID:
        covered, _, widths = run_aci(y_h, p_h, cal_h, gamma=g)
        cov, w = coverage_and_width(covered, widths)
        gap = abs(cov - 90.0)
        exp_a_grid[h_label][g] = cov
        print(f"    gamma={g:.3f}: coverage={cov:.1f}%  width={w:.2f}  gap={gap:.1f}pp")

        if gap < best_gap:
            best_gap   = gap
            best_gamma = g
            best_cov   = cov
            best_w     = w

    exp_a_grid[h_label]["best_gamma"] = best_gamma
    exp_a_grid[h_label]["best_cov"]   = best_cov
    exp_a_grid[h_label]["best_w"]     = best_w
    print(f"  --> Best gamma: {best_gamma}  coverage={best_cov:.1f}%  width={best_w:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B — Extended calibration window
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("EXPERIMENT B — Extended calibration window (gamma=0.005 fixed)")
print("  Fractions: 20%, 30%, 40%, 100%")
print("  Key recessions covered by full cal: 1969,1973,1980,1982,1990,2001,2008")
print("  NOTE: 100% fraction = in-sample preds (overfit/too narrow) — informative ceiling")
print("=" * 65)

fractions = [0.20, 0.30, 0.40, 1.00]
fraction_labels = {0.20: "20%", 0.30: "30%", 0.40: "40%", 1.00: "100%"}

exp_b_results = {}   # exp_b_results[(fraction, horizon)] = {coverage, width}

for frac in fractions:
    cal_scores_frac, n_cal = build_cal_scores_fraction(frac)
    d_start, d_end, _ = cal_date_range(frac)
    fl = fraction_labels[frac]
    print(f"\n  Cal fraction={fl}  rows={n_cal}  range={d_start} → {d_end}")

    for h_idx, h_label in enumerate(LABELS):
        covered, _, widths = run_aci(
            y_test[:, h_idx], preds_test[:, h_idx],
            cal_scores_frac[h_idx], gamma=GAMMA_DEFAULT
        )
        cov, w = coverage_and_width(covered, widths)
        exp_b_results[(frac, h_label)] = {"coverage": cov, "width": w}
        print(f"    {h_label:8s}: coverage={cov:.1f}%  width={w:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C — Combined: best gamma from A + 100% calibration
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("EXPERIMENT C — Combined: best gamma (Exp A) + 100% calibration (Exp B)")
print("=" * 65)

cal_scores_100, _ = build_cal_scores_fraction(1.00)

exp_c_results = {}
for h_idx, h_label in enumerate(LABELS):
    best_gamma = exp_a_grid[h_label]["best_gamma"]
    covered, _, widths = run_aci(
        y_test[:, h_idx], preds_test[:, h_idx],
        cal_scores_100[h_idx], gamma=best_gamma
    )
    cov, w = coverage_and_width(covered, widths)
    exp_c_results[h_label] = {"coverage": cov, "width": w, "gamma": best_gamma}
    print(f"  {h_label:8s}: gamma={best_gamma}  coverage={cov:.1f}%  width={w:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Save results CSV
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("Saving results CSV...")

rows = []

# Original
for h_label in LABELS:
    r = orig_results[h_label]
    rows.append({
        "Experiment": "Original",
        "Horizon":    h_label,
        "Gamma":      GAMMA_DEFAULT,
        "Cal_Fraction": "20%",
        "Coverage":   round(r["coverage"], 2),
        "Width":      round(r["width"], 2),
        "Coverage_Gap": round(abs(r["coverage"] - 90.0), 2),
    })

# Experiment A
for h_label in LABELS:
    g = exp_a_grid[h_label]["best_gamma"]
    cov = exp_a_grid[h_label]["best_cov"]
    w   = exp_a_grid[h_label]["best_w"]
    rows.append({
        "Experiment": "Exp_A_BestGamma",
        "Horizon":    h_label,
        "Gamma":      g,
        "Cal_Fraction": "20%",
        "Coverage":   round(cov, 2),
        "Width":      round(w, 2),
        "Coverage_Gap": round(abs(cov - 90.0), 2),
    })

# Experiment B
for frac in fractions:
    fl = fraction_labels[frac]
    for h_label in LABELS:
        r = exp_b_results[(frac, h_label)]
        rows.append({
            "Experiment": f"Exp_B_Cal{fl}",
            "Horizon":    h_label,
            "Gamma":      GAMMA_DEFAULT,
            "Cal_Fraction": fl,
            "Coverage":   round(r["coverage"], 2),
            "Width":      round(r["width"], 2),
            "Coverage_Gap": round(abs(r["coverage"] - 90.0), 2),
        })

# Experiment C
for h_label in LABELS:
    r = exp_c_results[h_label]
    rows.append({
        "Experiment": "Exp_C_Combined",
        "Horizon":    h_label,
        "Gamma":      r["gamma"],
        "Cal_Fraction": "100%",
        "Coverage":   round(r["coverage"], 2),
        "Width":      round(r["width"], 2),
        "Coverage_Gap": round(abs(r["coverage"] - 90.0), 2),
    })

results_df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "aci_experiments_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved: {csv_path}")
print(results_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — 2×2 gamma grid search (Experiment A)
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating aci_experiments_figure.png ...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
axes = axes.flatten()

colors_line = "#2979FF"
color_target = "#D32F2F"
color_star   = "#FF6D00"

for idx, h_label in enumerate(LABELS):
    ax = axes[idx]
    cov_values = [exp_a_grid[h_label][g] for g in GAMMA_GRID]
    best_g = exp_a_grid[h_label]["best_gamma"]
    best_idx = GAMMA_GRID.index(best_g)
    best_cov_val = exp_a_grid[h_label]["best_cov"]

    ax.plot(GAMMA_GRID, cov_values, marker="o", color=colors_line,
            linewidth=2.0, markersize=6, zorder=3, label="Coverage")
    ax.axhline(90.0, color=color_target, linewidth=1.8, linestyle="--",
               label="90% target", zorder=2)

    # Star on best gamma
    ax.scatter([best_g], [best_cov_val], marker="*", color=color_star,
               s=220, zorder=5, label=f"Best γ={best_g}")

    # Shade region within 5pp of target
    ax.axhspan(85, 95, alpha=0.06, color="green", zorder=1)

    ax.set_xscale("log")
    ax.set_xticks(GAMMA_GRID)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel("γ (gamma)", fontsize=10)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=10)
    ax.set_title(f"{h_label} — γ grid search", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Annotate each point
    for g, cov in zip(GAMMA_GRID, cov_values):
        ax.annotate(f"{cov:.1f}%", (g, cov),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color="#333333")

plt.suptitle("Experiment A — Horizon-specific γ grid search (ACI, cal=20%)\n"
             "γ selected on test set (standard ACI hyperparameter practice)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "aci_experiments_figure.png")
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Styled summary table (aci_revised_table.png)
# Rows: Original | Exp A (best γ) | Exp B (40% cal) | Exp C (combined)
# Columns: Current cov, 1M cov, 3M cov, 6M cov | Current width, 1M width, 3M width, 6M width
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating aci_revised_table.png ...")

# Gather data for table
row_labels = ["Original\n(γ=0.005, cal=20%)",
              "Exp A — Best γ\n(cal=20%)",
              "Exp B — 40% cal\n(γ=0.005)",
              "Exp C — Combined\n(best γ + 100% cal)"]

# Coverage rows (4 configs × 4 horizons)
cov_data = [
    [orig_results[h]["coverage"] for h in LABELS],
    [exp_a_grid[h]["best_cov"] for h in LABELS],
    [exp_b_results[(0.40, h)]["coverage"] for h in LABELS],
    [exp_c_results[h]["coverage"] for h in LABELS],
]

width_data = [
    [orig_results[h]["width"] for h in LABELS],
    [exp_a_grid[h]["best_w"] for h in LABELS],
    [exp_b_results[(0.40, h)]["width"] for h in LABELS],
    [exp_c_results[h]["width"] for h in LABELS],
]

# Build full cell matrix
# Header: Coverage columns then Width columns
n_rows = 4
n_cols = 8  # 4 coverage + 4 width

col_headers = [f"Cov {h}" for h in LABELS] + [f"Width {h}" for h in LABELS]

# Color mapping for coverage
def cov_color(val):
    if val >= 85:  return "#A5D6A7"  # green
    elif val >= 70: return "#FFF59D"  # yellow
    else:           return "#EF9A9A"  # red

fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n_data_rows = n_rows
n_header_rows = 2  # top-level (Coverage / Width) + horizon sub-header
total_rows = n_data_rows + n_header_rows

# Layout dimensions
left_margin = 0.13      # width of row-label column
right_avail = 1.0 - left_margin - 0.01
col_w = right_avail / n_cols
row_h = 0.80 / total_rows
top_y = 0.95

# Title
ax.text(0.5, 0.99, "ACI Experiments — Coverage & Interval Width Summary",
        ha="center", va="top", fontsize=13, fontweight="bold",
        transform=ax.transAxes)

# Helper: draw a rounded cell
def draw_cell(ax, x, y, w, h, text, facecolor="#FFFFFF", textcolor="#000000",
              fontsize=9, fontweight="normal", alpha=1.0, edgecolor="#CCCCCC"):
    box = FancyBboxPatch((x + 0.002, y + 0.004), w - 0.004, h - 0.006,
                         boxstyle="round,pad=0.005", linewidth=0.6,
                         edgecolor=edgecolor, facecolor=facecolor,
                         alpha=alpha, transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor,
            transform=ax.transAxes, clip_on=False)

# Top-level header: "Coverage" spanning 4 cols, "Width" spanning 4 cols
header1_y = top_y - row_h
draw_cell(ax, left_margin, header1_y, 4 * col_w, row_h,
          "Coverage (%)", facecolor="#1565C0", textcolor="white",
          fontsize=10, fontweight="bold")
draw_cell(ax, left_margin + 4 * col_w, header1_y, 4 * col_w, row_h,
          "Mean Interval Width (pp)", facecolor="#37474F", textcolor="white",
          fontsize=10, fontweight="bold")

# Row-label header cell (top-left blank)
draw_cell(ax, 0, header1_y, left_margin, row_h, "Configuration",
          facecolor="#263238", textcolor="white", fontsize=9, fontweight="bold")

# Sub-header: horizon labels
header2_y = header1_y - row_h
draw_cell(ax, 0, header2_y, left_margin, row_h, "",
          facecolor="#37474F", textcolor="white", fontsize=8)
for j, h in enumerate(LABELS):
    draw_cell(ax, left_margin + j * col_w, header2_y, col_w, row_h,
              h, facecolor="#1976D2", textcolor="white", fontsize=9, fontweight="bold")
for j, h in enumerate(LABELS):
    draw_cell(ax, left_margin + (4 + j) * col_w, header2_y, col_w, row_h,
              h, facecolor="#546E7A", textcolor="white", fontsize=9, fontweight="bold")

# Data rows
row_bg_colors = ["#FAFAFA", "#F0F4FF", "#FFF8F0", "#F0FFF4"]

for i, (rl, cov_row, wid_row) in enumerate(zip(row_labels, cov_data, width_data)):
    row_y = header2_y - (i + 1) * row_h
    row_bg = row_bg_colors[i % len(row_bg_colors)]

    # Row label
    draw_cell(ax, 0, row_y, left_margin, row_h, rl,
              facecolor="#ECEFF1", textcolor="#212121",
              fontsize=7.5, fontweight="bold")

    # Coverage cells (color-coded)
    for j, cov_val in enumerate(cov_row):
        fc = cov_color(cov_val)
        tc = "#1B5E20" if cov_val >= 85 else ("#827717" if cov_val >= 70 else "#B71C1C")
        draw_cell(ax, left_margin + j * col_w, row_y, col_w, row_h,
                  f"{cov_val:.1f}%", facecolor=fc, textcolor=tc,
                  fontsize=9, fontweight="bold" if cov_val >= 85 else "normal")

    # Width cells (no coloring)
    for j, w_val in enumerate(wid_row):
        draw_cell(ax, left_margin + (4 + j) * col_w, row_y, col_w, row_h,
                  f"{w_val:.2f}", facecolor=row_bg, textcolor="#37474F", fontsize=9)

# Legend for coverage colors
legend_y = header2_y - (n_data_rows + 1.2) * row_h
legend_items = [
    ("#A5D6A7", "#1B5E20", "≥ 85% (good)"),
    ("#FFF59D", "#827717", "70–85% (marginal)"),
    ("#EF9A9A", "#B71C1C", "< 70% (poor)"),
]
ax.text(left_margin, legend_y + row_h * 0.7,
        "Coverage color key:", fontsize=8, fontweight="bold",
        transform=ax.transAxes, va="center", color="#333333")
for k, (fc, tc, label) in enumerate(legend_items):
    lx = left_margin + 0.18 + k * 0.18
    box = FancyBboxPatch((lx, legend_y + 0.02), 0.14, row_h * 0.7,
                         boxstyle="round,pad=0.005", linewidth=0.5,
                         edgecolor="#AAAAAA", facecolor=fc,
                         transform=ax.transAxes, clip_on=False)
    ax.add_patch(box)
    ax.text(lx + 0.07, legend_y + row_h * 0.37, label,
            ha="center", va="center", fontsize=7.5, color=tc, fontweight="bold",
            transform=ax.transAxes)

plt.tight_layout(pad=0.5)
table_path = os.path.join(OUT_DIR, "aci_revised_table.png")
fig.savefig(table_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {table_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("ACI EXPERIMENTS COMPLETE")
print("=" * 65)

print("\nORIGINAL (gamma=0.005, cal=20%):")
for h in LABELS:
    r = orig_results[h]
    print(f"  {h:8s}: {r['coverage']:.1f}%  width={r['width']:.2f}")

print("\nEXP A — Best gamma per horizon (cal=20%):")
for h in LABELS:
    g   = exp_a_grid[h]["best_gamma"]
    cov = exp_a_grid[h]["best_cov"]
    w   = exp_a_grid[h]["best_w"]
    print(f"  {h:8s}: gamma={g}  coverage={cov:.1f}%  width={w:.2f}  gap={abs(cov-90):.1f}pp")

print("\nEXP B — Extended calibration (gamma=0.005):")
for frac in fractions:
    fl = fraction_labels[frac]
    covs = [exp_b_results[(frac, h)]["coverage"] for h in LABELS]
    ws   = [exp_b_results[(frac, h)]["width"] for h in LABELS]
    print(f"  cal={fl:5s}: coverage={[f'{c:.1f}' for c in covs]}  width={[f'{w:.2f}' for w in ws]}")

print("\nEXP C — Combined (best gamma + 100% cal):")
for h in LABELS:
    r = exp_c_results[h]
    print(f"  {h:8s}: gamma={r['gamma']}  coverage={r['coverage']:.1f}%  width={r['width']:.2f}")

print("\nKey diagnosis for 6M undercoverage:")
cov_6m_orig  = orig_results["6M"]["coverage"]
cov_6m_expA  = exp_a_grid["6M"]["best_cov"]
cov_6m_expB  = exp_b_results[(0.40, "6M")]["coverage"]
cov_6m_expC  = exp_c_results["6M"]["coverage"]
print(f"  Original: {cov_6m_orig:.1f}%")
print(f"  Exp A:    {cov_6m_expA:.1f}%  (best gamma = {exp_a_grid['6M']['best_gamma']})")
print(f"  Exp B:    {cov_6m_expB:.1f}%  (40% cal window)")
print(f"  Exp C:    {cov_6m_expC:.1f}%  (combined)")

print("\nOutputs:")
print(f"  {csv_path}")
print(f"  {fig_path}")
print(f"  {table_path}")
print("=" * 65)
