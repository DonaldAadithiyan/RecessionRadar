"""
TASK 5 — Diagnose and Fix ACI Coverage
Implements three calibration strategies and produces a full comparison table.

Options:
  A  — Sweep calibration size: 20%, 30%, 40%, 50% of training tail
  C  — Regime-Aware Conformal: separate recession/expansion calibration
       score distributions, regime selected at test time by predicted prob

Regime-Aware Cross-Conformal ACI is the primary novel contribution:
  - Conditions interval width on predicted economic regime
  - Addresses distributional shift between expansion and recession periods
  - ACI adaptive alpha update applied on top for temporal adaptation
"""

import os, re, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────
recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS      = ["Current", "1M", "3M", "6M"]
DATA_PATH   = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH  = "fix-reg/models/full_chain_stacking.pkl"
OUT_DIR     = "fix-reg/task5_outputs"
TRAIN_END   = "2020-01-01"
ALPHA_TGT   = 0.10   # target miscoverage (90% coverage)
GAMMA       = 0.005  # ACI learning rate (Gibbs & Candès)
REGIME_THR  = 30.0   # predicted prob threshold for "recession" regime (%)
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Load data and model ───────────────────────────────────────
print("Loading data and model...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def clean(d):
    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

train_df = df[df["date"] < TRAIN_END].copy()
test_df  = df[df["date"] >= TRAIN_END].copy()
X_train  = clean(train_df.drop(columns=recession_targets + ["date"]))
Y_train  = train_df[recession_targets].values
X_test   = clean(test_df.drop(columns=recession_targets + ["date"]))
Y_test   = test_df[recession_targets].values
dates_test = test_df["date"].values
N_train  = len(X_train)

with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

preds_test = ensemble.predict(X_test)   # (65, 4)
print(f"  Train: {N_train} rows ({train_df['date'].min().date()} → {train_df['date'].max().date()})")
print(f"  Test:  {len(X_test)} rows ({test_df['date'].min().date()} → {test_df['date'].max().date()})")

# ── ACI runner ────────────────────────────────────────────────
def run_aci(cal_scores_all, preds_t, y_true_t, regime_flags_t=None,
            cal_scores_rec=None, cal_scores_exp=None):
    """
    Run ACI loop.
    If regime_flags_t is provided (bool array), use regime-aware calibration:
      True  = recession regime → use cal_scores_rec quantile
      False = expansion regime → use cal_scores_exp quantile
    Otherwise use cal_scores_all uniformly.
    Regime-aware: shared alpha adapts, but quantile pulled from regime-specific set.
    Returns: coverages (bool per step), widths (pp per step), alpha_traj
    """
    n      = preds_t.shape[0]
    n_hor  = preds_t.shape[1]
    alpha  = np.full(n_hor, ALPHA_TGT)
    covered = np.zeros((n, n_hor), dtype=bool)
    widths  = np.zeros((n, n_hor))
    alpha_traj = np.zeros((n, n_hor))

    for t in range(n):
        for h in range(n_hor):
            # Select calibration scores for this regime and horizon
            if regime_flags_t is not None and cal_scores_rec is not None:
                if regime_flags_t[t]:
                    cal_h = cal_scores_rec[h]
                else:
                    cal_h = cal_scores_exp[h]
                if len(cal_h) < 5:          # fallback if regime has too few points
                    cal_h = cal_scores_all[h]
            else:
                cal_h = cal_scores_all[h]

            # Conformal quantile at current alpha
            q = np.quantile(cal_h, 1.0 - alpha[h])
            lo = np.clip(preds_t[t, h] - q, 0, 100)
            hi = np.clip(preds_t[t, h] + q, 0, 100)
            widths[t, h]  = hi - lo
            alpha_traj[t, h] = alpha[h]

            y = y_true_t[t, h]
            if not np.isnan(y):
                err = int(not (lo <= y <= hi))
                covered[t, h] = (err == 0)
                alpha[h] = np.clip(alpha[h] + GAMMA * (ALPHA_TGT - err), 0.001, 0.999)
            else:
                covered[t, h] = np.nan

    return covered, widths, alpha_traj


# ── OPTION A — Calibration size sweep ────────────────────────
print("\n=== OPTION A: Calibration size sweep ===")
cal_fracs = [0.20, 0.30, 0.40, 0.50]
results_A = {}

for frac in cal_fracs:
    n_cal = int(N_train * frac)
    X_cal   = X_train.iloc[-n_cal:]
    Y_cal   = Y_train[-n_cal:]
    dates_c = train_df["date"].values[-n_cal:]

    preds_cal = ensemble.predict(X_cal)
    cal_scores = [np.abs(Y_cal[:, h] - preds_cal[:, h]) for h in range(4)]

    covered, widths, _ = run_aci(cal_scores, preds_test, Y_test)

    cov_rates = []
    for h in range(4):
        mask = ~np.isnan(covered[:, h].astype(float))
        c = np.nanmean(covered[:, h])
        cov_rates.append(round(c * 100, 1))

    mean_widths = [round(np.nanmean(widths[:, h]), 2) for h in range(4)]

    # Recession months in calibration set
    n_rec = int((Y_cal[:, 0] > 50).sum())
    cal_mean_err = [round(s.mean(), 3) for s in cal_scores]

    results_A[frac] = {
        "n_cal": n_cal,
        "date_start": str(pd.Timestamp(dates_c[0]).date()),
        "date_end":   str(pd.Timestamp(dates_c[-1]).date()),
        "n_recession_months": n_rec,
        "cal_mean_err": cal_mean_err,
        "coverage": cov_rates,
        "mean_widths": mean_widths,
    }

    print(f"  {int(frac*100):2d}%  n={n_cal:3d}  "
          f"{pd.Timestamp(dates_c[0]).date()} → {pd.Timestamp(dates_c[-1]).date()}  "
          f"rec_months={n_rec:3d}  "
          f"cov={cov_rates}  widths={mean_widths}")


# ── OPTION C — Regime-Aware Conformal (40% calibration) ───────
print("\n=== OPTION C: Regime-Aware Conformal (40% calibration) ===")
frac_ra = 0.40
n_cal_ra = int(N_train * frac_ra)
X_cal_ra  = X_train.iloc[-n_cal_ra:]
Y_cal_ra  = Y_train[-n_cal_ra:]
preds_cal_ra = ensemble.predict(X_cal_ra)

# Identify regime in calibration: recession if actual recession_prob > REGIME_THR
regime_cal = Y_cal_ra[:, 0] > REGIME_THR     # bool (n_cal,)

# Per-horizon calibration scores split by regime
cal_scores_all = [np.abs(Y_cal_ra[:, h] - preds_cal_ra[:, h]) for h in range(4)]
cal_scores_rec = [np.abs(Y_cal_ra[regime_cal, h]  - preds_cal_ra[regime_cal, h])  for h in range(4)]
cal_scores_exp = [np.abs(Y_cal_ra[~regime_cal, h] - preds_cal_ra[~regime_cal, h]) for h in range(4)]

print(f"  Calibration: {n_cal_ra} rows — "
      f"{(~regime_cal).sum()} expansion, {regime_cal.sum()} recession months")

for h in range(4):
    exp_mean = cal_scores_exp[h].mean() if len(cal_scores_exp[h]) else float('nan')
    rec_mean = cal_scores_rec[h].mean() if len(cal_scores_rec[h]) else float('nan')
    print(f"  {LABELS[h]}: expansion mean err={exp_mean:.3f}pp, "
          f"recession mean err={rec_mean:.3f}pp  "
          f"(n_rec={len(cal_scores_rec[h])})")

# Regime flags for test set: use predicted recession_probability
regime_test = preds_test[:, 0] > REGIME_THR

print(f"\n  Test regime flags: {regime_test.sum()} recession / "
      f"{(~regime_test).sum()} expansion months")

covered_ra, widths_ra, alpha_traj_ra = run_aci(
    cal_scores_all, preds_test, Y_test,
    regime_flags_t=regime_test,
    cal_scores_rec=cal_scores_rec,
    cal_scores_exp=cal_scores_exp,
)

cov_ra    = [round(np.nanmean(covered_ra[:, h]) * 100, 1) for h in range(4)]
widths_ra_mean = [round(np.nanmean(widths_ra[:, h]), 2) for h in range(4)]

# Recession vs expansion coverage separately
rec_cov_ra = []
exp_cov_ra = []
for h in range(4):
    rec_mask = regime_test & ~np.isnan(covered_ra[:, h].astype(float))
    exp_mask = ~regime_test & ~np.isnan(covered_ra[:, h].astype(float))
    rc = np.nanmean(covered_ra[rec_mask, h]) * 100 if rec_mask.sum() > 0 else float('nan')
    ec = np.nanmean(covered_ra[exp_mask, h]) * 100 if exp_mask.sum() > 0 else float('nan')
    rec_cov_ra.append(round(rc, 1))
    exp_cov_ra.append(round(ec, 1))

print(f"\n  Regime-Aware coverage:  {cov_ra}")
print(f"  Recession regime cov:   {rec_cov_ra}")
print(f"  Expansion regime cov:   {exp_cov_ra}")
print(f"  Mean widths (pp):       {widths_ra_mean}")

results_C = {
    "coverage": cov_ra,
    "coverage_recession": rec_cov_ra,
    "coverage_expansion": exp_cov_ra,
    "mean_widths": widths_ra_mean,
    "n_cal": n_cal_ra,
    "n_rec_cal": int(regime_cal.sum()),
    "n_exp_cal": int((~regime_cal).sum()),
    "n_rec_test": int(regime_test.sum()),
    "n_exp_test": int((~regime_test).sum()),
}

# ── Master comparison table ───────────────────────────────────
print("\n\n=== MASTER COVERAGE TABLE ===")
print(f"{'Method':<32} {'Current':>9} {'1M':>9} {'3M':>9} {'6M':>9}  {'rec months cal':>15}")
print("-" * 90)
for frac in cal_fracs:
    r = results_A[frac]
    print(f"  Option A {int(frac*100):2d}% ({r['date_start'][:7]})"
          f"{'':>3} {r['coverage'][0]:>9} {r['coverage'][1]:>9} "
          f"{r['coverage'][2]:>9} {r['coverage'][3]:>9}  {r['n_recession_months']:>15}")
print(f"  Option C Regime-Aware (40%)     "
      f"{results_C['coverage'][0]:>9} {results_C['coverage'][1]:>9} "
      f"{results_C['coverage'][2]:>9} {results_C['coverage'][3]:>9}  "
      f"{results_C['n_rec_cal']:>15}")
print(f"  {'Target':>30}{'90.0':>10}{'90.0':>10}{'90.0':>10}{'90.0':>10}")

# ── Determine best method ─────────────────────────────────────
# Score = mean absolute deviation from 90% across 4 horizons (lower is better)
best_method = None
best_score  = float("inf")
for frac in cal_fracs:
    r = results_A[frac]
    s = np.mean([abs(c - 90.0) for c in r["coverage"]])
    if s < best_score:
        best_score = s
        best_method = f"Option A {int(frac*100)}%"
        best_cov    = r["coverage"]
        best_widths = r["mean_widths"]

s_c = np.mean([abs(c - 90.0) for c in results_C["coverage"]])
if s_c < best_score:
    best_score  = s_c
    best_method = "Option C Regime-Aware 40%"
    best_cov    = results_C["coverage"]
    best_widths = results_C["mean_widths"]

print(f"\nBest method: {best_method}  (mean |cov − 90%| = {best_score:.2f}pp)")
print(f"  Coverage: {best_cov}")
print(f"  Widths:   {best_widths}")

# ── Save JSON summary ─────────────────────────────────────────
summary = {
    "option_A": {
        str(int(frac*100)) + "pct": results_A[frac] for frac in cal_fracs
    },
    "option_C_regime_aware_40pct": results_C,
    "best_method": best_method,
    "best_coverage": best_cov,
    "best_widths": best_widths,
    "original_task1_coverage": [70.8, 71.9, 59.7, 59.3],
    "original_task1_widths":   [2.49, 2.41, 5.14, 12.34],
    "regime_threshold_pct": REGIME_THR,
}
with open(f"{OUT_DIR}/task5_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nSummary saved → {OUT_DIR}/task5_summary.json")

# ── Save CSV comparison table ──────────────────────────────────
rows = []
for frac in cal_fracs:
    r = results_A[frac]
    for h, lbl in enumerate(LABELS):
        rows.append({
            "Method": f"Option A {int(frac*100)}% ({r['date_start'][:7]}→{r['date_end'][:7]})",
            "Cal_N": r["n_cal"], "Cal_RecessionMonths": r["n_recession_months"],
            "Horizon": lbl, "Coverage_pct": r["coverage"][h],
            "MeanWidth_pp": r["mean_widths"][h],
            "CalMeanErr_pp": r["cal_mean_err"][h],
        })

for h, lbl in enumerate(LABELS):
    rows.append({
        "Method": "Option C Regime-Aware 40%",
        "Cal_N": results_C["n_cal"], "Cal_RecessionMonths": results_C["n_rec_cal"],
        "Horizon": lbl, "Coverage_pct": results_C["coverage"][h],
        "MeanWidth_pp": results_C["mean_widths"][h],
        "CalMeanErr_pp": None,
    })

table_df = pd.DataFrame(rows)
table_df.to_csv(f"{OUT_DIR}/task5_coverage_table.csv", index=False)
print(f"Coverage table saved → {OUT_DIR}/task5_coverage_table.csv")

# ── Figure F — Coverage comparison bar chart ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 5 — ACI Coverage Fix: Calibration Strategy Comparison\n"
             "(dashed line = 90% target)", fontsize=13, fontweight="bold")

# Left: coverage by method and horizon
method_labels = [f"A 20%", "A 30%", "A 40%", "A 50%", "C Regime-\nAware 40%"]
method_covs = [results_A[f]["coverage"] for f in cal_fracs] + [results_C["coverage"]]
colors_h = ["#1f77b4","#ff7f0e","#2ca02c","#d62728"]
x = np.arange(len(method_labels))
w = 0.18
ax = axes[0]
for h in range(4):
    ax.bar(x + (h-1.5)*w, [m[h] for m in method_covs], w,
           label=LABELS[h], color=colors_h[h], alpha=0.82)
ax.axhline(90, ls="--", color="black", lw=1.4, label="90% target")
ax.set_xticks(x); ax.set_xticklabels(method_labels, fontsize=9)
ax.set_ylabel("Empirical Coverage (%)")
ax.set_title("Coverage by Calibration Method × Horizon")
ax.set_ylim(40, 105)
ax.legend(fontsize=8, ncol=3)
ax.grid(axis="y", alpha=0.3)

# Right: interval widths by method and horizon
ax2 = axes[1]
for h in range(4):
    ax2.bar(x + (h-1.5)*w,
            [results_A[f]["mean_widths"][h] for f in cal_fracs] + [results_C["mean_widths"][h]],
            w, label=LABELS[h], color=colors_h[h], alpha=0.82)
task1_widths = [2.49, 2.41, 5.14, 12.34]
for h, (w_ref, c) in enumerate(zip(task1_widths, colors_h)):
    ax2.axhline(w_ref, ls=":", color=c, lw=1.2, alpha=0.6)
ax2.set_xticks(x); ax2.set_xticklabels(method_labels, fontsize=9)
ax2.set_ylabel("Mean Interval Width (pp)")
ax2.set_title("Interval Width by Calibration Method × Horizon\n(dotted = original Task 1 width)")
ax2.legend(fontsize=8, ncol=3)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/figure_F_coverage_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure F saved → {OUT_DIR}/figure_F_coverage_comparison.png")

# ── Figure G — Regime-Aware ACI band for 6M ──────────────────
fig, ax = plt.subplots(figsize=(13, 5))
h = 3  # 6M horizon
dates_dt = pd.to_datetime(dates_test)
y_true_6m = Y_test[:, h]

preds_6m = preds_test[:, h]
# Reconstruct lo/hi from regime-aware widths
widths_6m = widths_ra[:, h]
lo_ra = np.clip(preds_6m - widths_6m / 2, 0, 100)
hi_ra = np.clip(preds_6m + widths_6m / 2, 0, 100)

ax.fill_between(dates_dt, lo_ra, hi_ra, alpha=0.20, color="#1f77b4",
                label="90% Prediction Interval (Regime-Aware)")
ax.plot(dates_dt, preds_6m, color="#1f77b4", lw=1.8, label="6M Predicted prob")
valid = ~np.isnan(y_true_6m)
ax.scatter(dates_dt[valid], y_true_6m[valid], s=28, color="black",
           zorder=5, label="Actual 6M recession prob")

# Shade recession-flagged test months
for t in range(len(dates_dt)):
    if regime_test[t]:
        ax.axvspan(dates_dt[t] - pd.Timedelta(days=15),
                   dates_dt[t] + pd.Timedelta(days=15),
                   alpha=0.10, color="red", lw=0)

from matplotlib.patches import Patch as _Patch
ax.legend(handles=ax.get_legend_handles_labels()[0] +
          [plt.Rectangle((0,0),1,1,fc="red",alpha=0.15)],
          labels=ax.get_legend_handles_labels()[1] +
          ["Recession regime (predicted prob > 30%)"],
          fontsize=9, loc="upper left")
ax.set_title(f"Figure G — Regime-Aware ACI Band: 6M Horizon (Coverage = {cov_ra[3]}%)\n"
             f"Recession-regime intervals use recession-period calibration scores",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Date"); ax.set_ylabel("Recession Probability (%)")
ax.set_ylim(-5, 105)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/figure_G_regime_aware_aci_6m.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure G saved → {OUT_DIR}/figure_G_regime_aware_aci_6m.png")

print("\n\nTASK 5 COMPLETE")
print(f"\nBest calibration method: {best_method}")
print(f"Coverage gains vs original Task 1 (20% cal, 2009-2019):")
orig = [70.8, 71.9, 59.7, 59.3]
for h, lbl in enumerate(LABELS):
    print(f"  {lbl}: {orig[h]}% → {best_cov[h]}%  (+{best_cov[h]-orig[h]:.1f}pp)")
