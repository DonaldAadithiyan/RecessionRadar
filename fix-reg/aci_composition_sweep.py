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

DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
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



# ============================================================================
# P0.2 — Calibration-composition sweep at 20/25/30/35/40%
# Same methodology as Experiment B: cal scores = |pred-actual| over the LAST
# `fraction` of training rows (temporal tail); gamma=0.005; nominal 90%.
# Re-slices existing predictions only — NO retraining.
# ============================================================================
import csv as _csv
fractions=[0.20,0.25,0.30,0.35,0.40]
NOMINAL=90.0

# recession-month = smoothed recession_probability >= 50 (the spike months)
rec_prob_train = train_df["recession_probability"].values

rows_out=[]
print("\n"+"="*70)
print("P0.2 CALIBRATION-COMPOSITION SWEEP (gamma=0.005, nominal=90%)")
print("="*70)
comp_summary={}
for frac in fractions:
    cal_scores_frac, n_cal = build_cal_scores_fraction(frac)
    d_start,d_end,_=cal_date_range(frac)
    n_total=len(train_df); start_idx=n_total-int(n_total*frac)
    rec_in_cal=int((rec_prob_train[start_idx:]>=50).sum())
    comp_pct=100.0*rec_in_cal/n_cal
    comp_summary[frac]=(n_cal,rec_in_cal,comp_pct,str(d_start),str(d_end))
    print(f"\n cal_frac={int(frac*100)}%  rows={n_cal}  range={d_start}->{d_end}  "
          f"recession_months(>=50)={rec_in_cal}  composition={comp_pct:.1f}%")
    for h_idx,h_label in enumerate(LABELS):
        covered,_,widths=run_aci(y_test[:,h_idx],preds_test[:,h_idx],
                                 cal_scores_frac[h_idx],gamma=GAMMA_DEFAULT)
        cov,w=coverage_and_width(covered,widths)
        gap=cov-NOMINAL
        print(f"   {h_label:8s}: coverage={cov:6.2f}%  width={w:7.3f}  gap={gap:+.2f}pp")
        rows_out.append(dict(Cal_Fraction=f"{int(frac*100)}%",Cal_Rows=n_cal,
            Recession_Months=rec_in_cal,Composition_pct=round(comp_pct,2),
            Cal_Start=str(d_start),Cal_End=str(d_end),Horizon=h_label,Gamma=GAMMA_DEFAULT,
            Coverage=round(cov,2),Width=round(w,3),Coverage_Gap=round(gap,2),Nominal=NOMINAL))

with open("fix-reg/aci_composition_sweep.csv","w",newline="") as f:
    wtr=_csv.DictWriter(f,fieldnames=list(rows_out[0].keys())); wtr.writeheader(); wtr.writerows(rows_out)
print("\nSaved fix-reg/aci_composition_sweep.csv")

# ── Plot: coverage vs cal fraction, one line per horizon ──
import matplotlib.pyplot as _plt
fig,ax=_plt.subplots(figsize=(8,5))
xs=[int(f*100) for f in fractions]
colors={"Current":"#1a1a2e","1M":"#1565c0","3M":"#e65100","6M":"#2ca02c"}
for h in LABELS:
    ys=[next(r["Coverage"] for r in rows_out if r["Horizon"]==h and r["Cal_Fraction"]==f"{int(fr*100)}%") for fr in fractions]
    ax.plot(xs,ys,marker="o",lw=2,label=h,color=colors[h])
    for x,y in zip(xs,ys): ax.annotate(f"{y:.0f}",(x,y),textcoords="offset points",xytext=(0,6),fontsize=7,ha="center",color=colors[h])
ax.axhline(90,color="gray",ls="--",lw=1,label="Nominal 90%")
ax.set_xlabel("Calibration set size (% of training tail)"); ax.set_ylabel("Empirical coverage (%)")
ax.set_title("ACI coverage vs calibration-set composition (5-point sweep)")
ax.set_xticks(xs); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.spines[["top","right"]].set_visible(False)
_plt.tight_layout(); _plt.savefig("fix-reg/aci_composition_sweep.png",dpi=140); _plt.close()
print("Saved fix-reg/aci_composition_sweep.png")

# composition summary table
print("\n=== COMPOSITION SUMMARY ===")
print(f"{'frac':>5} {'rows':>5} {'rec_mo':>7} {'comp%':>7} {'range':>25}")
for fr in fractions:
    n,r,c,s,e=comp_summary[fr]
    print(f"{int(fr*100):>4}% {n:>5} {r:>7} {c:>6.1f}% {s} -> {e}")
