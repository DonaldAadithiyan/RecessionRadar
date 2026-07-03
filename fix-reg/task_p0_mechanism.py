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
# P0 — Nonconformity scores by regime (rare-event vs expansion)
# Score = |actual - predicted| (un-normalized absolute residual), computed on
# the training/calibration pool via the saved ensemble (preds_train_full).
# ACI uses np.quantile(cal_scores, 1-alpha_t), alpha_target=0.10 -> ~p90.
# ============================================================================
import json as _json
LAB=["Current","1M","3M","6M"]
rec_prob = train_df["recession_probability"].values  # regime label per training month

def pctile(a,q): return float(np.percentile(a,q)) if len(a) else float("nan")

print("\n"+"="*70); print("P0 — Nonconformity scores by regime (FULL training pool)"); print("="*70)
rows=[]
for h_idx,h in enumerate(LAB):
    y_h=y_train_full[:,h_idx]; p_h=preds_train_full[:,h_idx]
    mask=~np.isnan(y_h)
    scores=np.abs(p_h[mask]-y_h[mask]); reg=rec_prob[mask]
    exp_s=scores[reg<50]; rare_s=scores[reg>=50]
    row=dict(Horizon=h,
        n_exp=int(len(exp_s)), n_rare=int(len(rare_s)),
        exp_mean=round(exp_s.mean(),3), exp_median=round(np.median(exp_s),3),
        exp_p75=round(pctile(exp_s,75),3), exp_p90=round(pctile(exp_s,90),3),
        rare_mean=round(rare_s.mean(),3), rare_median=round(np.median(rare_s),3),
        rare_p75=round(pctile(rare_s,75),3), rare_p90=round(pctile(rare_s,90),3),
        ratio_mean=round(rare_s.mean()/exp_s.mean(),2) if exp_s.mean()>0 else None)
    rows.append(row)
    print(f"\n{h}:  n_exp={row['n_exp']} n_rare={row['n_rare']}")
    print(f"   expansion : mean={row['exp_mean']:7.3f} median={row['exp_median']:7.3f} p75={row['exp_p75']:7.3f} p90={row['exp_p90']:7.3f}")
    print(f"   rare-event: mean={row['rare_mean']:7.3f} median={row['rare_median']:7.3f} p75={row['rare_p75']:7.3f} p90={row['rare_p90']:7.3f}")
    print(f"   rare/exp mean ratio = {row['ratio_mean']}x")
    # overlap check: fraction of expansion scores exceeding rare-event median
    ov=float((exp_s> np.median(rare_s)).mean()*100)
    print(f"   overlap: {ov:.1f}% of expansion scores exceed the rare-event median")
    rows[-1]["pct_exp_above_rare_median"]=round(ov,1)

import pandas as _pd
_pd.DataFrame(rows).to_csv("fix-reg/nonconformity_by_regime_stats.csv",index=False)
print("\nSaved fix-reg/nonconformity_by_regime_stats.csv")

# ── 20% -> 25% transition: what entered, and what p90 did ──
print("\n"+"="*70); print("P0 — 20% -> 25% window transition (per horizon p90 before/after)"); print("="*70)
trans=[]
for frac_a,frac_b in [(0.20,0.25)]:
    csa,_=build_cal_scores_fraction(frac_a); csb,_=build_cal_scores_fraction(frac_b)
    # which rows newly entered (the block between the two window starts)
    n=len(train_df); sa=n-int(n*frac_a); sb=n-int(n*frac_b)
    entered_idx=range(sb,sa)  # older rows added when window grows 20->25
    for h_idx,h in enumerate(LAB):
        p90a=pctile(csa[h_idx],90); p90b=pctile(csb[h_idx],90)
        # newly entered scores at this horizon (with regime)
        yy=y_train_full[sb:sa,h_idx]; pp=preds_train_full[sb:sa,h_idx]; rr=rec_prob[sb:sa]
        m=~np.isnan(yy); new_scores=np.abs(pp[m]-yy[m]); new_reg=rr[m]
        new_rare=new_scores[new_reg>=50]
        print(f"\n{h}: p90(20%)={p90a:.3f} -> p90(25%)={p90b:.3f}  (Δ={p90b-p90a:+.3f})")
        print(f"   newly-entered rows: {int(m.sum())}, of which rare-event={int((new_reg>=50).sum())}")
        if len(new_rare): print(f"   newly-entered rare-event scores: max={new_rare.max():.3f} mean={new_rare.mean():.3f}")
        trans.append(dict(Horizon=h,p90_20=round(p90a,3),p90_25=round(p90b,3),delta=round(p90b-p90a,3),
            new_rows=int(m.sum()),new_rare=int((new_reg>=50).sum()),
            new_rare_max=round(float(new_rare.max()),3) if len(new_rare) else None))
_pd.DataFrame(trans).to_csv("fix-reg/nonconformity_transition_20_25.csv",index=False)
print("\nSaved fix-reg/nonconformity_transition_20_25.csv")

# also dump raw scores (with regime) for P1 figure
raw=[]
for h_idx,h in enumerate(LAB):
    y_h=y_train_full[:,h_idx]; p_h=preds_train_full[:,h_idx]; mask=~np.isnan(y_h)
    s=np.abs(p_h[mask]-y_h[mask]); reg=rec_prob[mask]
    for sc,rg in zip(s,reg):
        raw.append(dict(Horizon=h,score=round(float(sc),4),regime="rare" if rg>=50 else "expansion",rec_prob=round(float(rg),2)))
_pd.DataFrame(raw).to_csv("fix-reg/nonconformity_scores_raw.csv",index=False)
print("Saved fix-reg/nonconformity_scores_raw.csv (for P1 figure)")
