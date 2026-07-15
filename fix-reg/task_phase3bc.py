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
# PHASE 3b/3c — additional conformal baselines (calibration = trailing N=254).
# 3c: PID-conformal (Angelopoulos et al. 2024) — ACI proportional update + I + D.
# 3b: EVT-tail conformal — fit a Generalized Pareto tail to the calibration
#     scores and set the (1-alpha) quantile from the fitted tail (accessible
#     stand-in for extreme-value CQR, which would need quantile-regression fits;
#     flagged honestly). Both vs pooled ACI. No retraining.
# ============================================================================
import pandas as _pd
from scipy import stats as _st
LAB=["Current","1M","3M","6M"]; N_FIX=254
n=len(train_df); sel=np.arange(n-N_FIX,n)
def cal_s(h): 
    s=np.abs(preds_train_full[sel,h]-y_train_full[sel,h]); return s[~np.isnan(s)]

def run_pid(y_h,p_h,cs,gamma=GAMMA_DEFAULT,at=0.10,ki=0.02,kd=0.01):
    T=len(y_h); a=at; covered=[]; widths=[]; err_int=0.0; prev_err=0.0
    for t in range(T):
        q=np.quantile(cs,np.clip(1-a,0,1)); lo=p_h[t]-q; hi=p_h[t]+q; widths.append(2*q)
        yt=y_h[t]
        if np.isnan(yt): covered.append(np.nan); continue
        miss=1 if (yt<lo or yt>hi) else 0; covered.append(1-miss)
        err=at-miss; err_int+=err; deriv=err-prev_err; prev_err=err
        a=np.clip(a+gamma*err+ki*gamma*err_int+kd*deriv, 0.01, 0.99)  # P + I + D
    return np.array(covered),widths

def evt_quantile(cs, alpha=0.10):
    """Fit GPD to exceedances over the 80th pctile; return the (1-alpha) quantile."""
    u=np.quantile(cs,0.80); exc=cs[cs>u]-u
    if len(exc)<10: return np.quantile(cs,1-alpha)
    try:
        c,loc,scale=_st.genpareto.fit(exc,floc=0)
        p_exceed=len(exc)/len(cs); target=(alpha)/p_exceed
        if target>=1 or target<=0: return np.quantile(cs,1-alpha)
        q_tail=u+_st.genpareto.ppf(1-target,c,loc=0,scale=scale)
        return float(q_tail) if np.isfinite(q_tail) else np.quantile(cs,1-alpha)
    except Exception:
        return np.quantile(cs,1-alpha)

def run_evt_aci(y_h,p_h,cs,gamma=GAMMA_DEFAULT,at=0.10):
    T=len(y_h); a=at; covered=[]; widths=[]
    for t in range(T):
        q=evt_quantile(cs,alpha=np.clip(a,0.001,0.5)); lo=p_h[t]-q; hi=p_h[t]+q; widths.append(2*q)
        yt=y_h[t]
        if np.isnan(yt): covered.append(np.nan); continue
        miss=1 if (yt<lo or yt>hi) else 0; covered.append(1-miss)
        a=np.clip(a+gamma*(at-miss),0.01,0.99)
    return np.array(covered),widths

print("\n"+"="*70); print("PHASE 3b/3c — PID-conformal & EVT-tail conformal vs pooled ACI"); print("="*70)
rows=[]
for h_idx,h in enumerate(LAB):
    cs=cal_s(h_idx)
    cov_p,_,w_p=run_aci(y_test[:,h_idx],preds_test[:,h_idx],cs,gamma=GAMMA_DEFAULT); cp,wp=coverage_and_width(cov_p,w_p)
    cov_pid,w_pid=run_pid(y_test[:,h_idx],preds_test[:,h_idx],cs); v=~np.isnan(cov_pid); cpid=float(np.nanmean(cov_pid[v])*100); wpid=float(np.mean(w_pid))
    cov_e,w_e=run_evt_aci(y_test[:,h_idx],preds_test[:,h_idx],cs); v=~np.isnan(cov_e); ce=float(np.nanmean(cov_e[v])*100); we=float(np.mean(w_e))
    print(f"  {h:8}: pooled cov={cp:6.2f} w={wp:6.2f} | PID cov={cpid:6.2f} w={wpid:6.2f} | EVT cov={ce:6.2f} w={we:6.2f}")
    rows.append(dict(Horizon=h,pooled_cov=round(cp,2),pooled_w=round(wp,2),
        PID_cov=round(cpid,2),PID_w=round(wpid,2),EVT_cov=round(ce,2),EVT_w=round(we,2)))
_pd.DataFrame(rows).to_csv("fix-reg/phase3bc_extra_baselines.csv",index=False)
print("\nSaved fix-reg/phase3bc_extra_baselines.csv")
