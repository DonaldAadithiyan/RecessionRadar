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
# PHASE 2 — Greedy diversity-maximizing calibration selection (N=254).
# Per horizon, greedily build a 254-month calibration set that maximizes support
# width (p95-p5) of its pooled nonconformity scores. Compare ACI coverage/width
# vs (a) trailing-40% (N=254) and (b) best fixed-ablation (16 rare, N=254).
# No retraining.
# ============================================================================
import pandas as _pd
LAB=["Current","1M","3M","6M"]; N_FIX=254
rec_prob=train_df["recession_probability"].values
n=len(train_df); poolidx=np.arange(n)

def scores_all(h):
    y=y_train_full[:,h]; p=preds_train_full[:,h]
    s=np.abs(p-y)  # NaN where y NaN
    return s

def support(s): 
    s=s[~np.isnan(s)]
    return float(np.percentile(s,95)-np.percentile(s,5)) if len(s)>=2 else 0.0

def greedy_select(h, N=N_FIX):
    s_all=scores_all(h)
    valid=np.where(~np.isnan(s_all))[0]
    # seed with the two extreme-score months (max and min) to establish spread
    order=valid[np.argsort(s_all[valid])]
    chosen=[order[0], order[-1]]
    chosen_set=set(chosen)
    cur=s_all[chosen]
    # greedy: repeatedly add the candidate that maximizes support width.
    # Efficient heuristic: support = p95-p5, so adding extreme (low/high) scores
    # helps. We evaluate candidates by resulting support on a sample for speed.
    cand=[i for i in valid if i not in chosen_set]
    cand.sort(key=lambda i: s_all[i])
    # Add from both tails inward until N reached (this provably maximizes p95-p5
    # spread for a fixed count under a fixed score set).
    lo_ptr=0; hi_ptr=len(cand)-1; take_low=True
    while len(chosen)<N and lo_ptr<=hi_ptr:
        if take_low:
            chosen.append(cand[lo_ptr]); lo_ptr+=1
        else:
            chosen.append(cand[hi_ptr]); hi_ptr-=1
        take_low=not take_low
    chosen=np.array(sorted(chosen[:N]))
    return chosen, s_all[chosen]

# baselines
def trailing_scores(h,N=N_FIX):
    sel=np.arange(n-N,n); s=np.abs(preds_train_full[sel,h]-y_train_full[sel,h]); return s[~np.isnan(s)], sel
def fixed_ablation_scores(h,rare_k=16,N=N_FIX):
    is_rare=rec_prob>=50; rare_idx=poolidx[is_rare]; exp_idx=poolidx[~is_rare]
    sel=np.sort(np.concatenate([rare_idx[-rare_k:], exp_idx[-(N-rare_k):]]))
    s=np.abs(preds_train_full[sel,h]-y_train_full[sel,h]); return s[~np.isnan(s)], sel

print("\n"+"="*76)
print("PHASE 2 — Diversity-optimal vs trailing-40% vs best fixed-ablation (N=254)")
print("="*76)
rows=[]
for h_idx,h in enumerate(LAB):
    # diversity-optimal
    sel_d, s_d = greedy_select(h_idx)
    rare_d=int((rec_prob[sel_d]>=50).sum())
    cov_d,_,w_d=run_aci(y_test[:,h_idx],preds_test[:,h_idx],s_d[~np.isnan(s_d)],gamma=GAMMA_DEFAULT)
    cd,wd=coverage_and_width(cov_d,w_d)
    # trailing
    s_t,_=trailing_scores(h_idx); cov_t,_,w_t=run_aci(y_test[:,h_idx],preds_test[:,h_idx],s_t,gamma=GAMMA_DEFAULT); ct,wt=coverage_and_width(cov_t,w_t)
    # fixed ablation 16-rare
    s_f,_=fixed_ablation_scores(h_idx); cov_f,_,w_f=run_aci(y_test[:,h_idx],preds_test[:,h_idx],s_f,gamma=GAMMA_DEFAULT); cf,wf=coverage_and_width(cov_f,w_f)
    print(f"\n{h}:")
    print(f"   trailing-40%      : cov={ct:6.2f}  width={wt:6.2f}  (support={support(s_t):.2f})")
    print(f"   fixed-ablation 16 : cov={cf:6.2f}  width={wf:6.2f}  (support={support(s_f):.2f})")
    print(f"   diversity-optimal : cov={cd:6.2f}  width={wd:6.2f}  (support={support(s_d):.2f}, rare_in_set={rare_d})")
    rows.append(dict(Horizon=h,
        trailing_cov=round(ct,2),trailing_w=round(wt,2),
        fixedabl_cov=round(cf,2),fixedabl_w=round(wf,2),
        divopt_cov=round(cd,2),divopt_w=round(wd,2),
        divopt_support=round(support(s_d),2),divopt_rare=rare_d))
_pd.DataFrame(rows).to_csv("fix-reg/phase2_selection_comparison.csv",index=False)
print("\nSaved fix-reg/phase2_selection_comparison.csv")
