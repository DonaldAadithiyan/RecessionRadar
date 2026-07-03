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
# PHASE 1 — Quantify diversity <-> ACI coverage over random fixed-size subsets.
# N=254 fixed. Draw 200 random subsets from the 635-month training pool.
# Diversity stats on the pooled nonconformity scores of the subset:
#   D1 = IQR (p75-p25),  D2 = support width (p95-p5),  D3 = Shannon entropy(20 bins)
# Coverage via same ACI (gamma=0.005, alpha=0.10). Also record rare-event count.
# No model retraining (uses saved preds_train_full).
# ============================================================================
import pandas as _pd
from scipy import stats as _st
LAB=["Current","1M","3M","6M"]; N_FIX=254; N_DRAWS=200
rng=np.random.default_rng(7)
rec_prob=train_df["recession_probability"].values
n_pool=len(train_df); pool=np.arange(n_pool)

def scores_for(sel, h):
    y=y_train_full[sel,h]; p=preds_train_full[sel,h]; m=~np.isnan(y)
    return np.abs(p[m]-y[m])

def shannon(a, bins=20):
    if len(a)<2: return 0.0
    h,_=np.histogram(a,bins=bins); pph=h/h.sum(); pph=pph[pph>0]
    return float(-(pph*np.log(pph)).sum())

rows=[]
for d in range(N_DRAWS):
    sel=np.sort(rng.choice(pool,size=N_FIX,replace=False))
    rare=int((rec_prob[sel]>=50).sum())
    rec=dict(draw=d, rare_count=rare)
    for h_idx,h in enumerate(LAB):
        s=scores_for(sel,h_idx)
        iqr=float(np.percentile(s,75)-np.percentile(s,25))
        supp=float(np.percentile(s,95)-np.percentile(s,5))
        ent=shannon(s)
        covered,_,widths=run_aci(y_test[:,h_idx],preds_test[:,h_idx],s,gamma=GAMMA_DEFAULT)
        cov,w=coverage_and_width(covered,widths)
        rec[f"{h}_IQR"]=round(iqr,4); rec[f"{h}_supp"]=round(supp,4)
        rec[f"{h}_ent"]=round(ent,4); rec[f"{h}_cov"]=round(cov,3); rec[f"{h}_width"]=round(w,3)
    rows.append(rec)

df1=_pd.DataFrame(rows); df1.to_csv("fix-reg/phase1_random_sweep.csv",index=False)
print(f"Saved fix-reg/phase1_random_sweep.csv  ({len(df1)} draws, N={N_FIX})")

print("\n"+"="*74)
print("PHASE 1 — correlations (Spearman rho) of predictors vs coverage, per horizon")
print("="*74)
print(f"{'Horizon':8} {'IQR':>16} {'supp(p95-p5)':>16} {'entropy':>12} {'rare_count':>12}")
summary=[]
for h in LAB:
    cov=df1[f"{h}_cov"].values
    out={}
    for stat,col in [("IQR",f"{h}_IQR"),("supp",f"{h}_supp"),("ent",f"{h}_ent"),("rare","rare_count")]:
        x=df1[col].values
        if np.std(cov)==0 or np.std(x)==0:
            rho=float("nan")
        else:
            rho=_st.spearmanr(x,cov).correlation
        out[stat]=rho
    print(f"{h:8} {out['IQR']:16.3f} {out['supp']:16.3f} {out['ent']:12.3f} {out['rare']:12.3f}")
    summary.append(dict(Horizon=h,rho_IQR=round(out['IQR'],3),rho_supp=round(out['supp'],3),
                        rho_entropy=round(out['ent'],3),rho_rare=round(out['rare'],3),
                        cov_std=round(float(np.std(cov)),3),cov_mean=round(float(np.mean(cov)),2)))
_pd.DataFrame(summary).to_csv("fix-reg/phase1_correlations.csv",index=False)

# Pearson too (for R^2), on the best diversity stat = support width
print("\nPearson r and R^2 (support width vs coverage) and (rare_count vs coverage):")
for h in LAB:
    cov=df1[f"{h}_cov"].values
    if np.std(cov)==0:
        print(f"  {h:8}: coverage CONSTANT (std=0) -> no relationship to detect"); continue
    rs=_st.pearsonr(df1[f"{h}_supp"].values,cov); rr=_st.pearsonr(df1["rare_count"].values,cov)
    print(f"  {h:8}: supp r={rs.statistic:+.3f} R2={rs.statistic**2:.3f} | rare r={rr.statistic:+.3f} R2={rr.statistic**2:.3f}")

# ── Matched-pairs: bin by support width, compare coverage across rare_count ──
print("\n"+"="*74)
print("PHASE 1 — Matched-pairs: within support-width tertiles, does rare_count still matter?")
print("="*74)
mp_rows=[]
for h in LAB:
    cov=df1[f"{h}_cov"].values
    if np.std(cov)==0:
        print(f"\n{h}: coverage constant across draws -> rare_count irrelevant by construction"); 
        mp_rows.append(dict(Horizon=h,note="coverage constant")); continue
    supp=df1[f"{h}_supp"].values; rare=df1["rare_count"].values
    terts=np.quantile(supp,[0,1/3,2/3,1.0])
    print(f"\n{h}: support-width tertiles = [{terts[0]:.2f},{terts[1]:.2f},{terts[2]:.2f},{terts[3]:.2f}]")
    for ti in range(3):
        lo,hi=terts[ti],terts[ti+1]
        m=(supp>=lo)&(supp<=hi if ti==2 else supp<hi)
        if m.sum()<6: continue
        # within this diversity band, correlate rare_count with coverage
        if np.std(rare[m])>0 and np.std(cov[m])>0:
            rho_in=_st.spearmanr(rare[m],cov[m]).correlation
        else: rho_in=float("nan")
        print(f"   tertile {ti+1}: n={int(m.sum())}  cov mean={cov[m].mean():.2f} (sd {cov[m].std():.2f})  "
              f"rare range=[{rare[m].min()},{rare[m].max()}]  rho(rare,cov|band)={rho_in:+.3f}")
        mp_rows.append(dict(Horizon=h,tertile=ti+1,n=int(m.sum()),cov_mean=round(cov[m].mean(),2),
                            cov_sd=round(cov[m].std(),2),rare_min=int(rare[m].min()),rare_max=int(rare[m].max()),
                            rho_rare_cov_within=None if np.isnan(rho_in) else round(rho_in,3)))
_pd.DataFrame(mp_rows).to_csv("fix-reg/phase1_matched_pairs.csv",index=False)
print("\nSaved phase1_correlations.csv, phase1_matched_pairs.csv")
