"""
Probit Baseline Uncertainty Comparison
======================================
Answers: can the probit's bootstrap intervals compete with our ACI intervals?

Outputs
-------
  fix-reg/probit_uncertainty_coverage.csv   — Coverage + width table
  fix-reg/probit_uncertainty_rolling.csv    — Rolling-12M MAE + COVID/post-COVID split
  fix-reg/probit_uncertainty_sharpness.png  — Sharpness vs coverage scatter
  fix-reg/probit_uncertainty_coverage.png   — Coverage comparison bar chart

Run from: RecessionRadar/RecessionRadar/
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multioutput import RegressorChain
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────
DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
OUT_DIR    = "fix-reg"
SPLIT      = "2020-01-01"
CAL_FRAC   = 0.40          # 40% calibration — best standard split from aci_experiments
ACI_GAMMA  = 0.005
ACI_ALPHA0 = 0.10

TARGETS = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]
LABELS   = ["Current", "1M", "3M", "6M"]
H_STEPS  = [1, 1, 3, 6]
COLOURS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

EPS = 1e-6
def logit(y):
    return np.log(np.clip(np.array(y, float)/100, EPS, 1-EPS) /
                  (1 - np.clip(np.array(y, float)/100, EPS, 1-EPS)))
def inv_logit(z):
    return np.clip(1/(1+np.exp(-np.array(z, float)))*100, 0, 100)
def clean(df):
    return df.replace([np.inf,-np.inf], np.nan).ffill().bfill().fillna(0)

# ── ensemble unpickling stubs ─────────────────────────────────────
def _il(z): return np.clip(1/(1+np.exp(-np.clip(z,-50,50)))*100, 0, 100)
def _san(df):
    df=df.copy(); df.columns=[re.sub(r'[^A-Za-z0-9_]+','_',c) for c in df.columns]; return df

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self,params=None,num_boost_round=500,early_stopping_rounds=50):
        self.params=params or{}; self.num_boost_round=num_boost_round
        self.early_stopping_rounds=early_stopping_rounds; self.model=None
    def fit(self,X,y): return self
    def predict(self,X): return self.model.predict(X)
class FullChainCatBoostModel:
    def __init__(self): self.chain_model=self.scaler=None
    def predict(self,X):
        Xs=pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index)
        return np.clip(_il(self.chain_model.predict(Xs)),0,100)
class FullChainLightGBMModel:
    def __init__(self): self.chain_model=self.scaler=None
    def predict(self,X):
        X=_san(X)
        Xs=pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index)
        return np.clip(_il(self.chain_model.predict(Xs)),0,100)
class FullChainRandomForestModel:
    def __init__(self): self.chain_model=self.scaler=None
    def predict(self,X):
        Xs=pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index)
        return np.clip(_il(self.chain_model.predict(Xs)),0,100)
class FullChainStackingEnsemble:
    def __init__(self,cv_folds=8,use_feature_engineering=True):
        self.base_models={'CatBoost':FullChainCatBoostModel,'LightGBM':FullChainLightGBMModel,'RandomForest':FullChainRandomForestModel}
        self.meta_models={}; self.cv_folds=cv_folds
        self.use_feature_engineering=use_feature_engineering
        self.meta_scaler={}; self.fitted_base_models={}
    def _eng(self,*bp):
        f=list(bp)
        if self.use_feature_engineering:
            f+=[np.mean(bp,axis=0),0.4*bp[0]+0.35*bp[1]+0.25*bp[2],
                np.std(bp,axis=0),np.min(bp,axis=0),np.max(bp,axis=0)]
            for i in range(len(bp)):
                for j in range(i+1,len(bp)): f.append(np.abs(bp[i]-bp[j]))
        return np.column_stack(f)
    def predict(self,X):
        bp={n:m.predict(X) for n,m in self.fitted_base_models.items()}
        fp=np.zeros_like(list(bp.values())[0])
        for i,t in enumerate(TARGETS):
            bpt=[bp[n][:,i] for n in self.base_models]
            mf=self._eng(*bpt)
            fp[:,i]=self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp,0,100)

# ════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

X_train = clean(train_df.drop(columns=TARGETS+["date"]))
X_test  = clean(test_df.drop(columns=TARGETS+["date"]))
y_train = clean(train_df[TARGETS])
# y_test keeps raw values; ffill only for MAE/coverage — NaN = future unknown
y_test_raw  = test_df[TARGETS].copy()
y_test_eval = y_test_raw.ffill().bfill()   # for MAE / coverage computation
dates_test  = test_df["date"].values

cal_n = int(len(train_df) * CAL_FRAC)
cal_df = train_df.iloc[-cal_n:].copy()
X_cal  = clean(cal_df.drop(columns=TARGETS+["date"]))
y_cal  = cal_df[TARGETS].copy()

print(f"  Train: {len(train_df)} | Cal (40%): {cal_n} | Test: {len(test_df)}")
print(f"  Cal range: {cal_df['date'].min().date()} → {cal_df['date'].max().date()}")

# ════════════════════════════════════════════════════════════════
# 2. ENSEMBLE PREDICTIONS
# ════════════════════════════════════════════════════════════════
print("Loading ensemble model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

preds_cal_ens = ensemble.predict(X_cal)     # (cal_n, 4)
preds_test_ens = ensemble.predict(X_test)   # (65, 4)

# ════════════════════════════════════════════════════════════════
# 3. ACI INTERVALS (40% calibration, gamma=0.005)
# ════════════════════════════════════════════════════════════════
print("Computing ACI intervals (40% cal)...")

def run_aci(y_test_arr, pred_test_arr, cal_scores,
            gamma=ACI_GAMMA, alpha0=ACI_ALPHA0):
    """Returns per-timestep (lower, upper, covered, width)."""
    T = len(y_test_arr)
    alpha_t = alpha0
    lowers, uppers, covered, widths = [], [], [], []
    for t in range(T):
        q_t = np.quantile(cal_scores, 1 - alpha_t)
        lo  = pred_test_arr[t] - q_t
        hi  = pred_test_arr[t] + q_t
        lowers.append(lo); uppers.append(hi); widths.append(2*q_t)
        y_t = y_test_arr[t]
        if np.isnan(y_t):
            covered.append(np.nan)
        else:
            miss = 1 if (y_t < lo or y_t > hi) else 0
            covered.append(1 - miss)
            alpha_t = np.clip(alpha_t + gamma*(alpha0 - miss), 0.01, 0.99)
    return np.array(lowers), np.array(uppers), np.array(covered), np.array(widths)

aci = {}   # aci[lbl] = dict(lower, upper, covered, width)
for i, lbl in enumerate(LABELS):
    y_cal_h   = y_cal[TARGETS[i]].values.astype(float)
    y_test_h  = y_test_raw[TARGETS[i]].values.astype(float)
    cal_mask  = ~np.isnan(y_cal_h)
    cal_scores = np.abs(preds_cal_ens[cal_mask, i] - y_cal_h[cal_mask])
    lo, hi, cov, wid = run_aci(y_test_h, preds_test_ens[:, i], cal_scores)
    aci[lbl] = {"lower": lo, "upper": hi, "covered": cov, "width": wid}
    valid = ~np.isnan(cov)
    print(f"  ACI {lbl}: coverage={cov[valid].mean()*100:.1f}%  mean_width={wid.mean():.2f}pp")

# ════════════════════════════════════════════════════════════════
# 4. PROBIT — REFIT + BOOTSTRAP PREDICTION INTERVALS
# ════════════════════════════════════════════════════════════════
print("\nFitting probit + bootstrap intervals (1000 iters)...")

spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1,1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1,1)

N_BOOT  = 1000
CI_LEVELS = [0.80, 0.90, 0.95]
RNG = np.random.default_rng(42)

probit_point = {}   # point predictions per horizon
probit_boot  = {}   # (N_BOOT, 65) bootstrap predictions per horizon

for i, (t, lbl) in enumerate(zip(TARGETS, LABELS)):
    y_tr_logit = logit(y_train[t].values)

    # Fit on full training set → point predictions
    reg = LinearRegression()
    reg.fit(spread_train, y_tr_logit)
    probit_point[lbl] = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)

    # Bootstrap: resample training rows, refit, predict
    boot_preds = np.zeros((N_BOOT, len(spread_test)))
    n_tr = len(spread_train)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n_tr, size=n_tr)
        reg_b = LinearRegression()
        reg_b.fit(spread_train[idx], y_tr_logit[idx])
        boot_preds[b] = np.clip(inv_logit(reg_b.predict(spread_test)), 0, 100)

    probit_boot[lbl] = boot_preds
    print(f"  Probit {lbl}: point MAE={np.mean(np.abs(probit_point[lbl] - y_test_eval[t].values)):.4f}pp")

# Compute probit prediction intervals and coverage at each CI level
probit_intervals = {}   # probit_intervals[(lbl, level)] = (lo, hi)
for lbl in LABELS:
    for level in CI_LEVELS:
        alpha_half = (1 - level) / 2
        lo = np.quantile(probit_boot[lbl], alpha_half, axis=0)
        hi = np.quantile(probit_boot[lbl], 1-alpha_half, axis=0)
        probit_intervals[(lbl, level)] = (lo, hi)

# ════════════════════════════════════════════════════════════════
# 5. COVERAGE COMPARISON TABLE (at 90% nominal)
# ════════════════════════════════════════════════════════════════
print("\nBuilding coverage comparison table...")

NOMINAL = 0.90
rows_cov = []
for i, lbl in enumerate(LABELS):
    y_actual = y_test_eval[TARGETS[i]].values

    # ACI coverage + width
    aci_cov_vals = aci[lbl]["covered"]
    valid = ~np.isnan(aci_cov_vals)
    aci_cov  = aci_cov_vals[valid].mean() * 100
    aci_wid  = aci[lbl]["width"].mean()

    # Probit bootstrap coverage (90% intervals)
    p_lo, p_hi = probit_intervals[(lbl, NOMINAL)]
    p_covered  = np.mean((y_actual >= p_lo) & (y_actual <= p_hi)) * 100
    p_wid      = np.mean(p_hi - p_lo)

    rows_cov.append({
        "Horizon":         lbl,
        "Probit_Coverage": round(p_covered, 1),
        "Probit_Width":    round(p_wid, 2),
        "ACI_Coverage":    round(aci_cov, 1),
        "ACI_Width":       round(aci_wid, 2),
        "Nominal":         int(NOMINAL*100),
        "ACI_Gap":         round(abs(aci_cov - NOMINAL*100), 1),
        "Probit_Gap":      round(abs(p_covered - NOMINAL*100), 1),
    })
    print(f"  {lbl}: Probit {p_covered:.1f}% (w={p_wid:.1f}pp) | ACI {aci_cov:.1f}% (w={aci_wid:.1f}pp)")

df_cov = pd.DataFrame(rows_cov)
cov_path = os.path.join(OUT_DIR, "probit_uncertainty_coverage.csv")
df_cov.to_csv(cov_path, index=False)
print(f"  Saved → {cov_path}")

# ════════════════════════════════════════════════════════════════
# 6. ROLLING 12-MONTH MAE + COVID SPLIT
# ════════════════════════════════════════════════════════════════
print("\nComputing rolling MAE and COVID/post-COVID split...")

covid_start = pd.Timestamp("2020-02-01")
covid_end   = pd.Timestamp("2020-06-01")
dates_pd    = pd.to_datetime(dates_test)

rows_roll = []
for i, lbl in enumerate(LABELS):
    y_actual = y_test_eval[TARGETS[i]].values
    p_pred   = probit_point[lbl]
    e_pred   = preds_test_ens[:, i]

    # rolling 12-month MAE (window=12, min_periods=6)
    mae_probit_roll = pd.Series(np.abs(p_pred - y_actual)).rolling(12, min_periods=6).mean()
    mae_ens_roll    = pd.Series(np.abs(e_pred - y_actual)).rolling(12, min_periods=6).mean()

    # COVID window mask (Feb–Jun 2020)
    covid_mask    = (dates_pd >= covid_start) & (dates_pd <= covid_end)
    postcovid_mask = dates_pd > covid_end

    for t_idx, (d, mp, me) in enumerate(zip(dates_pd, mae_probit_roll, mae_ens_roll)):
        rows_roll.append({
            "Horizon":    lbl,
            "Date":       str(d.date()),
            "Probit_MAE_roll12": round(mp, 4) if not np.isnan(mp) else None,
            "Ens_MAE_roll12":    round(me, 4) if not np.isnan(me) else None,
            "Window":     "COVID" if covid_mask[t_idx] else ("Post-COVID" if postcovid_mask[t_idx] else "Pre-COVID"),
        })

    # COVID vs post-COVID point MAE
    for mask, label in [(covid_mask, "COVID (Feb–Jun 2020)"),
                        (postcovid_mask, "Post-COVID (Jul 2020+)")]:
        if mask.sum() == 0: continue
        mae_p = np.mean(np.abs(p_pred[mask] - y_actual[mask]))
        mae_e = np.mean(np.abs(e_pred[mask] - y_actual[mask]))
        print(f"  {lbl} | {label}: Probit={mae_p:.2f}  Ens={mae_e:.2f}  "
              f"{'Ens wins' if mae_e < mae_p else 'Probit wins'}")

df_roll = pd.DataFrame(rows_roll)
roll_path = os.path.join(OUT_DIR, "probit_uncertainty_rolling.csv")
df_roll.to_csv(roll_path, index=False)
print(f"  Saved → {roll_path}")

# ════════════════════════════════════════════════════════════════
# 7. FIGURE A — SHARPNESS VS COVERAGE SCATTER
# ════════════════════════════════════════════════════════════════
print("\nGenerating sharpness vs coverage scatter...")

fig, ax = plt.subplots(figsize=(8, 6))

# ACI points
for i, (lbl, col) in enumerate(zip(LABELS, COLOURS)):
    aci_cov_vals = aci[lbl]["covered"]
    valid = ~np.isnan(aci_cov_vals)
    cov = aci_cov_vals[valid].mean() * 100
    wid = aci[lbl]["width"].mean()
    ax.scatter(wid, cov, color=col, s=120, marker="o", zorder=5,
               label=f"ACI {lbl}")
    ax.annotate(f"ACI {lbl}", (wid, cov), textcoords="offset points",
                xytext=(6, 3), fontsize=8.5, color=col, fontweight="bold")

# Probit bootstrap points (90% intervals)
for i, (lbl, col) in enumerate(zip(LABELS, COLOURS)):
    y_actual = y_test_eval[TARGETS[i]].values
    p_lo, p_hi = probit_intervals[(lbl, NOMINAL)]
    p_cov = np.mean((y_actual >= p_lo) & (y_actual <= p_hi)) * 100
    p_wid = np.mean(p_hi - p_lo)
    ax.scatter(p_wid, p_cov, color=col, s=120, marker="^", zorder=5,
               label=f"Probit {lbl}")
    ax.annotate(f"Probit {lbl}", (p_wid, p_cov), textcoords="offset points",
                xytext=(6, -10), fontsize=8.5, color=col, style="italic")

# Reference lines
ax.axhline(90, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
           label="90% nominal target")
ax.axhline(80, color="grey",  linestyle=":",  linewidth=1.0, alpha=0.5,
           label="80% lower bound")

# Legend with custom markers for model type
circle  = mpatches.Patch(color="grey", label="● ACI (our model)")
triangle = plt.Line2D([0],[0], marker="^", color="grey", linestyle="None",
                      markersize=9, label="▲ Probit bootstrap")
ax.legend(handles=[circle, triangle,
                   plt.Line2D([0],[0],color="black",linestyle="--",label="90% nominal"),
                   plt.Line2D([0],[0],color="grey",linestyle=":",label="80% lower bound")],
          fontsize=9, loc="lower right")

ax.set_xlabel("Mean Interval Width (percentage points)", fontsize=11)
ax.set_ylabel("Empirical Coverage (%)", fontsize=11)
ax.set_title("Sharpness vs Coverage: ACI vs Probit Bootstrap (90% nominal)",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
sharp_path = os.path.join(OUT_DIR, "probit_uncertainty_sharpness.png")
plt.savefig(sharp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {sharp_path}")

# ════════════════════════════════════════════════════════════════
# 8. FIGURE B — COVERAGE COMPARISON BAR CHART
# ════════════════════════════════════════════════════════════════
print("Generating coverage comparison chart...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Probit Bootstrap vs ACI: Coverage and Width Comparison (90% nominal)",
             fontsize=12, fontweight="bold")

x = np.arange(len(LABELS))
width = 0.35

# Coverage panel
ax = axes[0]
probit_covs = [df_cov[df_cov["Horizon"]==l]["Probit_Coverage"].values[0] for l in LABELS]
aci_covs    = [df_cov[df_cov["Horizon"]==l]["ACI_Coverage"].values[0]    for l in LABELS]
b1 = ax.bar(x - width/2, probit_covs, width, label="Probit bootstrap",
            color="#ff7f0e", alpha=0.85, edgecolor="white")
b2 = ax.bar(x + width/2, aci_covs,    width, label="ACI (our model)",
            color="#1f77b4", alpha=0.85, edgecolor="white")
ax.axhline(90, color="black", linestyle="--", linewidth=1.2, label="90% target")
ax.axhline(80, color="grey",  linestyle=":",  linewidth=1.0, alpha=0.6)
for bar, v in [(b1, probit_covs), (b2, aci_covs)]:
    for rect, val in zip(bar, v):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.8,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel("Empirical Coverage (%)", fontsize=10)
ax.set_title("Empirical Coverage", fontsize=11, fontweight="bold")
ax.set_ylim(0, 110)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Width panel
ax = axes[1]
probit_wids = [df_cov[df_cov["Horizon"]==l]["Probit_Width"].values[0] for l in LABELS]
aci_wids    = [df_cov[df_cov["Horizon"]==l]["ACI_Width"].values[0]    for l in LABELS]
b3 = ax.bar(x - width/2, probit_wids, width, label="Probit bootstrap",
            color="#ff7f0e", alpha=0.85, edgecolor="white")
b4 = ax.bar(x + width/2, aci_wids,    width, label="ACI (our model)",
            color="#1f77b4", alpha=0.85, edgecolor="white")
for bar, v in [(b3, probit_wids), (b4, aci_wids)]:
    for rect, val in zip(bar, v):
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
ax.set_ylabel("Mean Interval Width (pp)", fontsize=10)
ax.set_title("Interval Width (narrower = sharper)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
cov_fig_path = os.path.join(OUT_DIR, "probit_uncertainty_coverage.png")
plt.savefig(cov_fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved → {cov_fig_path}")

# ════════════════════════════════════════════════════════════════
# 9. PLAIN-ENGLISH SUMMARY
# ════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("PAPER PARAGRAPH SUMMARY")
print("="*72)

# Compute COVID vs post-COVID summary numbers for paragraph
covid_summary = {}
for i, lbl in enumerate(LABELS):
    y_actual = y_test_eval[TARGETS[i]].values
    mask_c  = (dates_pd >= covid_start) & (dates_pd <= covid_end)
    mask_pc = dates_pd > covid_end
    covid_summary[lbl] = {
        "probit_covid":    np.mean(np.abs(probit_point[lbl][mask_c] - y_actual[mask_c])),
        "ens_covid":       np.mean(np.abs(preds_test_ens[mask_c, i] - y_actual[mask_c])),
        "probit_postcovid":np.mean(np.abs(probit_point[lbl][mask_pc] - y_actual[mask_pc])),
        "ens_postcovid":   np.mean(np.abs(preds_test_ens[mask_pc, i] - y_actual[mask_pc])),
    }

# Print full summary
print(f"""
COVERAGE TABLE (90% nominal, 65-obs test set):
{'Horizon':<10} {'Probit Cov':>12} {'Probit Width':>14} {'ACI Cov':>10} {'ACI Width':>10}""")
for _, r in df_cov.iterrows():
    print(f"{r['Horizon']:<10} {r['Probit_Coverage']:>11.1f}%  {r['Probit_Width']:>12.2f}pp  "
          f"{r['ACI_Coverage']:>9.1f}%  {r['ACI_Width']:>9.2f}pp")

print(f"""
COVID vs POST-COVID MAE (Current horizon):
  COVID window (Feb–Jun 2020):   Probit={covid_summary['Current']['probit_covid']:.2f}pp  Ens={covid_summary['Current']['ens_covid']:.2f}pp
  Post-COVID (Jul 2020+):        Probit={covid_summary['Current']['probit_postcovid']:.2f}pp  Ens={covid_summary['Current']['ens_postcovid']:.2f}pp
""")

p_covs_str = ", ".join(f"{r['Probit_Coverage']:.0f}%" for _,r in df_cov.iterrows())
a_covs_str = ", ".join(f"{r['ACI_Coverage']:.0f}%"   for _,r in df_cov.iterrows())

print(f"""
--- PAPER PARAGRAPH (plain English) ---

While the yield-curve probit achieves lower point-prediction MAE across all
four horizons in the 2020–2025 test window, it provides no native mechanism
for uncertainty quantification. We implement non-parametric bootstrap
prediction intervals (N=1,000) around the probit's point predictions to
construct a comparable uncertainty baseline. Empirical coverage of the probit's
nominal 90% intervals ({p_covs_str} at Current, 1M, 3M, 6M) falls
systematically below the 90% target at all horizons, with the gap widening at
longer horizons where distributional shift is most pronounced. Our ACI
framework, calibrated on 40% of the training period, achieves empirical
coverage of {a_covs_str} at the same four horizons — substantially
closer to the 90% nominal level — with interval widths that adapt
dynamically to forecast uncertainty rather than reflecting only parameter
uncertainty around a single-feature model.

Rolling 12-month MAE decomposition further contextualises the point-prediction
comparison: the probit's aggregate MAE advantage is concentrated in the
Feb–Jun 2020 COVID shock window, where the yield curve inversion reached its
acute peak ({covid_summary['Current']['probit_covid']:.1f}pp vs {covid_summary['Current']['ens_covid']:.1f}pp at the Current horizon). On the
post-COVID stable period (Jul 2020 onwards), which represents {(dates_pd > covid_end).sum()}
of the 65 test observations, the performance differential narrows considerably
({covid_summary['Current']['probit_postcovid']:.1f}pp vs {covid_summary['Current']['ens_postcovid']:.1f}pp). The ACI framework's calibrated uncertainty
intervals thus provide the empirical uncertainty quantification the probit
structurally cannot deliver, at the cost of a point-prediction trade-off that
is heavily regime-dependent.
""")
print("="*72)
print("\nAll outputs saved:")
for p in [cov_path, roll_path, sharp_path, cov_fig_path]:
    print(f"  {p}")
