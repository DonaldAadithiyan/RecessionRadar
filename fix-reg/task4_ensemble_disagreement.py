"""
TASK 4 — Ensemble Disagreement as Uncertainty Signal
Computes standard deviation of CatBoost / LightGBM / RandomForest predictions
at each test observation as a model-internal uncertainty signal.
Compares with ACI interval widths from Task 1 via Pearson correlation.
Produces Figure E time series plot.
"""

import os, re, pickle, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

# ── Class definitions for unpickling ─────────────────────────
recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS = ["Current", "1M", "3M", "6M"]

eps = 1e-8
def safe_logit(y):
    return np.log(np.clip(np.clip(y,0,100)/100, eps, 1-eps) /
                  (1 - np.clip(np.clip(y,0,100)/100, eps, 1-eps)))
def safe_inv_logit(z):
    return np.clip(1/(1+np.exp(-np.clip(z,-50,50)))*100, 0, 100)
def sanitize_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]+','_',c) for c in df.columns]
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

# ── Config ───────────────────────────────────────────────────
DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
OUT_DIR    = "fix-reg/task4_outputs"
SPLIT      = "2020-01-01"
ACI_DIR    = "fix-reg/task1_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def clean(d):
    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

test_df = df[df["date"] >= SPLIT].copy()
X_test  = clean(test_df.drop(columns=recession_targets + ["date"]))
dates   = pd.to_datetime(test_df["date"].values)
print(f"  Test rows: {len(X_test)}, Date range: {dates[0].date()} → {dates[-1].date()}")

# ── Load model and get per-base-model predictions ─────────────
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

print("Getting predictions from each base model...")
base_preds = {}
for name, model_obj in ensemble.fitted_base_models.items():
    preds_i = model_obj.predict(X_test)   # (n_test, 4)
    base_preds[name] = preds_i
    print(f"  {name}: shape {preds_i.shape}, "
          f"6M mean={preds_i[:,3].mean():.2f}%")

# Stack into array: (3, n_test, 4)
pred_stack = np.array([base_preds[n] for n in ["CatBoost", "LightGBM", "RandomForest"]])

# Disagreement = std across 3 base models at each (test, horizon)
disagreement = pred_stack.std(axis=0)   # (n_test, 4)

print(f"\nDisagreement array shape: {disagreement.shape}")
print("Mean disagreement by horizon:")
for h_idx, h in enumerate(LABELS):
    print(f"  {h}: {disagreement[:,h_idx].mean():.4f}pp")

# ── Load ACI interval widths from Task 1 ─────────────────────
print("\nLoading ACI interval widths from Task 1...")
task1_table = pd.read_csv(f"{ACI_DIR}/table5_aci_numbers.csv")
aci_width_6m_mean = float(task1_table.loc[task1_table["Horizon"]=="6M", "Mean_Width"].values[0])
print(f"  ACI mean width for 6M: {aci_width_6m_mean:.4f}pp")

# Load alpha trajectory to reconstruct per-step ACI widths for 6M
with open(f"{ACI_DIR}/alpha_trajectory_6m.json") as f:
    alpha_info = json.load(f)

# Reload data to recompute ACI widths per time step
# (We stored alpha trajectory so we can reconstruct widths)
# Reload calibration scores (20% of training data)
train_df  = df[df["date"] < SPLIT].copy()
cal_size  = int(len(train_df) * 0.20)
cal_df    = train_df.iloc[-cal_size:].copy()
X_cal     = clean(cal_df.drop(columns=recession_targets + ["date"]))
y_cal     = cal_df[recession_targets].values

preds_cal  = ensemble.predict(X_cal)
preds_test = ensemble.predict(X_test)
cal_scores_6m = np.abs(preds_cal[:,3] - y_cal[:,3])
cal_scores_6m = cal_scores_6m[~np.isnan(cal_scores_6m)]

alphas_6m = np.array(alpha_info["trajectory"])
aci_widths_6m_ts = np.array([
    2 * np.quantile(cal_scores_6m, np.clip(1 - a, 0, 1))
    for a in alphas_6m
])

print(f"  ACI width time series: min={aci_widths_6m_ts.min():.4f}, "
      f"max={aci_widths_6m_ts.max():.4f}, mean={aci_widths_6m_ts.mean():.4f}")

# ── Period averages for 6M disagreement ──────────────────────
disagree_6m = disagreement[:, 3]   # (n_test,)

# Non-recession stable periods: exclude COVID (Q1–Q2 2020) and Fed tightening (Mar 2022–Dec 2023)
covid_mask    = (dates >= "2020-01-01") & (dates <= "2020-06-30")
tighten_mask  = (dates >= "2022-03-01") & (dates <= "2023-12-31")
stable_mask   = ~covid_mask & ~tighten_mask

avg_stable   = float(disagree_6m[stable_mask].mean())
avg_covid    = float(disagree_6m[covid_mask].mean())
avg_tighten  = float(disagree_6m[tighten_mask].mean())
peak_val     = float(disagree_6m.max())
peak_date    = pd.Timestamp(dates[disagree_6m.argmax()]).strftime("%Y-%m-%d")

print(f"\n6M Disagreement averages:")
print(f"  Non-recession stable:  {avg_stable:.4f}pp")
print(f"  COVID (Q1–Q2 2020):    {avg_covid:.4f}pp")
print(f"  Fed tightening cycle:  {avg_tighten:.4f}pp")
print(f"  Peak:                  {peak_val:.4f}pp on {peak_date}")

# ── Same for 1M and 3M ────────────────────────────────────────
horizon_avgs = {}
for h_idx, h in enumerate(LABELS):
    d_h = disagreement[:, h_idx]
    horizon_avgs[h] = {
        "stable":   round(float(d_h[stable_mask].mean()), 4),
        "covid":    round(float(d_h[covid_mask].mean()), 4),
        "tighten":  round(float(d_h[tighten_mask].mean()), 4),
        "mean":     round(float(d_h.mean()), 4),
        "peak_val": round(float(d_h.max()), 4),
        "peak_date": pd.Timestamp(dates[d_h.argmax()]).strftime("%Y-%m-%d"),
    }
print("\nDisagreement by horizon (stable / COVID / tightening):")
for h in LABELS:
    a = horizon_avgs[h]
    print(f"  {h:8s}: stable={a['stable']:.4f}  covid={a['covid']:.4f}  "
          f"tighten={a['tighten']:.4f}  peak={a['peak_val']:.4f} ({a['peak_date']})")

# ── Pearson correlation: disagreement vs ACI width (6M) ──────
# Both are length 65 (n_test)
corr, pval = stats.pearsonr(disagree_6m, aci_widths_6m_ts)
print(f"\nPearson correlation (6M disagreement vs ACI interval width): r={corr:.4f}, p={pval:.4f}")
if corr > 0.6:
    print("  → BONUS: r > 0.6! Two independent uncertainty signals converge.")

# ── Save all numbers ──────────────────────────────────────────
summary = {
    "6M_avg_stable":    round(avg_stable, 4),
    "6M_avg_covid":     round(avg_covid, 4),
    "6M_avg_tighten":   round(avg_tighten, 4),
    "6M_peak_val":      round(peak_val, 4),
    "6M_peak_date":     peak_date,
    "pearson_r_6m":     round(float(corr), 4),
    "pearson_p_6m":     round(float(pval), 4),
    "by_horizon":       horizon_avgs,
}
with open(f"{OUT_DIR}/task4_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved → {OUT_DIR}/task4_summary.json")

# Save disagreement time series as CSV
disagree_df = pd.DataFrame({
    "date": [d.strftime("%Y-%m-%d") for d in dates],
    "disagreement_Current": disagreement[:,0],
    "disagreement_1M":      disagreement[:,1],
    "disagreement_3M":      disagreement[:,2],
    "disagreement_6M":      disagreement[:,3],
    "aci_width_6M":         aci_widths_6m_ts,
})
disagree_df.to_csv(f"{OUT_DIR}/disagreement_time_series.csv", index=False)
print(f"Disagreement time series → {OUT_DIR}/disagreement_time_series.csv")

# ── Figure E — 6M disagreement time series ───────────────────
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(dates, disagree_6m, color="#1a1a2e", linewidth=2.0,
        label="Ensemble disagreement (6M)", zorder=5)
ax.fill_between(dates, 0, disagree_6m, alpha=0.15, color="#1a1a2e", zorder=2)

# Highlight periods
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"),
           alpha=0.08, color="#c62828", label="COVID recession (Q1–Q2 2020)")
ax.axvspan(pd.Timestamp("2022-03-01"), pd.Timestamp("2023-12-31"),
           alpha=0.08, color="#e65100", label="Fed tightening (Mar 2022–Dec 2023)")

# Event lines
covid_date   = pd.Timestamp("2020-03-01")
tighten_date = pd.Timestamp("2022-03-01")
ax.axvline(covid_date,   color="#c62828", linewidth=2, linestyle="--", alpha=0.9)
ax.axvline(tighten_date, color="#e65100", linewidth=2, linestyle="--", alpha=0.9)

ylim = ax.get_ylim()
ymax = max(disagree_6m) * 1.15
ax.set_ylim(0, ymax)
ax.text(covid_date,   ymax*0.95, " Mar 2020\n COVID",
        color="#c62828", fontsize=9, va="top", fontweight="bold")
ax.text(tighten_date, ymax*0.95, " Mar 2022\n Fed Tightening",
        color="#e65100", fontsize=9, va="top", fontweight="bold")

# Annotate peak
peak_ts = pd.Timestamp(peak_date)
ax.annotate(f"Peak: {peak_val:.2f}pp\n{peak_date}",
            xy=(peak_ts, peak_val),
            xytext=(peak_ts, peak_val * 0.7),
            fontsize=8.5, color="#1a1a2e",
            arrowprops=dict(arrowstyle="->", color="#1a1a2e"),
            ha="center")

ax.set_xlabel("Date", fontsize=11)
ax.set_ylabel("Ensemble Disagreement (σ across 3 models, pp)", fontsize=11)
ax.set_title(
    f"Figure E — Ensemble Disagreement: 6M Horizon (2020–2025)\n"
    f"Pearson correlation with ACI interval width: r = {corr:.3f}",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=9, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/figure_E_disagreement_6m.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure E saved → {OUT_DIR}/figure_E_disagreement_6m.png")

# ── Final summary print ───────────────────────────────────────
print("\n" + "="*60)
print("TASK 4 COMPLETE")
print("="*60)
print(f"6M Disagreement: stable={avg_stable:.4f}  covid={avg_covid:.4f}  tighten={avg_tighten:.4f}")
print(f"Peak: {peak_val:.4f}pp on {peak_date}")
print(f"Pearson r (disagreement vs ACI width, 6M): {corr:.4f}")
print("\nAll four horizons:")
for h in LABELS:
    a = horizon_avgs[h]
    print(f"  {h}: mean={a['mean']:.4f}  stable={a['stable']:.4f}  "
          f"covid={a['covid']:.4f}  tighten={a['tighten']:.4f}")
