"""
Tasks 2.1, 4.1, 4.2:
  2.1 – ACI on probit at 20% + 40% calibration + nonconformity score distributions
  4.1 – γ sweep [0.001 … 0.2] at 40% + 100% cal on 6M horizon
  4.2 – KS test across horizon pairs + horizon-specific vs shared calibration

Run from: RecessionRadar/RecessionRadar/
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multioutput import RegressorChain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import DMatrix, train as xgb_train
warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
SPLIT      = "2020-01-01"
OUT        = "fix-reg"

TARGETS = [
    "recession_probability",
    "1_month_recession_probability",
    "3_month_recession_probability",
    "6_month_recession_probability",
]
LABELS  = ["Current", "1-Month", "3-Month", "6-Month"]
H_STEPS = [1, 1, 3, 6]

# ── unpickling stubs ──────────────────────────────────────────────────────────
def _inv_logit(z):
    return np.clip(1 / (1 + np.exp(-np.clip(z, -50, 50))) * 100, 0, 100)
def _san(df):
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
    return df

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        self.params = params or {}; self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds; self.model = None
    def fit(self, X, y): return self
    def predict(self, X): return self.model.predict(X)

class FullChainCatBoostModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainLightGBMModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        X = _san(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainRandomForestModel:
    def __init__(self): self.chain_model = self.scaler = None
    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(_inv_logit(self.chain_model.predict(Xs)), 0, 100)

class FullChainStackingEnsemble:
    def __init__(self, cv_folds=8, use_feature_engineering=True):
        self.base_models = {"CatBoost": FullChainCatBoostModel,
                            "LightGBM": FullChainLightGBMModel,
                            "RandomForest": FullChainRandomForestModel}
        self.meta_models = {}; self.cv_folds = cv_folds
        self.use_feature_engineering = use_feature_engineering
        self.meta_scaler = {}; self.fitted_base_models = {}
    def _eng(self, *bp):
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
        for i, t in enumerate(TARGETS):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf  = self._eng(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)

# ── helpers ───────────────────────────────────────────────────────────────────
EPS = 1e-6
def logit(y):
    y_s = np.clip(np.array(y, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(y_s / (1 - y_s))
def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-np.array(z, dtype=float))) * 100.0
def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

def run_aci(cal_scores, test_preds, test_actual, gamma=0.02, alpha_target=0.10):
    """ACI (Gibbs & Candès 2021). Returns coverage, avg_width, alpha_trace."""
    alpha_t = alpha_target
    coverage_list = []
    width_list    = []
    alpha_trace   = []

    for i in range(len(test_preds)):
        scores_so_far = cal_scores if i == 0 else cal_scores  # static cal set
        q_t = np.quantile(scores_so_far, 1 - alpha_t)
        lo  = test_preds[i] - q_t
        hi  = test_preds[i] + q_t
        covered = (lo <= test_actual[i] <= hi)
        coverage_list.append(int(covered))
        width_list.append(2 * q_t)
        alpha_trace.append(alpha_t)
        miss = 0 if covered else 1
        alpha_t = np.clip(alpha_t + gamma * (alpha_target - miss), 0.01, 0.99)

    return np.mean(coverage_list)*100, np.mean(width_list), alpha_trace

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

train_df = df[df["date"] < SPLIT].copy()
test_df  = df[df["date"] >= SPLIT].copy()

X_train = clean(train_df.drop(columns=TARGETS + ["date"]))
X_test  = clean(test_df.drop(columns=TARGETS + ["date"]))
y_train = clean(train_df[TARGETS])
y_test  = clean(test_df[TARGETS])

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ── load ensemble ─────────────────────────────────────────────────────────────
print("Loading ensemble...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)
ens_preds_raw = ensemble.predict(X_test)
ens_preds = {lbl: ens_preds_raw[:, i] for i, lbl in enumerate(LABELS)}

# ── build probit ──────────────────────────────────────────────────────────────
spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1, 1)
probit_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    reg = LinearRegression()
    reg.fit(spread_train, logit(y_train[t].values))
    probit_preds[lbl] = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)

# ── Task 2.1: ACI on Probit ────────────────────────────────────────────────────
print("\n=== TASK 2.1: ACI on Probit (20% + 40% cal splits) ===")

GAMMA_PROBIT = 0.02
ALPHA_TARGET = 0.10
CAL_FRACS    = [0.20, 0.40]
rows_21      = []

for cal_frac in CAL_FRACS:
    n_test  = len(X_test)
    n_cal   = int(n_test * cal_frac)
    n_eval  = n_test - n_cal
    print(f"\n  cal_frac={cal_frac:.0%}  n_cal={n_cal}  n_eval={n_eval}")

    for lbl, t in zip(LABELS, TARGETS):
        actual_arr = pd.Series(y_test[t].values).ffill().bfill().values

        # Probit ACI
        probit_full = probit_preds[lbl]
        cal_scores_probit = np.abs(probit_full[:n_cal] - actual_arr[:n_cal])
        cov_p, wid_p, _ = run_aci(
            cal_scores_probit,
            probit_full[n_cal:],
            actual_arr[n_cal:],
            gamma=GAMMA_PROBIT, alpha_target=ALPHA_TARGET
        )

        # Ensemble ACI
        ens_full = ens_preds[lbl]
        cal_scores_ens = np.abs(ens_full[:n_cal] - actual_arr[:n_cal])
        cov_e, wid_e, _ = run_aci(
            cal_scores_ens,
            ens_full[n_cal:],
            actual_arr[n_cal:],
            gamma=GAMMA_PROBIT, alpha_target=ALPHA_TARGET
        )

        rows_21.append({
            "Cal_frac":          f"{cal_frac:.0%}",
            "Horizon":           lbl,
            "Probit_Coverage":   round(cov_p, 2),
            "Probit_Width":      round(wid_p, 2),
            "Ensemble_Coverage": round(cov_e, 2),
            "Ensemble_Width":    round(wid_e, 2),
            "Nominal_Coverage":  90.0,
        })
        print(f"    {lbl:<10}  Probit: cov={cov_p:.1f}% wid={wid_p:.2f}  "
              f"Ensemble: cov={cov_e:.1f}% wid={wid_e:.2f}")

    # Nonconformity score distribution (KS test: probit vs ensemble)
    print(f"\n  Nonconformity score distributions (probit vs ensemble, KS test):")
    for lbl, t in zip(LABELS, TARGETS):
        actual_arr = pd.Series(y_test[t].values).ffill().bfill().values
        n_cal_i = int(len(actual_arr) * cal_frac)
        s_probit = np.abs(probit_preds[lbl][:n_cal_i] - actual_arr[:n_cal_i])
        s_ens    = np.abs(ens_preds[lbl][:n_cal_i]    - actual_arr[:n_cal_i])
        ks_stat, ks_p = stats.ks_2samp(s_probit, s_ens)
        print(f"    {lbl:<10}  probit_med={np.median(s_probit):.2f}  "
              f"ens_med={np.median(s_ens):.2f}  KS={ks_stat:.3f}  p={ks_p:.3f}")

df_21 = pd.DataFrame(rows_21)
path_21 = os.path.join(OUT, "aci_probit_results.csv")
df_21.to_csv(path_21, index=False)
print(f"\n✓ Saved {path_21}")

# ── Task 4.1: γ Sweep ─────────────────────────────────────────────────────────
print("\n=== TASK 4.1: γ Sweep (6M horizon, 40% + 100% cal) ===")

GAMMAS     = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
CAL_FRACS4 = [0.40, 1.00]
rows_41    = []
TARGET_6M  = "6_month_recession_probability"
LBL_6M     = "6-Month"

actual_6m = pd.Series(y_test[TARGET_6M].values).ffill().bfill().values
pred_6m   = ens_preds[LBL_6M]
n6        = len(pred_6m)

for cal_frac in CAL_FRACS4:
    n_cal = min(int(n6 * cal_frac), n6 - 1)
    n_eval = n6 - n_cal if cal_frac < 1.0 else n6
    cal_scores = np.abs(pred_6m[:n_cal] - actual_6m[:n_cal])

    for gamma in GAMMAS:
        if cal_frac < 1.0:
            cov, wid, alpha_trace = run_aci(
                cal_scores, pred_6m[n_cal:], actual_6m[n_cal:],
                gamma=gamma, alpha_target=ALPHA_TARGET
            )
        else:
            # 100% cal: use leave-one-out style — calibrate on full, eval on full
            cov, wid, alpha_trace = run_aci(
                np.abs(pred_6m - actual_6m), pred_6m, actual_6m,
                gamma=gamma, alpha_target=ALPHA_TARGET
            )

        rows_41.append({
            "Cal_frac":  f"{cal_frac:.0%}",
            "Gamma":     gamma,
            "Coverage":  round(cov, 2),
            "Avg_Width": round(wid, 2),
            "N_cal":     n_cal,
            "N_eval":    n_eval if cal_frac < 1.0 else n6,
        })
        print(f"  cal={cal_frac:.0%}  γ={gamma:.3f}  cov={cov:.1f}%  wid={wid:.2f}")

df_41 = pd.DataFrame(rows_41)
path_41 = os.path.join(OUT, "gamma_sweep_results.csv")
df_41.to_csv(path_41, index=False)
print(f"\n✓ Saved {path_41}")

# Plot γ sweep
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, cal_frac in zip(axes, ["40%", "100%"]):
    sub = df_41[df_41["Cal_frac"] == cal_frac]
    ax.plot(sub["Gamma"], sub["Coverage"],  "o-", label="Coverage (%)", color="steelblue")
    ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90% target")
    ax2 = ax.twinx()
    ax2.plot(sub["Gamma"], sub["Avg_Width"], "s--", label="Avg Width (pp)", color="darkorange")
    ax.set_xscale("log")
    ax.set_xlabel("γ (log scale)")
    ax.set_ylabel("Coverage (%)", color="steelblue")
    ax2.set_ylabel("Avg Width (pp)", color="darkorange")
    ax.set_title(f"6M Horizon — ACI γ Sweep ({cal_frac} calibration)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

plt.tight_layout()
path_41_fig = os.path.join(OUT, "gamma_sweep_figure.png")
plt.savefig(path_41_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Saved {path_41_fig}")

# ── Task 4.2: KS Tests + Horizon-Specific Calibration ─────────────────────────
print("\n=== TASK 4.2: KS Tests + Horizon-Specific vs Shared Calibration ===")

CAL_FRAC_42 = 0.40
n_cal_42    = int(len(ens_preds["Current"]) * CAL_FRAC_42)

# Collect calibration nonconformity scores per horizon
scores_per_horizon = {}
for lbl, t in zip(LABELS, TARGETS):
    actual_arr = pd.Series(y_test[t].values).ffill().bfill().values
    scores_per_horizon[lbl] = np.abs(ens_preds[lbl][:n_cal_42] - actual_arr[:n_cal_42])

# KS tests for all horizon pairs
rows_42_ks = []
print("\n  KS tests between horizon pairs (calibration score distributions):")
for i, lbl1 in enumerate(LABELS):
    for j, lbl2 in enumerate(LABELS):
        if j <= i: continue
        ks_stat, ks_p = stats.ks_2samp(scores_per_horizon[lbl1], scores_per_horizon[lbl2])
        rows_42_ks.append({
            "Horizon_A": lbl1, "Horizon_B": lbl2,
            "KS_stat":   round(ks_stat, 4), "p_value": round(ks_p, 4),
            "Significant_5pct": ks_p < 0.05,
        })
        sig = "***" if ks_p < 0.01 else ("**" if ks_p < 0.05 else ("*" if ks_p < 0.10 else ""))
        print(f"    {lbl1} vs {lbl2}: KS={ks_stat:.4f}  p={ks_p:.4f} {sig}")

df_42_ks = pd.DataFrame(rows_42_ks)
path_42_ks = os.path.join(OUT, "ks_test_results.csv")
df_42_ks.to_csv(path_42_ks, index=False)
print(f"\n✓ Saved {path_42_ks}")

# Horizon-specific vs shared calibration
# Shared: pool all 4 horizons' cal scores into one pool
shared_scores = np.concatenate(list(scores_per_horizon.values()))
rows_42_cal = []
print("\n  Horizon-specific vs Shared calibration (γ=0.02, 40% cal):")
for lbl, t in zip(LABELS, TARGETS):
    actual_arr = pd.Series(y_test[t].values).ffill().bfill().values
    pred_arr   = ens_preds[lbl]

    # Horizon-specific
    cov_spec, wid_spec, _ = run_aci(
        scores_per_horizon[lbl], pred_arr[n_cal_42:], actual_arr[n_cal_42:],
        gamma=0.02, alpha_target=ALPHA_TARGET
    )
    # Shared pool
    cov_share, wid_share, _ = run_aci(
        shared_scores, pred_arr[n_cal_42:], actual_arr[n_cal_42:],
        gamma=0.02, alpha_target=ALPHA_TARGET
    )

    rows_42_cal.append({
        "Horizon":               lbl,
        "HorizonSpecific_Cov":   round(cov_spec,  2),
        "HorizonSpecific_Width": round(wid_spec,  2),
        "Shared_Cov":            round(cov_share, 2),
        "Shared_Width":          round(wid_share, 2),
        "Nominal_Coverage":      90.0,
    })
    print(f"    {lbl:<10}  specific: cov={cov_spec:.1f}% wid={wid_spec:.2f}  "
          f"shared: cov={cov_share:.1f}% wid={wid_share:.2f}")

df_42_cal = pd.DataFrame(rows_42_cal)
path_42_cal = os.path.join(OUT, "horizon_specific_cal_results.csv")
df_42_cal.to_csv(path_42_cal, index=False)
print(f"✓ Saved {path_42_cal}")

print("\n=== ALL TASKS COMPLETE ===")
print(f"  {path_21}")
print(f"  {path_41}")
print(f"  {path_41_fig}")
print(f"  {path_42_ks}")
print(f"  {path_42_cal}")
