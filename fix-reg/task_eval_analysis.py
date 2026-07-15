"""
Tasks 1.1, 1.3, 2.2:
  1.1 – Bootstrap CIs on MAE for all 6 models × 4 horizons
  1.3 – COVID exclusion (remove Mar+Apr 2020, recompute on 63 obs)
  2.2 – Residual analysis: recession vs expansion, monthly, error-prob correlation

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
from xgboost import DMatrix, train as xgb_train
import lightgbm as lgb
warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
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

# ── unpickling stubs ─────────────────────────────────────────────────────────
eps = 1e-8
def _inv_logit(z):
    return np.clip(1 / (1 + np.exp(-np.clip(z, -50, 50))) * 100, 0, 100)
def _san(df):
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", c) for c in df.columns]
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

# ── helpers ──────────────────────────────────────────────────────────────────
EPS = 1e-6
def logit(y):
    y_s = np.clip(np.array(y, dtype=float) / 100.0, EPS, 1 - EPS)
    return np.log(y_s / (1 - y_s))
def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-np.array(z, dtype=float))) * 100.0
def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

def mae(actual, pred):
    a = pd.Series(actual).ffill().bfill().values
    return np.mean(np.abs(a - pred))

def bootstrap_ci(actual, pred, n_boot=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    actual = pd.Series(actual).ffill().bfill().values
    n = len(actual)
    boot_maes = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_maes.append(np.mean(np.abs(actual[idx] - pred[idx])))
    lo = np.percentile(boot_maes, (100 - ci) / 2)
    hi = np.percentile(boot_maes, 100 - (100 - ci) / 2)
    return lo, hi

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

# ── build baselines ───────────────────────────────────────────────────────────
print("Building baselines...")

# Naive mean
naive_preds = {lbl: np.full(len(y_test), y_train[t].mean())
               for t, lbl in zip(TARGETS, LABELS)}

# Probit (yield curve)
spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1, 1)
probit_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    reg = LinearRegression()
    reg.fit(spread_train, logit(y_train[t].values))
    probit_preds[lbl] = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)

# Single-stage XGBoost
XGB_PARAMS = {
    "objective": "reg:squarederror", "max_depth": 5, "eta": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9, "seed": 42, "verbosity": 0,
}
sxgb_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    dtrain = DMatrix(X_train.values, label=logit(y_train[t].values))
    dtest  = DMatrix(X_test.values)
    model  = xgb_train(XGB_PARAMS, dtrain, num_boost_round=500)
    sxgb_preds[lbl] = np.clip(inv_logit(model.predict(dtest)), 0, 100)

# Ablation baselines (loaded from CSV if available, else compute naive versions)
abl_csv = os.path.join(OUT, "ablation_results.csv")
abl_preds = {lbl: {} for lbl in LABELS}
if os.path.exists(abl_csv):
    abl_df = pd.read_csv(abl_csv)
    print("  Ablation results CSV found — using stored MAEs for task 1.1.")
    abl_mae_stored = {}
    for _, row in abl_df.iterrows():
        abl_mae_stored[(row["Horizon"], row["Baseline"])] = row["Baseline_MAE"]

ALL_MODELS = {
    "Ensemble (Ours)":       ens_preds,
    "Naive Mean":            naive_preds,
    "Probit (YC)":           probit_preds,
    "Single-Stage XGB":      sxgb_preds,
}

# ── Task 1.1: Bootstrap CIs ────────────────────────────────────────────────────
print("\n=== TASK 1.1: Bootstrap CIs on MAE (1000 resamples, 95% CI) ===")

N_BOOT = 1000
rows_11 = []
for lbl, t in zip(LABELS, TARGETS):
    actual = y_test[t].values
    for model_name, preds_dict in ALL_MODELS.items():
        pred = preds_dict[lbl]
        point_mae = mae(actual, pred)
        lo, hi = bootstrap_ci(actual, pred, n_boot=N_BOOT, ci=95)
        rows_11.append({
            "Horizon":    lbl,
            "Model":      model_name,
            "MAE":        round(point_mae, 4),
            "CI_lo_95":   round(lo, 4),
            "CI_hi_95":   round(hi, 4),
        })
        print(f"  {lbl:<10} {model_name:<25} MAE={point_mae:.4f} CI=[{lo:.4f}, {hi:.4f}]")

df_11 = pd.DataFrame(rows_11)
path_11 = os.path.join(OUT, "bootstrap_ci_results.csv")
df_11.to_csv(path_11, index=False)
print(f"\n✓ Saved {path_11}")

# ── Task 1.3: COVID Exclusion ──────────────────────────────────────────────────
print("\n=== TASK 1.3: COVID Exclusion (remove Mar+Apr 2020) ===")

covid_mask = ~((test_df["date"].dt.year == 2020) &
               (test_df["date"].dt.month.isin([3, 4])))
print(f"  Test size before: {len(test_df)} | after COVID excl: {covid_mask.sum()}")

rows_13 = []
for lbl, t in zip(LABELS, TARGETS):
    actual_full = y_test[t].values
    actual_excl = y_test[t].values[covid_mask.values]

    for model_name, preds_dict in ALL_MODELS.items():
        pred_full = preds_dict[lbl]
        pred_excl = pred_full[covid_mask.values]

        mae_full = mae(actual_full, pred_full)
        mae_excl = mae(actual_excl, pred_excl)
        delta    = mae_excl - mae_full

        rows_13.append({
            "Horizon":      lbl,
            "Model":        model_name,
            "MAE_full":     round(mae_full, 4),
            "MAE_excl":     round(mae_excl, 4),
            "Delta_MAE":    round(delta, 4),
            "N_full":       len(actual_full),
            "N_excl":       int(covid_mask.sum()),
        })
        print(f"  {lbl:<10} {model_name:<25} full={mae_full:.4f} excl={mae_excl:.4f} Δ={delta:+.4f}")

df_13 = pd.DataFrame(rows_13)
path_13 = os.path.join(OUT, "covid_exclusion_results.csv")
df_13.to_csv(path_13, index=False)
print(f"\n✓ Saved {path_13}")

# ── Task 2.2: Residual Analysis ───────────────────────────────────────────────
print("\n=== TASK 2.2: Residual Analysis ===")

# Identify recession vs expansion months in test set
# Recession = recession_probability > 50
rec_flag = y_test["recession_probability"].values > 50
exp_flag = ~rec_flag
print(f"  Test months: recession={rec_flag.sum()}, expansion={exp_flag.sum()}")

# 2.2a: Recession vs expansion MAE per model × horizon
rows_22a = []
for lbl, t in zip(LABELS, TARGETS):
    actual = pd.Series(y_test[t].values).ffill().bfill().values
    for model_name, preds_dict in ALL_MODELS.items():
        pred = preds_dict[lbl]
        err  = np.abs(actual - pred)

        mae_rec = np.mean(err[rec_flag]) if rec_flag.any() else np.nan
        mae_exp = np.mean(err[exp_flag]) if exp_flag.any() else np.nan
        ratio   = mae_rec / mae_exp if mae_exp > 0 else np.nan

        rows_22a.append({
            "Horizon":        lbl,
            "Model":          model_name,
            "MAE_recession":  round(mae_rec, 4) if not np.isnan(mae_rec) else None,
            "MAE_expansion":  round(mae_exp, 4) if not np.isnan(mae_exp) else None,
            "Ratio_rec_exp":  round(ratio, 4)   if not np.isnan(ratio)   else None,
        })
        print(f"  {lbl:<10} {model_name:<25} rec={mae_rec:.4f} exp={mae_exp:.4f} ratio={ratio:.2f}")

df_22a = pd.DataFrame(rows_22a)
path_22a = os.path.join(OUT, "residual_recession_vs_expansion.csv")
df_22a.to_csv(path_22a, index=False)
print(f"\n✓ Saved {path_22a}")

# 2.2b: Monthly error profile (average absolute error by calendar month)
rows_22b = []
for lbl, t in zip(LABELS, TARGETS):
    actual = pd.Series(y_test[t].values).ffill().bfill().values
    months = test_df["date"].dt.month.values
    for model_name, preds_dict in ALL_MODELS.items():
        pred = preds_dict[lbl]
        err  = np.abs(actual - pred)
        for m in range(1, 13):
            mask = months == m
            if mask.any():
                rows_22b.append({
                    "Horizon": lbl,
                    "Model":   model_name,
                    "Month":   m,
                    "MAE":     round(np.mean(err[mask]), 4),
                    "N":       int(mask.sum()),
                })

df_22b = pd.DataFrame(rows_22b)
path_22b = os.path.join(OUT, "residual_monthly.csv")
df_22b.to_csv(path_22b, index=False)
print(f"✓ Saved {path_22b}")

# 2.2c: Error-probability correlation (Spearman r between ensemble pred and abs error)
print("\n  Error-probability correlation (Spearman r, ensemble pred vs |error|):")
rows_22c = []
for lbl, t in zip(LABELS, TARGETS):
    actual = pd.Series(y_test[t].values).ffill().bfill().values
    pred   = ens_preds[lbl]
    err    = np.abs(actual - pred)
    r, p   = stats.spearmanr(pred, err)
    rows_22c.append({
        "Horizon":       lbl,
        "Spearman_r":    round(r, 4),
        "p_value":       round(p, 4),
        "Interpretation": "higher-prob → more error" if r > 0 else "higher-prob → less error",
    })
    print(f"  {lbl:<10} r={r:.4f}  p={p:.4f}")

df_22c = pd.DataFrame(rows_22c)
path_22c = os.path.join(OUT, "residual_error_prob_corr.csv")
df_22c.to_csv(path_22c, index=False)
print(f"✓ Saved {path_22c}")

print("\n=== ALL TASKS COMPLETE ===")
print(f"  {path_11}")
print(f"  {path_13}")
print(f"  {path_22a}")
print(f"  {path_22b}")
print(f"  {path_22c}")
