"""
Tasks 5.1, 5.2, 5.3:
  5.1 – AR(1) Stage 1 proxy: forecast performance table for 12 base indicators
  5.2 – Oracle (actual indicators) vs Forecast (AR(1) indicators) comparison
  5.3 – Perturbation sensitivity heatmap (±1σ/±2σ per indicator)

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
import seaborn as sns
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
LABELS = ["Current", "1-Month", "3-Month", "6-Month"]

# 12 base indicators used in Stage 1 (economic indicators in the feature set)
BASE_INDICATORS = [
    "share_price",
    "OECD_CLI_index",
    "CSI_index",
    "gdp_per_capita_residual",
    "PPI_diff1",
    "1_year_rate",
    "3_months_rate",
    "6_months_rate",
    "10_year_rate",
    "unemployment_rate",
    "gdp_per_capita_diff1",
    "gdp_per_capita_pct_change1",
]

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
def clean(df):
    return df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
def mae(actual, pred):
    a = pd.Series(actual).ffill().bfill().values
    return float(np.mean(np.abs(a - np.array(pred))))

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

# Confirm which base indicators are actually present
avail_indicators = [c for c in BASE_INDICATORS if c in X_train.columns]
print(f"  Available base indicators: {len(avail_indicators)}/{len(BASE_INDICATORS)}")
print(f"  {avail_indicators}")

# ── load ensemble ─────────────────────────────────────────────────────────────
print("Loading ensemble...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

# ── Task 5.1: AR(1) Stage 1 Proxy Performance ─────────────────────────────────
print("\n=== TASK 5.1: AR(1) Stage 1 Proxy Forecast Performance ===")

# Rolling AR(1): for each test obs, fit AR(1) on training window, forecast 1 step
def ar1_rolling_forecast(series_all, train_mask, test_mask):
    """
    For each test row, fit AR(1) on all training data and produce a 1-step forecast.
    series_all: full series aligned with df.index
    """
    train_vals = series_all[train_mask].values
    test_vals  = series_all[test_mask].values
    forecasts  = []

    for i in range(len(test_vals)):
        # Expanding window: train data + all previous test observations
        window = np.concatenate([train_vals, test_vals[:i]])
        # Remove NaN/inf
        window = window[np.isfinite(window)]
        if len(window) < 3:
            forecasts.append(np.nan)
            continue
        try:
            # Fit AR(1): y_t = a + b*y_{t-1}
            y    = window[1:]
            ylag = window[:-1]
            coefs = np.polyfit(ylag, y, 1)
            b_coef, a_coef = coefs[0], coefs[1]
            forecast = a_coef + b_coef * window[-1]
            if not np.isfinite(forecast):
                forecast = window[-1]  # fallback: naive 1-step
        except Exception:
            forecast = window[-1]
        forecasts.append(forecast)

    return np.array(forecasts)

train_mask = df["date"] < SPLIT
test_mask  = df["date"] >= SPLIT

rows_51 = []
ar1_forecasts_all = {}

for ind in avail_indicators:
    series = df[ind]
    actual_test = X_test[ind].values
    ar1_fc = ar1_rolling_forecast(series, train_mask, test_mask)
    ar1_forecasts_all[ind] = ar1_fc

    # Clip NaN at start
    valid = ~np.isnan(ar1_fc)
    if valid.sum() == 0:
        mae_ar1 = np.nan
        corr    = np.nan
    else:
        mae_ar1 = float(np.mean(np.abs(actual_test[valid] - ar1_fc[valid])))
        corr, _ = stats.pearsonr(actual_test[valid], ar1_fc[valid])

    # Naive (mean of training) forecast for comparison
    naive_fc  = np.full(len(actual_test), series[train_mask].mean())
    mae_naive = float(np.mean(np.abs(actual_test - naive_fc)))

    rows_51.append({
        "Indicator":    ind,
        "MAE_AR1":      round(mae_ar1, 4) if not np.isnan(mae_ar1) else None,
        "MAE_Naive":    round(mae_naive, 4),
        "Pearson_r":    round(corr, 4) if not np.isnan(corr) else None,
        "Improvement":  round(mae_naive - mae_ar1, 4) if not np.isnan(mae_ar1) else None,
        "N_valid":      int(valid.sum()),
    })
    print(f"  {ind:<40}  AR1 MAE={mae_ar1:.4f}  Naive MAE={mae_naive:.4f}  "
          f"r={corr:.3f}")

df_51 = pd.DataFrame(rows_51)
path_51 = os.path.join(OUT, "stage1_forecast_performance.csv")
df_51.to_csv(path_51, index=False)
print(f"\n✓ Saved {path_51}")

# ── Task 5.2: Oracle vs Forecast Comparison ────────────────────────────────────
print("\n=== TASK 5.2: Oracle (actual) vs AR(1) Forecast indicators ===")

# Oracle predictions: use actual X_test features (what we already have)
oracle_preds_raw = ensemble.predict(X_test)
oracle_preds     = {lbl: oracle_preds_raw[:, i] for i, lbl in enumerate(LABELS)}

# AR(1) forecast version: replace each base indicator with its AR(1) forecast
X_test_ar1 = X_test.copy()
for ind in avail_indicators:
    fc = ar1_forecasts_all[ind]
    # Fill NaN with original value (first few rows)
    fc_series = pd.Series(fc)
    fc_series.iloc[0] = X_test_ar1[ind].iloc[0]  # fill first NaN
    fc_series = fc_series.ffill()
    X_test_ar1[ind] = fc_series.values

ar1_preds_raw = ensemble.predict(X_test_ar1)
ar1_preds     = {lbl: ar1_preds_raw[:, i] for i, lbl in enumerate(LABELS)}

rows_52 = []
rows_52_indiv = []
print("  Oracle vs AR1 MAE comparison:")
for lbl, t in zip(LABELS, TARGETS):
    actual = pd.Series(y_test[t].values).ffill().bfill().values
    mae_oracle = mae(actual, oracle_preds[lbl])
    mae_ar1    = mae(actual, ar1_preds[lbl])
    delta      = mae_ar1 - mae_oracle  # positive = AR1 is worse

    rows_52.append({
        "Horizon":      lbl,
        "MAE_Oracle":   round(mae_oracle, 4),
        "MAE_AR1":      round(mae_ar1,    4),
        "Delta_MAE":    round(delta,       4),
        "Pct_Degradation": round(100 * delta / mae_oracle, 2) if mae_oracle > 0 else None,
    })
    print(f"  {lbl:<10}  Oracle={mae_oracle:.4f}  AR1={mae_ar1:.4f}  "
          f"Δ={delta:+.4f} ({100*delta/mae_oracle:+.1f}%)")

# Per-indicator attribution: replace one indicator at a time with AR(1), measure delta
print("\n  Per-indicator attribution (one-at-a-time AR1 replacement):")
for ind in avail_indicators:
    X_test_one = X_test.copy()
    fc = ar1_forecasts_all[ind]
    fc_s = pd.Series(fc).ffill().bfill()
    X_test_one[ind] = fc_s.values

    one_preds_raw = ensemble.predict(X_test_one)
    for i, (lbl, t) in enumerate(zip(LABELS, TARGETS)):
        actual     = pd.Series(y_test[t].values).ffill().bfill().values
        mae_oracle = mae(actual, oracle_preds[lbl])
        mae_one    = mae(actual, one_preds_raw[:, i])
        delta      = mae_one - mae_oracle
        rows_52_indiv.append({
            "Indicator": ind,
            "Horizon":   lbl,
            "MAE_Oracle": round(mae_oracle, 4),
            "MAE_with_AR1": round(mae_one, 4),
            "Delta_MAE": round(delta, 4),
        })
    print(f"    {ind:<40}  Avg Δ={np.mean([r['Delta_MAE'] for r in rows_52_indiv[-4:]]):+.4f}")

df_52      = pd.DataFrame(rows_52)
df_52_indiv = pd.DataFrame(rows_52_indiv)

path_52 = os.path.join(OUT, "oracle_vs_forecast.csv")
path_52_indiv = os.path.join(OUT, "per_indicator_attribution.csv")
df_52.to_csv(path_52, index=False)
df_52_indiv.to_csv(path_52_indiv, index=False)
print(f"\n✓ Saved {path_52}")
print(f"✓ Saved {path_52_indiv}")

# ── Task 5.3: Perturbation Sensitivity ────────────────────────────────────────
print("\n=== TASK 5.3: Perturbation Sensitivity Heatmap ===")

N_SAMPLES = 200
SIGMAS    = [1, 2]

# Sample 200 rows from test set (or use all if <= 200)
np.random.seed(42)
if len(X_test) <= N_SAMPLES:
    sample_idx = np.arange(len(X_test))
else:
    sample_idx = np.random.choice(len(X_test), N_SAMPLES, replace=False)

X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
y_sample = y_test.iloc[sample_idx].reset_index(drop=True)

# Baseline ensemble predictions on sample
base_preds_raw = ensemble.predict(X_sample)
base_preds     = {lbl: base_preds_raw[:, i] for i, lbl in enumerate(LABELS)}

# Compute training std per indicator for perturbation scale
indicator_stds = {ind: float(X_train[ind].std()) for ind in avail_indicators}

# Sensitivity matrices: rows = indicators, cols = horizons
# Value = mean absolute change in prediction when indicator is perturbed by ±k*sigma
for sigma_k in SIGMAS:
    print(f"\n  Computing ±{sigma_k}σ perturbation sensitivity...")
    sens_matrix = np.zeros((len(avail_indicators), len(LABELS)))

    for row_i, ind in enumerate(avail_indicators):
        std = indicator_stds[ind]
        if std == 0:
            continue

        # + perturbation
        X_pos = X_sample.copy()
        X_pos[ind] = X_pos[ind] + sigma_k * std
        pred_pos = ensemble.predict(X_pos)

        # - perturbation
        X_neg = X_sample.copy()
        X_neg[ind] = X_neg[ind] - sigma_k * std
        pred_neg = ensemble.predict(X_neg)

        for col_j in range(len(LABELS)):
            delta_pos = np.abs(pred_pos[:, col_j] - base_preds_raw[sample_idx, col_j])
            delta_neg = np.abs(pred_neg[:, col_j] - base_preds_raw[sample_idx, col_j])
            sens_matrix[row_i, col_j] = float(np.mean((delta_pos + delta_neg) / 2))

        print(f"    {ind:<40}  avg sens = {sens_matrix[row_i].mean():.4f}")

    # Save matrix
    df_sens = pd.DataFrame(sens_matrix, index=avail_indicators, columns=LABELS)
    path_sens = os.path.join(OUT, f"sensitivity_matrix_{sigma_k}sigma.csv")
    df_sens.to_csv(path_sens)
    print(f"  ✓ Saved {path_sens}")

# Heatmap figure (both sigmas side by side)
fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(avail_indicators) * 0.5 + 1)))

for ax, sigma_k in zip(axes, SIGMAS):
    path_s = os.path.join(OUT, f"sensitivity_matrix_{sigma_k}sigma.csv")
    mat    = pd.read_csv(path_s, index_col=0)
    sns.heatmap(
        mat, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
        cbar_kws={"label": "Mean |Δ Pred| (pp)"},
        linewidths=0.3,
    )
    ax.set_title(f"Perturbation Sensitivity (±{sigma_k}σ)", fontweight="bold")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Indicator")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
path_heatmap = os.path.join(OUT, "sensitivity_heatmap.png")
plt.savefig(path_heatmap, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Saved {path_heatmap}")

print("\n=== ALL TASKS COMPLETE ===")
print(f"  {path_51}")
print(f"  {path_52}")
print(f"  {path_52_indiv}")
print(f"  {os.path.join(OUT, 'sensitivity_matrix_1sigma.csv')}")
print(f"  {os.path.join(OUT, 'sensitivity_matrix_2sigma.csv')}")
print(f"  {path_heatmap}")
