"""
Tasks 1.2, 3.1:
  1.2 – Rolling-Origin Backtesting Harness across 3 recession windows
         (2001-03, 2007-10, 2020-23). THE most important deliverable.
  3.1 – Stepwise Ablation (5 tiers, delta MAE between consecutive steps)

Run from: RecessionRadar/RecessionRadar/
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.multioutput import RegressorChain
from xgboost import DMatrix, train as xgb_train
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
def mae(actual, pred):
    a = pd.Series(actual).ffill().bfill().values
    return float(np.mean(np.abs(a - np.array(pred))))

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ── Task 1.2: Rolling-Origin Backtesting ──────────────────────────────────────
print("\n=== TASK 1.2: Rolling-Origin Backtesting ===")

# Three recession windows (start/end of evaluation period, training expands up to window start)
WINDOWS = [
    {"name": "Dot-Com/9-11",  "eval_start": "2001-03-01", "eval_end": "2003-12-31"},
    {"name": "GFC",           "eval_start": "2007-10-01", "eval_end": "2010-12-31"},
    {"name": "COVID",         "eval_start": "2020-01-01", "eval_end": "2023-12-31"},
]

# Baseline model builders
def build_naive(y_train_t):
    mu = y_train_t.mean()
    return lambda X_test: np.full(len(X_test), mu)

def build_probit(X_train, y_train_t):
    spread_tr = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1,1)
    reg = LinearRegression()
    reg.fit(spread_tr, logit(y_train_t.values))
    def pred(X_test):
        spread_te = (X_test["10_year_rate"] - X_test["3_months_rate"]).values.reshape(-1,1)
        return np.clip(inv_logit(reg.predict(spread_te)), 0, 100)
    return pred

XGB_PARAMS = {
    "objective": "reg:squarederror", "max_depth": 5, "eta": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.9, "seed": 42, "verbosity": 0,
}
def build_xgb(X_train, y_train_t):
    dtrain = DMatrix(X_train.values, label=logit(y_train_t.values))
    model  = xgb_train(XGB_PARAMS, dtrain, num_boost_round=500)
    def pred(X_test):
        return np.clip(inv_logit(model.predict(DMatrix(X_test.values))), 0, 100)
    return pred

# Ensemble requires retrain — we use a simplified single-target XGB chain as proxy
# for rolling windows (the full ensemble needs CatBoost/LGBM installed per window)
# We train independent XGBs per horizon (Baseline C equivalent but Optuna-free)
def build_xgb_chain(X_train, y_train_df):
    """Train 4 XGBs sequentially, each takes previous preds as extra features."""
    models = []
    prev_preds_train = []
    feat_names = list(X_train.columns)

    for i, t in enumerate(TARGETS):
        X_aug = X_train.copy()
        for j, pp in enumerate(prev_preds_train):
            X_aug[f"prev_pred_{j}"] = pp
        y_logit = logit(y_train_df[t].values)
        dtrain  = DMatrix(X_aug.values, label=y_logit)
        m = xgb_train(XGB_PARAMS, dtrain, num_boost_round=500)
        models.append((m, list(X_aug.columns)))
        # Get train predictions for next stage
        prev_preds_train.append(inv_logit(m.predict(dtrain)))

    def pred(X_test):
        prev_preds_test = []
        preds = np.zeros((len(X_test), 4))
        for i, (m, cols) in enumerate(models):
            X_aug = X_test.copy()
            for j, pp in enumerate(prev_preds_test):
                X_aug[f"prev_pred_{j}"] = pp
            p = np.clip(inv_logit(m.predict(DMatrix(X_aug[cols].values))), 0, 100)
            preds[:, i] = p
            prev_preds_test.append(p)
        return preds
    return pred

rows_12      = []
all_ens_preds_by_window = {}

for win in WINDOWS:
    name       = win["name"]
    eval_start = win["eval_start"]
    eval_end   = win["eval_end"]

    # Training data = everything strictly before eval_start
    tr_mask = df["date"] < eval_start
    te_mask = (df["date"] >= eval_start) & (df["date"] <= eval_end)

    if tr_mask.sum() < 60:
        print(f"  *** Skipping {name}: only {tr_mask.sum()} training rows ***")
        continue

    tr = df[tr_mask].copy()
    te = df[te_mask].copy()

    X_tr = clean(tr.drop(columns=TARGETS + ["date"]))
    X_te = clean(te.drop(columns=TARGETS + ["date"]))
    y_tr = clean(tr[TARGETS])
    y_te = clean(te[TARGETS])

    print(f"\n  Window: {name}  train={len(X_tr)}  eval={len(X_te)}")
    if len(X_te) == 0:
        print(f"    (no eval data, skipping)")
        continue

    # Align columns between train and test
    common_cols = [c for c in X_tr.columns if c in X_te.columns]
    X_tr = X_tr[common_cols]
    X_te = X_te[common_cols]

    # Build models
    naive_fns  = {t: build_naive(y_tr[t]) for t in TARGETS}
    probit_fns = {t: build_probit(X_tr, y_tr[t]) for t in TARGETS}
    xgb_fns    = {t: build_xgb(X_tr, y_tr[t]) for t in TARGETS}
    chain_pred = build_xgb_chain(X_tr, y_tr)

    chain_preds = chain_pred(X_te)
    all_ens_preds_by_window[name] = chain_preds

    for i, (lbl, t) in enumerate(zip(LABELS, TARGETS)):
        actual = y_te[t].values

        p_naive  = naive_fns[t](X_te)
        p_probit = probit_fns[t](X_te)
        p_xgb    = xgb_fns[t](X_te)
        p_chain  = chain_preds[:, i]

        mae_naive  = mae(actual, p_naive)
        mae_probit = mae(actual, p_probit)
        mae_xgb    = mae(actual, p_xgb)
        mae_chain  = mae(actual, p_chain)

        rows_12.append({
            "Window":       name,
            "Eval_Start":   eval_start,
            "Eval_End":     eval_end,
            "N_train":      len(X_tr),
            "N_eval":       len(X_te),
            "Horizon":      lbl,
            "MAE_Naive":    round(mae_naive,  4),
            "MAE_Probit":   round(mae_probit, 4),
            "MAE_XGB":      round(mae_xgb,    4),
            "MAE_Chain":    round(mae_chain,  4),
            "Chain_vs_XGB": round(mae_chain - mae_xgb, 4),
        })
        print(f"    {lbl:<10}  naive={mae_naive:.2f}  probit={mae_probit:.2f}  "
              f"xgb={mae_xgb:.2f}  chain={mae_chain:.2f}  "
              f"Δ(chain-xgb)={mae_chain-mae_xgb:+.2f}")

df_12 = pd.DataFrame(rows_12)
path_12 = os.path.join(OUT, "rolling_origin_results.csv")
df_12.to_csv(path_12, index=False)
print(f"\n✓ Saved {path_12}")

# Summary: average across windows per horizon
df_12_summary = (df_12.groupby("Horizon")[
    ["MAE_Naive", "MAE_Probit", "MAE_XGB", "MAE_Chain", "Chain_vs_XGB"]
].mean().round(4).reset_index())
path_12_sum = os.path.join(OUT, "rolling_origin_summary.csv")
df_12_summary.to_csv(path_12_sum, index=False)
print(f"✓ Saved {path_12_sum}")
print("\n  Summary (avg across windows):")
print(df_12_summary.to_string(index=False))

# Figure: Rolling-origin MAE bar chart per window
fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 5), sharey=False)
if len(WINDOWS) == 1: axes = [axes]

colors = {"MAE_Naive": "#aaa", "MAE_Probit": "#6baed6",
          "MAE_XGB": "#fd8d3c", "MAE_Chain": "#31a354"}
labels_map = {"MAE_Naive": "Naive Mean", "MAE_Probit": "Probit (YC)",
              "MAE_XGB": "Single XGB", "MAE_Chain": "XGB Chain"}
x = np.arange(len(LABELS))

for ax, win in zip(axes, WINDOWS):
    wdf = df_12[df_12["Window"] == win["name"]]
    if wdf.empty: continue
    width = 0.2
    for k, col in enumerate(["MAE_Naive", "MAE_Probit", "MAE_XGB", "MAE_Chain"]):
        vals = [wdf[wdf["Horizon"]==lbl][col].values[0] if not wdf[wdf["Horizon"]==lbl].empty else 0
                for lbl in LABELS]
        ax.bar(x + k*width - 0.3, vals, width, label=labels_map[col], color=colors[col])
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_title(win["name"])
    ax.set_ylabel("MAE (pp)")
    ax.legend(fontsize=7)

fig.suptitle("Rolling-Origin Backtesting: MAE by Window and Horizon", fontweight="bold")
plt.tight_layout()
path_12_fig = os.path.join(OUT, "rolling_origin_figure.png")
plt.savefig(path_12_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Saved {path_12_fig}")

# ── Task 3.1: Stepwise Ablation ────────────────────────────────────────────────
print("\n=== TASK 3.1: Stepwise Ablation (5 Tiers) ===")
"""
Tier 1: Naive Mean (floor)
Tier 2: Probit (single feature: yield spread)
Tier 3: Single-stage XGBoost (all features, independent targets)
Tier 4: XGBoost Chain (all features, sequential chain)
Tier 5: Full Stacking Ensemble (loaded from pkl)
"""

# Use the full test set (post-2020)
SPLIT      = "2020-01-01"
train_df   = df[df["date"] < SPLIT].copy()
test_df    = df[df["date"] >= SPLIT].copy()

X_train = clean(train_df.drop(columns=TARGETS + ["date"]))
X_test  = clean(test_df.drop(columns=TARGETS + ["date"]))
y_train = clean(train_df[TARGETS])
y_test  = clean(test_df[TARGETS])

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# Tier 1: Naive
tier1_preds = {lbl: np.full(len(y_test), y_train[t].mean())
               for t, lbl in zip(TARGETS, LABELS)}

# Tier 2: Probit
spread_train = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_test  = (X_test["10_year_rate"]  - X_test["3_months_rate"]).values.reshape(-1, 1)
tier2_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    reg = LinearRegression()
    reg.fit(spread_train, logit(y_train[t].values))
    tier2_preds[lbl] = np.clip(inv_logit(reg.predict(spread_test)), 0, 100)

# Tier 3: Single XGB (independent per horizon)
tier3_preds = {}
for t, lbl in zip(TARGETS, LABELS):
    dtrain = DMatrix(X_train.values, label=logit(y_train[t].values))
    dtest  = DMatrix(X_test.values)
    m = xgb_train(XGB_PARAMS, dtrain, num_boost_round=500)
    tier3_preds[lbl] = np.clip(inv_logit(m.predict(dtest)), 0, 100)

# Tier 4: XGB Chain
chain_fn  = build_xgb_chain(X_train, y_train)
chain_raw = chain_fn(X_test)
tier4_preds = {lbl: chain_raw[:, i] for i, lbl in enumerate(LABELS)}

# Tier 5: Full ensemble
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)
ens_raw = ensemble.predict(X_test)
tier5_preds = {lbl: ens_raw[:, i] for i, lbl in enumerate(LABELS)}

# Compute MAEs
tiers = [
    ("T1: Naive Mean",          tier1_preds),
    ("T2: Probit (YC)",         tier2_preds),
    ("T3: XGB Independent",     tier3_preds),
    ("T4: XGB Chain",           tier4_preds),
    ("T5: Full Ensemble",       tier5_preds),
]

tier_maes = {}
for tier_name, preds in tiers:
    tier_maes[tier_name] = {}
    for lbl, t in zip(LABELS, TARGETS):
        actual = y_test[t].values
        tier_maes[tier_name][lbl] = mae(actual, preds[lbl])

# Build stepwise table
rows_31 = []
tier_names = [t[0] for t in tiers]
for i, (tier_name, _) in enumerate(tiers):
    for lbl in LABELS:
        m_curr = tier_maes[tier_name][lbl]
        m_prev = tier_maes[tier_names[i-1]][lbl] if i > 0 else None
        delta  = m_curr - m_prev if m_prev is not None else None
        rows_31.append({
            "Tier":     tier_name,
            "Horizon":  lbl,
            "MAE":      round(m_curr, 4),
            "Delta_MAE": round(delta, 4) if delta is not None else None,
            "Pct_Gain":  round(100 * (-delta) / m_prev, 2) if (delta is not None and m_prev and m_prev > 0) else None,
        })

df_31 = pd.DataFrame(rows_31)
path_31 = os.path.join(OUT, "stepwise_ablation.csv")
df_31.to_csv(path_31, index=False)
print(f"\n✓ Saved {path_31}")
print("\n  Stepwise ablation table:")
print(df_31.pivot(index="Tier", columns="Horizon", values="MAE").reindex(
    tier_names, columns=LABELS
).to_string())

# Ablation figure: line chart of MAE by tier per horizon
fig, ax = plt.subplots(figsize=(9, 5))
colors_abl = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
markers    = ["s", "^", "D", "o", "*"]

for lbl, col, mk in zip(LABELS, ["#d62728","#ff7f0e","#2ca02c","#9467bd"], ["o","s","^","D"]):
    vals = [tier_maes[tn][lbl] for tn in tier_names]
    ax.plot(range(len(tiers)), vals, marker=mk, label=lbl, color=col, linewidth=1.5)

ax.set_xticks(range(len(tiers)))
ax.set_xticklabels([t.replace(": ", ":\n") for t in tier_names], fontsize=8)
ax.set_ylabel("MAE (percentage points)")
ax.set_title("Stepwise Ablation: MAE at Each Modelling Tier", fontweight="bold")
ax.legend(title="Horizon", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
path_31_fig = os.path.join(OUT, "stepwise_ablation_figure.png")
plt.savefig(path_31_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Saved {path_31_fig}")

print("\n=== ALL TASKS COMPLETE ===")
print(f"  {path_12}")
print(f"  {path_12_sum}")
print(f"  {path_12_fig}")
print(f"  {path_31}")
print(f"  {path_31_fig}")
