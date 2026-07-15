"""
TASK 2 — Horizon-Conditional Feature Attribution (SHAP)
Runs SHAP on each of the four horizon models in the CatBoost RegressorChain.
Produces the SHAP matrix CSV, top-12 rank analysis, and Figure C heatmap.
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
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
OUT_DIR    = "fix-reg/task2_outputs"
SPLIT      = "2020-01-01"

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

def clean(d):
    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

test_df = df[df["date"] >= SPLIT].copy()
X_test  = clean(test_df.drop(columns=recession_targets + ["date"]))
orig_feature_names = list(X_test.columns)
print(f"  Test rows: {len(X_test)}, Features: {len(orig_feature_names)}")

# ── Load ensemble ─────────────────────────────────────────────
print("Loading model...")
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)

cb_model = ensemble.fitted_base_models["CatBoost"]
chain    = cb_model.chain_model      # sklearn RegressorChain with 4 CatBoost estimators
scaler   = cb_model.scaler

# Scale X_test as CatBoost model expects
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

# ── SHAP for each horizon ─────────────────────────────────────
# The RegressorChain augments features: est[i] gets X + predictions from est[0..i-1]
# We compute SHAP on all columns but only keep SHAP values for original features.

print("\nComputing SHAP for all 4 horizons...")
shap_matrix = {}    # feature → {horizon → mean_abs_shap}

# Build augmented feature arrays as the chain presents them to each estimator
prev_preds = np.zeros((len(X_test_scaled), 0))  # start with no augmented columns

for h_idx, h_label in enumerate(LABELS):
    estimator = chain.estimators_[h_idx]

    # Build input for this estimator: scaled X + chain preds from previous horizons
    if prev_preds.shape[1] > 0:
        X_aug = np.hstack([X_test_scaled.values, prev_preds])
    else:
        X_aug = X_test_scaled.values

    print(f"  {h_label}: estimator input shape {X_aug.shape}, "
          f"computing TreeExplainer SHAP values...")

    explainer  = shap.TreeExplainer(estimator)
    shap_vals  = explainer.shap_values(X_aug)   # (n_test, n_aug_features)

    # Keep only the original feature SHAP values (first n_orig columns)
    n_orig   = len(orig_feature_names)
    sv_orig  = shap_vals[:, :n_orig]             # (n_test, n_orig)
    mean_abs = np.abs(sv_orig).mean(axis=0)      # (n_orig,)

    for feat_idx, feat in enumerate(orig_feature_names):
        if feat not in shap_matrix:
            shap_matrix[feat] = {}
        shap_matrix[feat][h_label] = float(mean_abs[feat_idx])

    # Augment prev_preds with this horizon's predictions (in logit space from chain)
    h_pred = estimator.predict(X_aug).reshape(-1, 1)
    prev_preds = np.hstack([prev_preds, h_pred])

    print(f"    Top 5 features: " +
          str(sorted(shap_matrix.keys(),
                     key=lambda f: shap_matrix[f][h_label], reverse=True)[:5]))

# ── Save SHAP matrix CSV ──────────────────────────────────────
shap_df = pd.DataFrame(shap_matrix).T   # rows=features, index=feature names
shap_df.index.name = "Feature"
shap_df = shap_df[LABELS]              # ensure column order
shap_df.to_csv(f"{OUT_DIR}/shap_matrix_all_features.csv")
print(f"\nSHAP matrix saved → {OUT_DIR}/shap_matrix_all_features.csv")
print(f"  Shape: {shap_df.shape}")

# ── Top 12 features by mean SHAP across all horizons ─────────
shap_df["mean_across_horizons"] = shap_df[LABELS].mean(axis=1)
top12_df = shap_df.nlargest(12, "mean_across_horizons")[LABELS + ["mean_across_horizons"]]
top12_features = list(top12_df.index)

print("\nTop 12 features (mean SHAP across all horizons):")
for i, feat in enumerate(top12_features):
    print(f"  {i+1:2d}. {feat:45s} {top12_df.loc[feat,'mean_across_horizons']:.4f}")

# ── Rank analysis: 1M vs 6M ───────────────────────────────────
# Rank all features by SHAP at each horizon (1=highest)
rank_1m = shap_df["1M"].rank(ascending=False).astype(int)
rank_6m = shap_df["6M"].rank(ascending=False).astype(int)

rank_df = pd.DataFrame({
    "Feature":      top12_features,
    "Mean_SHAP":    [round(top12_df.loc[f, "mean_across_horizons"], 4) for f in top12_features],
    "Rank_1M":      [int(rank_1m[f]) for f in top12_features],
    "Rank_6M":      [int(rank_6m[f]) for f in top12_features],
    "Rank_Change":  [int(rank_1m[f]) - int(rank_6m[f]) for f in top12_features],
})
rank_df["Direction"] = rank_df["Rank_Change"].apply(
    lambda x: "UP (forward-looking)" if x > 0 else ("DOWN (current-state)" if x < 0 else "STABLE")
)
rank_df.to_csv(f"{OUT_DIR}/top12_rank_analysis.csv", index=False)
print(f"\nTop-12 rank analysis saved → {OUT_DIR}/top12_rank_analysis.csv")
print(rank_df[["Feature","Rank_1M","Rank_6M","Rank_Change","Direction"]].to_string(index=False))

# ── Feature with biggest rank change ─────────────────────────
all_rank_changes = pd.DataFrame({
    "Feature":     shap_df.index,
    "Rank_1M":     rank_1m.values,
    "Rank_6M":     rank_6m.values,
    "Abs_Change":  (rank_1m - rank_6m).abs().values,
    "Change":      (rank_1m - rank_6m).values,
})
all_rank_changes = all_rank_changes.set_index("Feature")
biggest_change_feat = all_rank_changes["Abs_Change"].idxmax()
biggest_row = all_rank_changes.loc[biggest_change_feat]

print(f"\nBiggest rank change (1M→6M): '{biggest_change_feat}'")
print(f"  Rank at 1M: {biggest_row['Rank_1M']}, Rank at 6M: {biggest_row['Rank_6M']}, "
      f"Change: {biggest_row['Change']:+.0f}")

# Features moving UP (more important at 6M than 1M — forward-looking signals)
moved_up   = rank_df[rank_df["Rank_Change"] > 0]["Feature"].tolist()
moved_down = rank_df[rank_df["Rank_Change"] < 0]["Feature"].tolist()
print(f"\nFeatures moved UP in rank 1M→6M (forward-looking): {moved_up}")
print(f"Features moved DOWN in rank 1M→6M (current-state):  {moved_down}")

summary_txt = f"""TASK 2 SUMMARY
==============
Top 12 features: {top12_features}
Biggest rank change (1M→6M): {biggest_change_feat}  (1M: {int(biggest_row['Rank_1M'])}, 6M: {int(biggest_row['Rank_6M'])}, Δ={int(biggest_row['Change']):+d})
Features UP at 6M (forward-looking): {moved_up}
Features DOWN at 6M (current-state): {moved_down}
"""
with open(f"{OUT_DIR}/task2_summary.txt", "w") as f:
    f.write(summary_txt)
print(summary_txt)

# ── Figure C — SHAP heatmap ───────────────────────────────────
heatmap_data = top12_df[LABELS].values   # (12, 4)
feat_labels  = [f.replace("_", " ") for f in top12_features]

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=heatmap_data.max())

ax.set_xticks(range(len(LABELS)))
ax.set_xticklabels(LABELS, fontsize=12)
ax.set_yticks(range(len(top12_features)))
ax.set_yticklabels(feat_labels, fontsize=9)
ax.set_title("Figure C — Horizon-Conditional SHAP Feature Importance\n"
             "(Top 12 features; colour = mean |SHAP| — darker red = more important)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Forecast Horizon", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)

# Annotate cells
for i in range(len(top12_features)):
    for j in range(len(LABELS)):
        v = heatmap_data[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8, color="black" if v < heatmap_data.max()*0.6 else "white",
                fontweight="bold")

plt.colorbar(im, ax=ax, label="Mean |SHAP| value", shrink=0.8)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/figure_C_shap_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure C saved → {OUT_DIR}/figure_C_shap_heatmap.png")
print("\nTASK 2 COMPLETE")
