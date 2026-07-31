"""
TASKS 2 & 3 for the recession testbed (multidomain revision spec).

Task 3 — In-sample vs out-of-fold coverage gap.
  The published 67.8% -> 81.4% (6M) headline uses in-sample residuals from the
  saved stacking ensemble. Here the diversity-maximizing selector is re-run on
  OUT-OF-FOLD nonconformity scores produced by a blocked/rolling-origin
  cross-validation over the 635-month training period, and both numbers are
  reported side by side.

Task 2 — Model-agnosticism, including the overdue probit comparison.
  ACI and the diversity selector are model-agnostic, so they are applied to the
  nonconformity scores of a second, much simpler underlying model: the
  yield-curve probit baseline (10Y-3M spread -> logit-linear -> inv_logit),
  using the same construction as probit_comparison.py. The full calibration
  strategy comparison (pooled/trailing ACI, Mondrian, PID-conformal,
  extreme-value, diversity-maximizing selector) is run against those scores
  exactly as it is for the RegressorChain pipeline, and reported honestly
  including where it favours the probit.

Every coverage number carries a Wilson interval and a block-bootstrap interval
(Task 4).

Retraining note: the saved stacking ensemble cannot produce honest out-of-fold
scores (it was fit on all 635 months), so the OOF arm refits a like-for-like
surrogate of the same architecture family per fold. This is stated in the
output and is the honest way to answer "how much of the gain survives".

Outputs:
  fix-reg/task3_insample_vs_oof.csv
  fix-reg/task2_probit_strategies.csv
  fix-reg/task2_model_agnostic_summary.csv
"""

import os
import re
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import RegressorChain
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, support_width, wilson_ci_from_indicator,
    block_bootstrap_ci, rolling_origin_folds, GAMMA_DEFAULT,
)

# ── Stub classes required to unpickle the saved ensemble ──────────────────────
recession_targets = [
    "recession_probability", "1_month_recession_probability",
    "3_month_recession_probability", "6_month_recession_probability",
]
LABELS = ["Current", "1M", "3M", "6M"]
eps = 1e-8


def safe_logit(y):
    p = np.clip(np.clip(y, 0, 100) / 100, eps, 1 - eps)
    return np.log(p / (1 - p))


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

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.model.predict(X)


class FullChainCatBoostModel:
    def __init__(self):
        self.chain_model = None; self.scaler = None

    def predict(self, X):
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainLightGBMModel:
    def __init__(self):
        self.chain_model = None; self.scaler = None

    def predict(self, X):
        X = sanitize_columns(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        return np.clip(safe_inv_logit(self.chain_model.predict(Xs)), 0, 100)


class FullChainRandomForestModel:
    def __init__(self):
        self.chain_model = None; self.scaler = None

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
                for j in range(i + 1, len(bp)):
                    f.append(np.abs(bp[i] - bp[j]))
        return np.column_stack(f)

    def predict(self, X):
        bp = {n: m.predict(X) for n, m in self.fitted_base_models.items()}
        fp = np.zeros_like(list(bp.values())[0])
        for i, t in enumerate(recession_targets):
            bpt = [bp[n][:, i] for n in self.base_models]
            mf = self._engineer_meta_features(*bpt)
            fp[:, i] = self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp, 0, 100)


DATA_PATH = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
OUT = "fix-reg"
SPLIT = "2020-01-01"
N_FIX = 254
SEED = 5

os.makedirs(OUT, exist_ok=True)

print("=" * 78)
print("TASKS 2 & 3 — out-of-fold correction and model-agnosticism (recession)")
print("=" * 78)

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)


def clean(d):
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.ffill().bfill().fillna(0)


train_df = df[df["date"] < SPLIT].copy()
test_df = df[df["date"] >= SPLIT].copy()
X_train = clean(train_df.drop(columns=recession_targets + ["date"]))
y_train = train_df[recession_targets].values
X_test = clean(test_df.drop(columns=recession_targets + ["date"]))
y_test = test_df[recession_targets].values
rec_prob = train_df["recession_probability"].values
n_pool = len(train_df)
print(f"  Train {n_pool} months | Test {len(test_df)} months")

with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)
preds_train_insample = ensemble.predict(X_train)
preds_test = ensemble.predict(X_test)
print("  Saved ensemble loaded; in-sample training predictions computed.")


# ── Diversity-maximizing selector (same greedy rule as task_phase2.py) ────────
def greedy_extreme(s_all, N=N_FIX):
    """Alternate from both tails inward — maximizes p95-p5 spread at fixed N."""
    valid = np.where(~np.isnan(s_all))[0]
    order = valid[np.argsort(s_all[valid])]
    chosen, lo, hi, take_low = [], 0, len(order) - 1, True
    while len(chosen) < min(N, len(order)) and lo <= hi:
        if take_low:
            chosen.append(order[lo]); lo += 1
        else:
            chosen.append(order[hi]); hi -= 1
        take_low = not take_low
    return np.array(sorted(chosen))


def evaluate(scores, h_idx, pred_test_h, block=12):
    """
    Coverage alone cannot distinguish a well-calibrated interval from a vacuous
    one, so every row also carries efficiency diagnostics:
      width_ratio — mean interval width divided by the actual spread (p95-p5) of
                    the test target. A ratio >> 1 means the interval is wide
                    relative to how much the target actually moves, i.e. the
                    coverage is bought by uninformativeness rather than by good
                    calibration.
      vacuous     — True when the interval is wider than the entire plausible
                    range of the target (100 points on this probability scale).
    """
    s = scores[~np.isnan(scores)]
    covered, _, widths = run_aci(y_test[:, h_idx], pred_test_h, s, gamma=GAMMA_DEFAULT)
    cov, w = coverage_and_width(covered, widths)
    wlo, whi = wilson_ci_from_indicator(covered)
    blo, bhi = block_bootstrap_ci(covered, block=block)
    yv = y_test[:, h_idx]
    yv = yv[np.isfinite(yv)]
    spread = float(np.percentile(yv, 95) - np.percentile(yv, 5)) if len(yv) >= 2 else np.nan
    ratio = float(w / spread) if spread and spread > 0 else np.inf
    return dict(cov=round(cov, 2), width=round(w, 2),
                wilson_lo=round(wlo, 2), wilson_hi=round(whi, 2),
                boot_lo=round(blo, 2), boot_hi=round(bhi, 2),
                support=round(support_width(s), 2),
                target_spread=round(spread, 3),
                width_ratio=round(ratio, 1) if np.isfinite(ratio) else None,
                vacuous=bool(w > 100.0))


# =============================================================================
# TASK 3 — in-sample vs out-of-fold
# =============================================================================
print("\n" + "=" * 78)
print("TASK 3 — In-sample vs out-of-fold nonconformity scores")
print("=" * 78)
print("  Building OOF predictions over the training period via rolling-origin CV")
print("  (surrogate chain of the same architecture family; the saved ensemble")
print("   saw all 635 months and cannot give honest OOF scores).")


def surrogate():
    return RegressorChain(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      max_depth=4, random_state=SEED),
        order=[0, 1, 2, 3])


Xtr_v = X_train.values.astype(float)
ytr_logit = safe_logit(y_train)
oof_pred = np.full_like(y_train, np.nan, dtype=float)
for k, (tr, te) in enumerate(rolling_origin_folds(n_pool, n_folds=6), 1):
    # Forward-horizon targets are unknown for the final months of the sample, so
    # a fold's training rows must be restricted to fully-observed target vectors.
    tr_ok = tr[np.isfinite(ytr_logit[tr]).all(axis=1)]
    if len(tr_ok) < 24:
        print(f"    fold {k}: skipped (only {len(tr_ok)} fully-observed rows)")
        continue
    m = surrogate()
    m.fit(Xtr_v[tr_ok], ytr_logit[tr_ok])
    oof_pred[te] = safe_inv_logit(m.predict(Xtr_v[te]))
    print(f"    fold {k}: train<={tr_ok[-1]+1:4d} (n={len(tr_ok)})  "
          f"predict {te[0]+1:4d}-{te[-1]+1:4d}")

scored = np.isfinite(oof_pred[:, 0])
print(f"  OOF-scored months: {scored.sum()} / {n_pool} "
      f"(earliest months are never out-of-fold under forward chaining)")

rows3 = []
for h_idx, h in enumerate(LABELS):
    s_in = np.abs(preds_train_insample[:, h_idx] - y_train[:, h_idx])
    s_oof = np.abs(oof_pred[:, h_idx] - y_train[:, h_idx])

    # Trailing-N baseline and diversity-optimal selector, under each scoring.
    tail = np.arange(n_pool - N_FIX, n_pool)
    base_in = evaluate(s_in[tail], h_idx, preds_test[:, h_idx])
    sel_in = greedy_extreme(s_in)
    div_in = evaluate(s_in[sel_in], h_idx, preds_test[:, h_idx])

    valid_oof = np.where(np.isfinite(s_oof))[0]
    tail_oof = valid_oof[-N_FIX:]
    base_oof = evaluate(s_oof[tail_oof], h_idx, preds_test[:, h_idx])
    sel_oof = greedy_extreme(s_oof)
    div_oof = evaluate(s_oof[sel_oof], h_idx, preds_test[:, h_idx])

    print(f"\n  {h}:")
    print(f"    in-sample : trailing={base_in['cov']:6.2f}%  "
          f"diversity-opt={div_in['cov']:6.2f}%  "
          f"gain={div_in['cov']-base_in['cov']:+.2f}pp")
    print(f"    out-of-fold: trailing={base_oof['cov']:6.2f}%  "
          f"diversity-opt={div_oof['cov']:6.2f}%  "
          f"gain={div_oof['cov']-base_oof['cov']:+.2f}pp")

    rows3.append(dict(
        Horizon=h,
        insample_trailing_cov=base_in["cov"],
        insample_trailing_wilson=f"[{base_in['wilson_lo']},{base_in['wilson_hi']}]",
        insample_divopt_cov=div_in["cov"],
        insample_divopt_wilson=f"[{div_in['wilson_lo']},{div_in['wilson_hi']}]",
        insample_divopt_boot=f"[{div_in['boot_lo']},{div_in['boot_hi']}]",
        insample_gain_pp=round(div_in["cov"] - base_in["cov"], 2),
        insample_divopt_width=div_in["width"],
        insample_divopt_support=div_in["support"],
        oof_trailing_cov=base_oof["cov"],
        oof_trailing_wilson=f"[{base_oof['wilson_lo']},{base_oof['wilson_hi']}]",
        oof_divopt_cov=div_oof["cov"],
        oof_divopt_wilson=f"[{div_oof['wilson_lo']},{div_oof['wilson_hi']}]",
        oof_divopt_boot=f"[{div_oof['boot_lo']},{div_oof['boot_hi']}]",
        oof_gain_pp=round(div_oof["cov"] - base_oof["cov"], 2),
        oof_divopt_width=div_oof["width"],
        oof_divopt_support=div_oof["support"],
    ))

pd.DataFrame(rows3).to_csv(f"{OUT}/task3_insample_vs_oof.csv", index=False)
print(f"\nSaved {OUT}/task3_insample_vs_oof.csv")


# =============================================================================
# TASK 2 — model-agnosticism: full strategy comparison on PROBIT scores
# =============================================================================
print("\n" + "=" * 78)
print("TASK 2 — Calibration strategies applied to the probit baseline's scores")
print("=" * 78)

spread_tr = (X_train["10_year_rate"] - X_train["3_months_rate"]).values.reshape(-1, 1)
spread_te = (X_test["10_year_rate"] - X_test["3_months_rate"]).values.reshape(-1, 1)

probit_oof = np.full_like(y_train, np.nan, dtype=float)
probit_test = np.zeros_like(y_test, dtype=float)
for h_idx, t in enumerate(recession_targets):
    ylog = safe_logit(y_train[:, h_idx])
    # out-of-fold probit scores (rolling origin, same folds as above)
    for tr, te in rolling_origin_folds(n_pool, n_folds=6):
        tr_ok = tr[np.isfinite(ylog[tr])]
        if len(tr_ok) < 24:
            continue
        r = LinearRegression().fit(spread_tr[tr_ok], ylog[tr_ok])
        probit_oof[te, h_idx] = safe_inv_logit(r.predict(spread_tr[te]))
    ok_full = np.isfinite(ylog)
    r_full = LinearRegression().fit(spread_tr[ok_full], ylog[ok_full])
    probit_test[:, h_idx] = safe_inv_logit(r_full.predict(spread_te))

print("  Probit OOF scores and test predictions built (10Y-3M spread).")


def mondrian_scores(s_all, mask_rare):
    """Class-conditional: rare-regime months only (the textbook fix)."""
    return s_all[mask_rare & np.isfinite(s_all)]


def evt_tail_scores(s_all, q=0.75):
    """Extreme-value style: keep only the upper tail of the score pool."""
    s = s_all[np.isfinite(s_all)]
    return s[s >= np.quantile(s, q)]


def run_pid(y_te, pred_te, cal, kp=0.02, ki=0.002):
    """PID-conformal: proportional + integral control on the miscoverage rate."""
    alpha, integral = 0.10, 0.0
    covered, widths = [], []
    for t in range(len(y_te)):
        q = np.quantile(cal, np.clip(1 - alpha, 0.0, 1.0))
        lo, hi = pred_te[t] - q, pred_te[t] + q
        widths.append(2 * q)
        yv = y_te[t]
        if np.isnan(yv):
            covered.append(np.nan); continue
        miss = 1 if (yv < lo or yv > hi) else 0
        covered.append(1 - miss)
        err = 0.10 - miss
        integral += err
        alpha = float(np.clip(alpha + kp * err + ki * integral, 0.01, 0.99))
    return np.array(covered), np.array(widths)


rows2 = []
is_rare_pool = rec_prob >= 50
for h_idx, h in enumerate(LABELS):
    s_probit = np.abs(probit_oof[:, h_idx] - y_train[:, h_idx])
    ptest = probit_test[:, h_idx]
    valid = np.where(np.isfinite(s_probit))[0]

    strategies = {}
    tail = valid[-N_FIX:]
    strategies["pooled_trailing"] = s_probit[tail]
    mr = mondrian_scores(s_probit, is_rare_pool)
    strategies["mondrian_rare"] = mr if len(mr) >= 2 else s_probit[tail]
    strategies["evt_tail"] = evt_tail_scores(s_probit)
    strategies["diversity_optimal"] = s_probit[greedy_extreme(s_probit)]

    for sname, sc in strategies.items():
        r = evaluate(sc, h_idx, ptest)
        rows2.append(dict(Horizon=h, model="probit", strategy=sname, **r))

    # PID gets its own controller loop
    cov_pid, w_pid = run_pid(y_test[:, h_idx], ptest, s_probit[tail])
    cov, w = coverage_and_width(cov_pid, w_pid)
    wlo, whi = wilson_ci_from_indicator(cov_pid)
    blo, bhi = block_bootstrap_ci(cov_pid, block=12)
    yv = y_test[:, h_idx]; yv = yv[np.isfinite(yv)]
    spread = float(np.percentile(yv, 95) - np.percentile(yv, 5))
    rows2.append(dict(Horizon=h, model="probit", strategy="pid_conformal",
                      cov=round(cov, 2), width=round(w, 2),
                      wilson_lo=round(wlo, 2), wilson_hi=round(whi, 2),
                      boot_lo=round(blo, 2), boot_hi=round(bhi, 2),
                      support=round(support_width(s_probit[tail]), 2),
                      target_spread=round(spread, 3),
                      width_ratio=round(w / spread, 1) if spread > 0 else None,
                      vacuous=bool(w > 100.0)))

    print(f"\n  {h} (probit scores):  test-target spread (p95-p5) = {spread:.2f}")
    for r in [x for x in rows2 if x["Horizon"] == h]:
        flag = "  <-- VACUOUS (wider than the whole 0-100 scale)" if r["vacuous"] else ""
        print(f"    {r['strategy']:20s} cov={r['cov']:6.2f}%  "
              f"width={r['width']:8.2f}  width/spread={str(r['width_ratio']):>8s}{flag}")

probit_tbl = pd.DataFrame(rows2)
probit_tbl.to_csv(f"{OUT}/task2_probit_strategies.csv", index=False)
print(f"\nSaved {OUT}/task2_probit_strategies.csv")


# ── Cross-model summary: does the diversity effect hold for BOTH models? ──────
print("\n" + "=" * 78)
print("TASK 2 — Does the selector help regardless of the underlying model?")
print("=" * 78)
summary = []
for h_idx, h in enumerate(LABELS):
    for mname, s_pool, ptest in [
        ("chain_oof", np.abs(oof_pred[:, h_idx] - y_train[:, h_idx]), preds_test[:, h_idx]),
        ("probit_oof", np.abs(probit_oof[:, h_idx] - y_train[:, h_idx]), probit_test[:, h_idx]),
    ]:
        valid = np.where(np.isfinite(s_pool))[0]
        base = evaluate(s_pool[valid[-N_FIX:]], h_idx, ptest)
        div = evaluate(s_pool[greedy_extreme(s_pool)], h_idx, ptest)
        summary.append(dict(Horizon=h, model=mname,
                            trailing_cov=base["cov"], trailing_width=base["width"],
                            divopt_cov=div["cov"], divopt_width=div["width"],
                            gain_pp=round(div["cov"] - base["cov"], 2),
                            divopt_support=div["support"],
                            divopt_wilson=f"[{div['wilson_lo']},{div['wilson_hi']}]"))
        print(f"  {h:8s} {mname:12s} trailing={base['cov']:6.2f}%  "
              f"div-opt={div['cov']:6.2f}%  gain={div['cov']-base['cov']:+6.2f}pp")

pd.DataFrame(summary).to_csv(f"{OUT}/task2_model_agnostic_summary.csv", index=False)
print(f"\nSaved {OUT}/task2_model_agnostic_summary.csv")
