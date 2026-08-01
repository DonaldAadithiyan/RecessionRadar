"""
TASK 9 PART A — Extend the recession test window with real new FRED data.

The paper's six-month coverage verdict is "undetermined" at n=59 scored months.
More ground truth now exists than when the original window was built: FRED's
RECPROUSM156N runs through May 2026, versus the paper's window ending May 2025.

THIS IS AN EXTENSION, NOT A RETRAIN. Stage 1 / Stage 2 are not refit. The saved
stacking ensemble scores the new months exactly as it scored the original 65,
which is pure out-of-sample extension of a fixed model. The only refitting is
the out-of-fold selector arm's rolling-origin procedure, extended forward the
same way it already covered the original window — the same kind of refit that
produced the published OOF numbers, not a new kind.

Feature construction replicates the original pipeline exactly:
  - STL(seasonal=13, period=12) for _trend / _residual
  - rolling stats computed on .shift(1) (no same-month leakage)
  - diffs / pct_change / lags as in fix-reg/feature_engineering.ipynb
  - _anomaly flags at +/-3 sigma of the STL residual, using thresholds fit on
    the ORIGINAL training partition only (never refit on new data)

Outputs:
  fix-reg/task9a_extended_recession_window.csv   full 8-strategy comparison
  fix-reg/task9a_extension_data.csv              the rebuilt extended panel
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import requests
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from ensemble_stubs import recession_targets, safe_logit, safe_inv_logit  # noqa: E402

from domain_common import rolling_origin_folds  # noqa: E402
from task7_baseline_horse_race import run_all_strategies, N_FIX  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.multioutput import RegressorChain  # noqa: E402

DATA_PATH = "data/fix/feature_selected_reg_full.csv"
MODEL_PATH = "fix-reg/models/full_chain_stacking.pkl"
OUT = "fix-reg"
SPLIT = "2020-01-01"
SEED = 5
LABELS = ["Current", "1M", "3M", "6M"]

# FRED series -> the base indicator name used in the feature table
FRED_SERIES = {
    "RECPROUSM156N": "recession_probability",
    "CPIAUCSL": "CPI",
    "INDPRO": "INDPRO",
    "UNRATE": "unemployment_rate",
    "PCU3312103312100": "PPI",
    "SPASTT01USM661N": "share_price",
    "USALOLITOAASTSAM": "OECD_CLI_index",
    "UMCSENT": "CSI_index",
    "IRLTLT01USM156N": "10_year_rate",
    "DTB1YR": "1_year_rate",
    "DTB3": "3_months_rate",
    "DTB6": "6_months_rate",
    "A939RX0Q048SBEA": "gdp_per_capita",
}

print("=" * 78)
print("TASK 9A — Extending the recession test window with new FRED observations")
print("=" * 78)

api_key = None
for line in open(".env"):
    if line.startswith("FRED_API_KEY"):
        api_key = line.split("=", 1)[1].strip()
assert api_key, "FRED_API_KEY not found in .env"


def fetch(series_id):
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params=dict(series_id=series_id, api_key=api_key,
                                 file_type="json",
                                 observation_start="1967-01-01"), timeout=60)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    # Daily series (the Treasury bills) are averaged to month-start, exactly as
    # the original data-collection notebook did.
    d["date"] = d["date"].dt.to_period("M").dt.to_timestamp()
    return d.groupby("date", as_index=False)["value"].mean()


print("\nFetching series from FRED...")
raw = {}
for sid, name in FRED_SERIES.items():
    s = fetch(sid)
    raw[name] = s.rename(columns={"value": name})
    print(f"  {name:26s} ({sid:18s}) -> {len(s):4d} obs, "
          f"latest {s['date'].max().date()}")

# ── Assemble the base monthly panel ──────────────────────────────────────────
panel = None
for name, d in raw.items():
    panel = d if panel is None else panel.merge(d, on="date", how="outer")
panel = panel.sort_values("date").reset_index(drop=True)

# GDP per capita is quarterly -> forward-fill to monthly (as originally done)
panel["gdp_per_capita"] = panel["gdp_per_capita"].ffill()
panel = panel[panel["date"] >= "1967-02-01"].reset_index(drop=True)

# Forward-shifted targets, exactly as the original pipeline built them
panel["1_month_recession_probability"] = panel["recession_probability"].shift(-1)
panel["3_month_recession_probability"] = panel["recession_probability"].shift(-3)
panel["6_month_recession_probability"] = panel["recession_probability"].shift(-6)

print(f"\n  Base panel: {len(panel)} months, "
      f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

orig = pd.read_csv(DATA_PATH)
orig["date"] = pd.to_datetime(orig["date"])
orig_end = orig["date"].max()
new_months = panel[panel["date"] > orig_end]
print(f"  Original data ends {orig_end.date()}; "
      f"{len(new_months)} new months available "
      f"({new_months['date'].min().date()} -> {new_months['date'].max().date()})"
      if len(new_months) else "  No new months available")


# ── Rebuild engineered features on the FULL extended panel ───────────────────
INDICATORS = ["CPI", "INDPRO", "unemployment_rate", "share_price", "PPI",
              "OECD_CLI_index", "CSI_index", "gdp_per_capita",
              "10_year_rate", "1_year_rate", "3_months_rate", "6_months_rate"]

print("\nRebuilding engineered features (STL + rolling + diffs + anomalies)...")
fe = panel.copy()
for col in INDICATORS:
    s = fe[col].ffill().bfill()
    if s.notna().sum() >= 24:
        stl = STL(s, seasonal=13, period=12).fit()
        fe[f"{col}_trend"] = stl.trend
        fe[f"{col}_residual"] = stl.resid

    fe[f"{col}_diff1"] = fe[col].diff(1)
    fe[f"{col}_diff3"] = fe[col].diff(3)
    fe[f"{col}_pct_change1"] = fe[col].pct_change(1)
    for w in (3, 6, 12):
        sh = fe[col].shift(1)
        fe[f"{col}_rollstd{w}"] = sh.rolling(w).std()
        fe[f"{col}_rollmax{w}"] = sh.rolling(w).max()
        fe[f"{col}_rollmin{w}"] = sh.rolling(w).min()
    fe[f"{col}_lag6"] = fe[col].shift(6)

fe["PPI_CPI_diff"] = fe["PPI"] - fe["CPI"]

# Anomaly flags: thresholds from the ORIGINAL TRAINING partition only.
train_mask = fe["date"] < SPLIT
for col in ["CPI", "unemployment_rate", "share_price",
            "3_months_rate", "6_months_rate", "10_year_rate"]:
    rcol = f"{col}_residual"
    if rcol not in fe:
        continue
    mu = fe.loc[train_mask, rcol].mean()
    sd = fe.loc[train_mask, rcol].std()
    fe[f"{col}_anomaly"] = (((fe[rcol] < mu - 3 * sd) |
                             (fe[rcol] > mu + 3 * sd)).astype(int))

# Keep exactly the columns the saved model expects, in the same order.
feat_cols = [c for c in orig.columns if c not in recession_targets + ["date"]]
missing = [c for c in feat_cols if c not in fe.columns]
if missing:
    print(f"  WARNING: {len(missing)} expected features could not be rebuilt: "
          f"{missing[:8]}")
    for c in missing:
        fe[c] = np.nan

ext = fe[["date"] + feat_cols + recession_targets].copy()
ext.to_csv(f"{OUT}/task9a_extension_data.csv", index=False)


def clean(d):
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.ffill().bfill().fillna(0)


# ── Sanity check: do rebuilt features reproduce the originals? ───────────────
print("\n" + "-" * 78)
print("FEATURE RECONSTRUCTION CHECK (rebuilt vs original, overlapping months)")
print("-" * 78)
merged = orig[["date"] + feat_cols].merge(
    ext[["date"] + feat_cols], on="date", suffixes=("_o", "_n"))
corrs = []
for c in feat_cols:
    a, b = merged[f"{c}_o"], merged[f"{c}_n"]
    if a.std() > 0 and b.std() > 0:
        corrs.append((c, float(a.corr(b))))
corrs.sort(key=lambda x: x[1])
med = float(np.median([c for _, c in corrs]))
print(f"  median correlation across {len(corrs)} features: {med:.4f}")
print("  weakest 6:")
for c, v in corrs[:6]:
    print(f"    {c:34s} r={v:+.4f}")
print("\n  NOTE: STL is a global smoother, so re-running it over a longer panel")
print("  shifts trend/residual features even on overlapping months. The same")
print("  applies to the quarterly GDP series' interpolation. The extended")
print("  window is therefore evaluated on a CONSISTENTLY rebuilt panel")
print("  end-to-end (original months and new ones alike), so it is internally")
print("  like-for-like — but see the decomposition below before comparing any")
print("  of these numbers directly against the published ones.")

# ── Decompose: how much of any change is "rebuilt panel" vs "more data"? ─────
# Scoring the rebuilt panel on the ORIGINAL 65-month window isolates the
# feature-reconstruction effect from the effect of the 14 new months.
print("\n" + "-" * 78)
print("EFFECT DECOMPOSITION (rebuilt-panel effect vs new-data effect)")
print("-" * 78)

# ── Score with the SAVED model (no retraining) ───────────────────────────────
with open(MODEL_PATH, "rb") as f:
    ensemble = pickle.load(f)
print("\nSaved ensemble loaded (not refit).")

ext_train = ext[ext["date"] < SPLIT].reset_index(drop=True)
ext_test = ext[ext["date"] >= SPLIT].reset_index(drop=True)
X_tr = clean(ext_train[feat_cols])
X_te = clean(ext_test[feat_cols])
y_tr = ext_train[recession_targets].values
y_te = ext_test[recession_targets].values

preds_tr_insample = ensemble.predict(X_tr)
preds_te = ensemble.predict(X_te)
print(f"  Extended test window: {len(ext_test)} months "
      f"({ext_test['date'].min().date()} -> {ext_test['date'].max().date()})")
for h_idx, h in enumerate(LABELS):
    print(f"    {h:8s} scored n = {int(np.isfinite(y_te[:, h_idx]).sum())}")

# ── Out-of-fold arm: rolling origin extended forward, same procedure ─────────
print("\nBuilding out-of-fold scores (rolling-origin, extended forward)...")


def surrogate():
    return RegressorChain(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      max_depth=4, random_state=SEED),
        order=[0, 1, 2, 3])


Xtr_v = X_tr.values.astype(float)
ytr_logit = safe_logit(y_tr)
n_pool = len(ext_train)
oof_pred = np.full_like(y_tr, np.nan, dtype=float)
for k, (tr, te) in enumerate(rolling_origin_folds(n_pool, n_folds=6), 1):
    tr_ok = tr[np.isfinite(ytr_logit[tr]).all(axis=1)]
    if len(tr_ok) < 24:
        continue
    m = surrogate()
    m.fit(Xtr_v[tr_ok], ytr_logit[tr_ok])
    oof_pred[te] = safe_inv_logit(m.predict(Xtr_v[te]))
print(f"  OOF-scored training months: {int(np.isfinite(oof_pred[:,0]).sum())}"
      f" / {n_pool}")

# ── Regime check: is the new period a third regime? ─────────────────────────
print("\n" + "-" * 78)
print("REGIME CHECK on the new months")
print("-" * 78)
newm = ext_test[ext_test["date"] > orig_end]
if len(newm):
    print(f"  New months: {len(newm)} "
          f"({newm['date'].min().date()} -> {newm['date'].max().date()})")
    print(f"  recession_probability: mean={newm['recession_probability'].mean():.3f} "
          f"max={newm['recession_probability'].max():.3f}")
    old = ext_test[ext_test["date"] <= orig_end]
    print(f"  vs original window   : mean={old['recession_probability'].mean():.3f} "
          f"max={old['recession_probability'].max():.3f}")
    for c in ["3_months_rate", "unemployment_rate", "10_year_rate"]:
        print(f"  {c:20s} new mean={newm[c].mean():8.3f} | "
              f"orig-window mean={old[c].mean():8.3f}")

# ── Full 8-strategy comparison at the new n ─────────────────────────────────
rec_prob_tr = ext_train["recession_probability"].values
te_regime = ext_test["recession_probability"].values >= 50
is_rare = rec_prob_tr >= 50
horizon_steps = {"Current": 1, "1M": 1, "3M": 3, "6M": 6}

rows = []
for scoring, pool_preds in [("in-sample", preds_tr_insample),
                            ("out-of-fold", oof_pred)]:
    for h_idx, h in enumerate(LABELS):
        s = np.abs(pool_preds[:, h_idx] - y_tr[:, h_idx])
        rows += run_all_strategies(
            "Recession-extended", "stacking-chain", h, scoring, s, is_rare,
            y_te[:, h_idx], preds_te[:, h_idx], te_regime,
            block=12, horizon_steps=horizon_steps[h])

    # Same strategies, but restricted to the ORIGINAL 65-month window on the
    # REBUILT panel. Differences vs the published numbers are then purely the
    # feature-reconstruction effect; differences vs the extended rows above are
    # purely the new-data effect.
    if scoring == "out-of-fold":
        orig_mask = (ext_test["date"] <= orig_end).values
        for h_idx, h in enumerate(LABELS):
            s = np.abs(pool_preds[:, h_idx] - y_tr[:, h_idx])
            rows += [dict(r, scoring="out-of-fold-origwindow")
                     for r in run_all_strategies(
                         "Recession-extended", "stacking-chain", h,
                         "out-of-fold", s, is_rare,
                         y_te[orig_mask, h_idx], preds_te[orig_mask, h_idx],
                         te_regime[orig_mask], block=12,
                         horizon_steps=horizon_steps[h])]

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/task9a_extended_recession_window.csv", index=False)

print("\n" + "-" * 78)
print("DECOMPOSITION at 6M: published (orig panel, n=59) vs rebuilt panel on")
print("the same 65-month window vs rebuilt panel extended (n=71)")
print("-" * 78)
PUBLISHED_6M = {"pooled_trailing": (84.75, 59), "diversity_optimal": (96.61, 59)}
for strat, (pub_cov, pub_n) in PUBLISHED_6M.items():
    a = res[(res.scoring == "out-of-fold-origwindow") &
            (res.horizon == "6M") & (res.strategy == strat)]
    b = res[(res.scoring == "out-of-fold") &
            (res.horizon == "6M") & (res.strategy == strat)]
    if a.empty or b.empty:
        continue
    a, b = a.iloc[0], b.iloc[0]
    print(f"  {strat:18s} published={pub_cov:6.2f} (n={pub_n})  "
          f"rebuilt/same-window={a['coverage']:6.2f} (n={int(a['n'])})  "
          f"rebuilt/extended={b['coverage']:6.2f} (n={int(b['n'])})")
    print(f"    {'':18s} reconstruction effect={a['coverage']-pub_cov:+6.2f}pp  "
          f"new-data effect={b['coverage']-a['coverage']:+6.2f}pp")

print("\n" + "=" * 78)
print("EXTENDED-WINDOW RESULTS (out-of-fold)")
print("=" * 78)
oo = res[res["scoring"] == "out-of-fold"]
print(oo[["horizon", "strategy", "n", "coverage", "wilson_lo", "wilson_hi",
          "mean_width"]].to_string(index=False))

print("\n" + "=" * 78)
print("SIX-MONTH VERDICT")
print("=" * 78)
for scoring in ["in-sample", "out-of-fold"]:
    # exact match only — excludes the "out-of-fold-origwindow" decomposition rows
    d6 = res[(res["scoring"] == scoring) & (res["horizon"] == "6M")]
    d6 = d6[d6["scoring"] == scoring]
    for strat in ["pooled_trailing", "diversity_optimal"]:
        r = d6[d6["strategy"] == strat]
        if r.empty:
            continue
        r = r.iloc[0]
        if r["wilson_lo"] > 90:
            v = "CLEARS 90% (interval entirely above)"
        elif r["wilson_hi"] < 90:
            v = "FALLS SHORT of 90% (interval entirely below)"
        else:
            v = "STRADDLES 90% — still undetermined"
        print(f"  {scoring:12s} {strat:18s} n={int(r['n'])} "
              f"cov={r['coverage']:6.2f}% [{r['wilson_lo']:.2f},"
              f"{r['wilson_hi']:.2f}] -> {v}")

print(f"\nSaved {OUT}/task9a_extended_recession_window.csv")
