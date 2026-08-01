"""
TASK 8 (climate) — persistence and climatology baselines.

Our per-region-month storm-intensity target is a custom aggregate built for this
paper, not a standard forecasting task with an established point-prediction
literature. The defensible convention in the meteorological forecasting
literature — used by WeatherBench (Rasp et al., 2020, "WeatherBench: A benchmark
dataset for data-driven weather forecasting", JAMES 12(11)), TCBench, and NOAA's
own hurricane guidance suite — is a baseline tier of:

  persistence   next month's regional intensity = this month's
  climatology   next month's regional intensity = that region's historical mean
                for the same calendar month (the "climatological normal")

This is the field's own baseline convention, not a novel choice made for this
paper. A tropical-cyclone intensity model (SHIPS/SHIFOR) is deliberately NOT
attempted: those predict per-storm intensity from storm-centred predictors, and
forcing one onto a monthly regional aggregate would be a mismatch dressed up as
a comparison — flagged here the same way CPTC was flagged in Task 7.

Leakage discipline: the climatological normal is computed from the pre-test
partition only, consistent with the causal-window discipline used elsewhere in
this paper. Both baselines are evaluated out-of-fold.

Outputs:
  fix-reg/task8_climate_literature_baselines.csv
  fix-reg/task8_climate_point_prediction.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_common import support_width  # noqa: E402
from task7_baseline_horse_race import run_all_strategies  # noqa: E402

PANEL = "data/domains/noaa_panel.csv"
OUT = "fix-reg"
N_CAL = 254
MIN_YEARS_FOR_NORMAL = 5   # a region-month needs this many past years to be stable

print("=" * 78)
print("TASK 8 (climate) — persistence & climatology baselines")
print("=" * 78)

panel = pd.read_csv(PANEL)
panel = panel.sort_values(["region", "year", "month"]).reset_index(drop=True)
panel["target"] = 100.0 * panel["n_signif"] / panel["n_events"].clip(lower=1)
print(f"  Panel: {len(panel):,} region-months, {panel['region'].nunique()} regions, "
      f"{panel['year'].min()}-{panel['year'].max()}")

# Temporal ordering, identical split convention to domain_climate.py
panel["t"] = panel["year"] * 12 + panel["month"]
panel = panel.sort_values("t", kind="stable").reset_index(drop=True)

n = len(panel)
n_test = int(0.15 * n)
pool_idx = np.arange(0, n - n_test)
test_idx = np.arange(n - n_test, n)

y_all = panel["target"].values.astype(float)
RARE_THRESH = float(np.percentile(y_all, 90))
is_rare_all = y_all >= RARE_THRESH

y_pool, y_test = y_all[pool_idx], y_all[test_idx]
rare_pool = is_rare_all[pool_idx]
N_CAL_EFF = min(N_CAL, int(0.6 * len(pool_idx)))
print(f"  Calibration pool: {len(y_pool):,} ({rare_pool.sum()} rare) | "
      f"Test: {len(y_test):,} | N_cal={N_CAL_EFF}")

region = panel["region"].values
month = panel["month"].values

# ── Baseline 1: persistence ──────────────────────────────────────────────────
# Prediction for a region-month = that region's previous observed month.
persist = panel.groupby("region")["target"].shift(1).values

# ── Baseline 2: climatology ─────────────────────────────────────────────────
# Region x calendar-month historical mean, computed EXPANDING over the past only
# (so no future information, and no test data, enters any prediction).
clim = np.full(n, np.nan)
hist = {}
counts = {}
for i in range(n):
    key = (region[i], month[i])
    if key in hist and counts[key] >= 1:
        clim[i] = hist[key] / counts[key]
    # update AFTER predicting, so the current observation never informs itself
    hist[key] = hist.get(key, 0.0) + y_all[i]
    counts[key] = counts.get(key, 0) + 1

# Stability check: how many test region-months have a normal built from enough
# history? Reported rather than silently accepted (task guardrail).
years_of_hist = np.array([counts.get((region[i], month[i]), 0) for i in test_idx])
thin = int((years_of_hist < MIN_YEARS_FOR_NORMAL).sum())
print(f"\n  Climatological-normal stability: {thin} of {len(test_idx)} test "
      f"region-months have < {MIN_YEARS_FOR_NORMAL} years of history "
      f"({100*thin/len(test_idx):.1f}%)")
print(f"  Missing persistence predictions (first month per region): "
      f"{int(np.isnan(persist).sum())}")

summary_rows, calib_rows = [], []
for name, pred_all in [("persistence", persist), ("climatology", clim)]:
    print("\n" + "-" * 78)
    print(f"BASELINE: {name}")
    print("-" * 78)

    scores_all = np.abs(pred_all[pool_idx] - y_pool)
    pred_test = pred_all[test_idx]

    # Any test point without a prediction cannot be scored; drop those and say so.
    ok = np.isfinite(pred_test) & np.isfinite(y_test)
    dropped = int((~ok).sum())
    if dropped:
        print(f"  Dropping {dropped} test points with no baseline prediction")
    y_te, p_te = y_test[ok], pred_test[ok]
    te_month = month[test_idx][ok]

    mae_oof = float(np.nanmean(scores_all))
    mae_test = float(np.nanmean(np.abs(p_te - y_te)))
    n_scored = int(np.isfinite(scores_all).sum())
    print(f"  OOF MAE = {mae_oof:.3f} | test MAE = {mae_test:.3f} "
          f"| scored pool points = {n_scored}")
    print(f"  score support (p95-p5) = {support_width(scores_all):.3f}")

    summary_rows.append(dict(domain="Climate", baseline=name,
                             oof_mae=round(mae_oof, 3),
                             test_mae=round(mae_test, 3),
                             n_pool_scored=n_scored, n_test=int(ok.sum()),
                             score_support=round(support_width(scores_all), 3)))

    # Mondrian regime: named-storm season (Jun-Nov), same as Task 7.
    te_regime = np.isin(te_month, [6, 7, 8, 9, 10, 11])

    calib_rows += run_all_strategies(
        "Climate", name, "region-month", "out-of-fold", scores_all,
        rare_pool, y_te, p_te, te_regime, block=12, horizon_steps=1)

cal = pd.DataFrame(calib_rows)
summ = pd.DataFrame(summary_rows)
cal.to_csv(f"{OUT}/task8_climate_literature_baselines.csv", index=False)
summ.to_csv(f"{OUT}/task8_climate_point_prediction.csv", index=False)

print("\n" + "=" * 78)
print("CALIBRATION COMPARISON on literature-baseline scores")
print("=" * 78)
print(cal[["model", "strategy", "n", "coverage", "wilson_lo", "wilson_hi",
           "mean_width"]].to_string(index=False))
print(f"\nSaved {OUT}/task8_climate_literature_baselines.csv")
print(f"Saved {OUT}/task8_climate_point_prediction.csv")
