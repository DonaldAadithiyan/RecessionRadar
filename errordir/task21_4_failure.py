"""
TASK 21, Item 4 — stress-test where beta is confidently wrong.

Runs REGARDLESS of Item 3's outcome (spec guardrail). The method's specific new
risk, not shared by any uniform-width baseline: a confidently-wrong LOW
difficulty projection produces an artificially NARROW interval — actively worse
than the selector for that case, not merely neutral.

Identifies the prediction-time cases with the lowest difficulty projection but
the largest realized error, and checks whether they cluster on anything
structural (period, regime, model state) rather than being scattered noise.

Output: errordir/task21_4_failures.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg"))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
import task21_2_scaling as SC  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_train as y_pool,
    y_test, oof_pred as preds_pool_oof, preds_test, n_pool, test_df,
)
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = "errordir"
H_IDX, H = 3, "6M"
FIT_FRAC = 0.60

X_train, X_test = X_train_df.values, X_test_df.values
n_fit = int(round(FIT_FRAC * n_pool))
FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)
abs_err = np.abs(preds_pool_oof[:, H_IDX] - y_pool[:, H_IDX])
fit_ok = FIT_IDX[np.isfinite(abs_err[FIT_IDX])]
cal_ok = CAL_IDX[np.isfinite(abs_err[CAL_IDX])]

scaler = StandardScaler().fit(X_train[fit_ok])
reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler.transform(X_train[fit_ok]),
                                                 abs_err[fit_ok])
beta = reg.coef_.astype(float)
proj_cal = scaler.transform(X_train[cal_ok]) @ beta
proj_test = scaler.transform(X_test) @ beta
u = SC.fit_cdf(proj_cal)(proj_test)
mult = SC.multiplier(u)

realized = np.abs(preds_test[:, H_IDX] - y_test[:, H_IDX])
dates = pd.to_datetime(test_df["date"].values)

d = pd.DataFrame(dict(date=dates, proj_rank=u, multiplier=mult,
                      realized_error=realized,
                      pred=preds_test[:, H_IDX], actual=y_test[:, H_IDX]))
d = d[np.isfinite(d.realized_error)].reset_index(drop=True)

# "Confidently wrong" = low projected difficulty (bottom tercile of rank) but
# large realized error (top tercile). Thresholds fixed by tercile, not tuned.
lo_rank = d["proj_rank"].quantile(1 / 3)
hi_err = d["realized_error"].quantile(2 / 3)
d["confidently_wrong"] = (d["proj_rank"] <= lo_rank) & (d["realized_error"] >= hi_err)
d["year"] = d["date"].dt.year
d.to_csv(f"{OUT}/task21_4_failures.csv", index=False)

print("=" * 96)
print(f"TASK 21 Item 4 — where beta is confidently wrong ({H})")
print("=" * 96)
print(f"  n_test={len(d)}   low-rank threshold={lo_rank:.3f}   "
      f"high-error threshold={hi_err:.3f}")
cw = d[d.confidently_wrong]
print(f"  confidently-wrong cases: {len(cw)} / {len(d)} ({100*len(cw)/len(d):.1f}%)\n")

if len(cw):
    print("  The cases (narrow interval issued, large error realized):")
    for r in cw.itertuples():
        print(f"    {r.date.date()}  rank={r.proj_rank:.3f}  mult={r.multiplier:.3f}  "
              f"realized_err={r.realized_error:8.2f}  pred={r.pred:7.2f} "
              f"actual={r.actual:7.2f}")

    print("\n  CLUSTERING CHECK — by year:")
    byyear = d.groupby("year").agg(n=("proj_rank", "size"),
                                   n_cw=("confidently_wrong", "sum"),
                                   mean_err=("realized_error", "mean"))
    byyear["pct_cw"] = (100 * byyear.n_cw / byyear.n).round(1)
    print(byyear.to_string())

    yrs = cw["year"].value_counts()
    top_yr = yrs.index[0]
    conc = 100 * yrs.iloc[0] / len(cw)
    print(f"\n  Most failures fall in {top_yr}: {yrs.iloc[0]}/{len(cw)} "
          f"({conc:.0f}%) -> "
          f"{'CLUSTERED (nameable failure mode)' if conc >= 50 else 'scattered'}")

    print(f"\n  Mean realized error in confidently-wrong cases: "
          f"{cw.realized_error.mean():.2f}")
    print(f"  Mean realized error elsewhere:                   "
          f"{d[~d.confidently_wrong].realized_error.mean():.2f}")
    print(f"  Mean multiplier issued in those cases: {cw.multiplier.mean():.3f} "
          f"(vs {d[~d.confidently_wrong].multiplier.mean():.3f} elsewhere)")
print(f"\nSaved {OUT}/task21_4_failures.csv")


# ── The intended failure mode, tested DIRECTLY ─────────────────────────────
# The tercile definition above is distorted by the covariate drift: because
# 15% of test ranks saturate at 1.0, the "bottom tercile" threshold is 0.984,
# so the flagged cases actually received WIDE multipliers (1.21-1.24), not
# narrow ones. That is not the failure mode this item exists to characterize.
#
# The real risk is: an interval NARROWED by the method (multiplier < 1.0) that
# then missed. Test that directly, and compare against what the unscaled
# baseline would have done for the same months.
print("\n" + "=" * 96)
print("DIRECT TEST OF THE ACTUAL RISK: intervals the method NARROWED (mult<1)")
print("=" * 96)
narrowed = d[d.multiplier < 1.0]
print(f"  months narrowed by the method: {len(narrowed)}/{len(d)} "
      f"({100*len(narrowed)/len(d):.1f}%)")
if len(narrowed):
    print(f"  mean realized error when narrowed: {narrowed.realized_error.mean():.2f}")
    print(f"  mean realized error otherwise:     "
          f"{d[d.multiplier >= 1.0].realized_error.mean():.2f}")
    print(f"  -> narrowing is {'CORRECTLY targeted (lower error)' if narrowed.realized_error.mean() < d[d.multiplier>=1.0].realized_error.mean() else 'MIS-targeted (higher error)'}")
    print("\n  narrowed months and their years:")
    print(narrowed.groupby("year").agg(n=("multiplier", "size"),
                                       mean_mult=("multiplier", "mean"),
                                       mean_err=("realized_error", "mean")).to_string())
else:
    print("  NONE — under the observed drift the method never narrowed an interval,")
    print("  so its distinctive new risk (artificially narrow intervals) never")
    print("  materialized on this test period. The risk is UNTESTED here, not absent.")
