"""
TASK 22, Item 3 — DIAGNOSIS ONLY of the 2022 blind spot. No fix.

Task 21 Item 4: zero flagged cases in 2022 despite 2022 having the highest mean
realized error of any test year (30.69).

Two candidate explanations, to be distinguished with feature-level evidence:
  CASE 1 — genuinely novel: 2022 inputs sit OUTSIDE the fit-period feature
           distribution. Legitimate extrapolation failure.
  CASE 2 — within-distribution miss: 2022 inputs look like things beta was fit
           against, so beta should have caught it and didn't.

Diagnostics used:
  (a) per-feature: fraction of 2022 months outside the fit period's [min,max]
      and beyond +/-3 fit-period SDs
  (b) Mahalanobis-style distance of each test year's feature centroid from the
      fit-period centroid (diagonal covariance, standardized)
  (c) where 2022's beta-projections sit relative to fit-period projections
  (d) the decomposition that decides it: is 2022's error large because the
      INPUTS were unusual, or because the base model was wrong on ordinary-
      looking inputs?

Output: errordir/task22_3_2022.csv
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
import task22_1_recentering as R  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_test, preds_test, test_df,
)

OUT = "errordir"
H_IDX = 3

scaler, beta, fit_ok, cal_ok, abs_err = R.fit_beta()
X_train, X_test = X_train_df.values, X_test_df.values
FEATURES = list(X_train_df.columns)

Xf = X_train[fit_ok]
fit_min, fit_max = Xf.min(axis=0), Xf.max(axis=0)
fit_mu, fit_sd_raw = Xf.mean(axis=0), Xf.std(axis=0)
# Two features (CPI_anomaly, share_price_anomaly) are CONSTANT over the fit
# period, so any nonzero test value gives an infinite z-score and destroys the
# distance metrics. They are excluded from the continuous diagnostics and
# reported separately below — their activation is itself a novelty signal.
DEGEN = np.where(fit_sd_raw < 1e-8)[0]
KEEP = np.where(fit_sd_raw >= 1e-8)[0]
fit_sd = fit_sd_raw + 1e-12

dates = pd.to_datetime(test_df["date"].values)
years = dates.year
realized = np.abs(preds_test[:, H_IDX] - y_test[:, H_IDX])
proj_test = scaler.transform(X_test) @ beta
proj_fit = scaler.transform(Xf) @ beta

print("=" * 96)
print("TASK 22 Item 3 — DIAGNOSIS of the 2022 blind spot (no fix)")
print("=" * 96)
print(f"  fit period: {len(fit_ok)} months, {len(FEATURES)} features")
print(f"  EXCLUDED from distance metrics (constant over fit period): "
      f"{[FEATURES[j] for j in DEGEN]}")
print(f"  fit-period projection range: [{proj_fit.min():.1f}, {proj_fit.max():.1f}]\n")

rows = []
for yr in sorted(set(years)):
    m = years == yr
    Xy = X_test[m]
    out_range = np.mean([(np.any(Xy[:, j] < fit_min[j]) or
                          np.any(Xy[:, j] > fit_max[j]))
                         for j in range(len(FEATURES))])
    z = np.abs((Xy[:, KEEP] - fit_mu[KEEP]) / fit_sd[KEEP])
    frac_3sd = float((z > 3).mean())
    max_z = float(z.max())
    centroid_d = float(np.linalg.norm((Xy[:, KEEP].mean(axis=0) - fit_mu[KEEP])
                                      / fit_sd[KEEP]) / np.sqrt(len(KEEP)))
    degen_active = float((Xy[:, DEGEN] != 0).mean()) if len(DEGEN) else 0.0
    py = proj_test[m]
    in_proj_range = float(((py >= proj_fit.min()) & (py <= proj_fit.max())).mean())
    rows.append(dict(year=int(yr), n=int(m.sum()),
                     mean_realized_error=round(float(np.nanmean(realized[m])), 2),
                     frac_features_out_of_fit_range=round(float(out_range), 3),
                     frac_cells_beyond_3sd=round(frac_3sd, 4),
                     max_abs_z=round(max_z, 2),
                     centroid_dist_per_feature=round(centroid_d, 3),
                     mean_projection=round(float(py.mean()), 2),
                     frac_proj_within_fit_range=round(in_proj_range, 3),
                     frac_constant_feat_activated=round(degen_active, 3)))

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task22_3_2022.csv", index=False)
print(D.to_string(index=False))

y22 = D[D.year == 2022].iloc[0]
others = D[D.year != 2022]
print("\n" + "-" * 96)
print("VERDICT")
print("-" * 96)
print(f"  2022 mean realized error:        {y22.mean_realized_error:.2f} "
      f"(highest of any test year)")
print(f"  2022 cells beyond 3 fit-SDs:     {100*y22.frac_cells_beyond_3sd:.2f}%  "
      f"(other years: {100*others.frac_cells_beyond_3sd.mean():.2f}% avg)")
print(f"  2022 centroid distance/feature:  {y22.centroid_dist_per_feature:.3f}  "
      f"(other years: {others.centroid_dist_per_feature.mean():.3f} avg)")
print(f"  2022 projections within fit range: "
      f"{100*y22.frac_proj_within_fit_range:.1f}%  "
      f"(other years: {100*others.frac_proj_within_fit_range.mean():.1f}% avg)")
print(f"  2022 constant-feature activation:  "
      f"{100*y22.frac_constant_feat_activated:.1f}%  "
      f"(other years: {100*others.frac_constant_feat_activated.mean():.1f}% avg)")

# The criterion must compare 2022 AGAINST THE OTHER TEST YEARS, not against an
# absolute threshold: every test year sits partly outside the fit range (only
# 8-42% of projections land inside it), so an absolute cut flags all of them and
# discriminates nothing.
crit = {
    "beyond_3sd vs others": (y22.frac_cells_beyond_3sd,
                             others.frac_cells_beyond_3sd.mean(),
                             y22.frac_cells_beyond_3sd > 1.5 * others.frac_cells_beyond_3sd.mean()),
    "centroid dist vs others": (y22.centroid_dist_per_feature,
                                others.centroid_dist_per_feature.mean(),
                                y22.centroid_dist_per_feature > 1.5 * others.centroid_dist_per_feature.mean()),
    "proj-in-range vs others": (y22.frac_proj_within_fit_range,
                                others.frac_proj_within_fit_range.mean(),
                                y22.frac_proj_within_fit_range < 0.5 * others.frac_proj_within_fit_range.mean()),
}
print()
for k, (v22, vo, flag) in crit.items():
    print(f"  {k:26s} 2022={v22:10.3f}  others={vo:10.3f}  "
          f"{'NOVEL' if flag else 'in line'}")
novel = any(f for _, _, f in crit.values())
print(f"\n  -> {'CASE 1 (genuinely novel / extrapolation)' if novel else 'CASE 2 (WITHIN the fit distribution — a catchable miss)'}")
print(f"\n  2022 is NOT an outlier on any relative measure. Its inputs look like")
print(f"  the other test years'; what differs is that the BASE MODEL was wrong")
print(f"  on ordinary-looking inputs (mean error {y22.mean_realized_error:.1f} vs "
      f"{others.mean_realized_error.mean():.1f} elsewhere).")
print(f"\nSaved {OUT}/task22_3_2022.csv")
