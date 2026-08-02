"""
TASK 10 (step 1) — What alpha range does ACI actually visit?

The quantile-targeted selector needs a weight function w(alpha) concentrated on
the alphas ACI genuinely operates at. That is an empirical question, not a
modelling choice, so it is measured here before the selector is built.

For each domain/horizon this records the ACI alpha trajectory under the
trailing-N calibration set and reports its distribution. The resulting range
defines the integration window for the new objective.

Output: fix-reg/task10_alpha_ranges.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import run_aci, GAMMA_DEFAULT  # noqa: E402

OUT = "fix-reg"
N_FIX = 254

rows = []


def record(domain, model, horizon, scores_pool, y_te, p_te):
    s = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s))[0]
    if len(valid) < 30:
        return
    trailing = s[valid[-N_FIX:]] if len(valid) >= N_FIX else s[valid]
    _, alpha_traj, _ = run_aci(y_te, p_te, trailing, gamma=GAMMA_DEFAULT)
    a = np.asarray(alpha_traj, dtype=float)
    rows.append(dict(domain=domain, model=model, horizon=horizon,
                     alpha_min=round(float(a.min()), 4),
                     alpha_p05=round(float(np.percentile(a, 5)), 4),
                     alpha_median=round(float(np.median(a)), 4),
                     alpha_p95=round(float(np.percentile(a, 95)), 4),
                     alpha_max=round(float(a.max()), 4),
                     alpha_mean=round(float(a.mean()), 4)))
    r = rows[-1]
    print(f"  {domain:12s} {model:14s} {horizon:12s} "
          f"alpha in [{r['alpha_min']:.4f}, {r['alpha_max']:.4f}]  "
          f"median={r['alpha_median']:.4f}")


print("=" * 78)
print("TASK 10 step 1 — empirical ACI alpha ranges")
print("=" * 78)

# ── Recession ───────────────────────────────────────────────────────────────
import task_oof_and_probit as T  # noqa: E402
LABELS = ["Current", "1M", "3M", "6M"]
print("\nRecession (out-of-fold scores):")
for h_idx, h in enumerate(LABELS):
    s = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    record("Recession", "stacking-chain", h, s,
           T.y_test[:, h_idx], T.preds_test[:, h_idx])

# ── Healthcare ──────────────────────────────────────────────────────────────
import runpy  # noqa: E402
print("\nHealthcare:")
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    record("Healthcare", name, "30-day", g["SCORES"][name],
           g["Y_TEST"], g["PREDS_TEST"][name])

# ── Climate ────────────────────────────────────────────────────────────────
print("\nClimate:")
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    record("Climate", name, "region-month", gc["SCORES"][name],
           gc["Y_TEST"], gc["PREDS_TEST"][name])

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/task10_alpha_ranges.csv", index=False)

print("\n" + "=" * 78)
print("SUMMARY — the alpha window the selector should target")
print("=" * 78)
print(df.to_string(index=False))
print(f"\n  Overall alpha range visited: "
      f"[{df['alpha_min'].min():.4f}, {df['alpha_max'].max():.4f}]")
print(f"  Central 90% of trajectories: "
      f"[{df['alpha_p05'].min():.4f}, {df['alpha_p95'].max():.4f}]")
print("\n  Interpretation: ACI does NOT sit at the nominal alpha=0.10. Where it")
print("  drifts materially below it, the operative calibration quantile is")
print("  deeper than 0.90 — which is exactly the region a p95-p5 support-width")
print("  objective fails to target.")
print(f"\nSaved {OUT}/task10_alpha_ranges.csv")
