"""
TASK 11b — Robustness of EVT pool augmentation.

Task 11 found augmentation improves on the plain baseline in 7 of 8 cells at
about a third of the selector's width cost, but rested on (a) one GPD threshold,
(b) one random seed. Both are load-bearing: the synthetic draws are stochastic,
and Task 11's step-1 diagnostics already showed the fitted shape parameter is
unstable across thresholds in the recession domain.

This runs both studies and reports the answer either way.

STUDY 1 — threshold stability
    Re-run augmentation at GPD thresholds q in {0.70, 0.75, 0.80, 0.85, 0.90}.
    A method whose coverage swings wildly with an arbitrary modelling choice is
    not deployable. Reports the spread of coverage across thresholds per cell.

STUDY 2 — seed variance
    Re-run at 20 seeds with the threshold fixed at 0.80. Reports the mean, sd
    and full range of coverage, so the Task 11 single-seed numbers can be placed
    within their sampling distribution.

Both studies also report the plain baseline and the selector for reference, and
the fraction of (threshold, seed) configurations in which augmentation still
improves on the baseline — the honest version of "does this work".

Outputs:
  fix-reg/task11b_threshold_stability.csv
  fix-reg/task11b_seed_variance.csv
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

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, wilson_ci_from_indicator, GAMMA_DEFAULT,
)
import selector_lib as SEL  # noqa: E402
import augment_lib as AUG  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]
SEEDS = list(range(20))
SYNTH_FRAC = 0.20

thr_rows, seed_rows = [], []


def cov_of(cal_scores, y_te, p_te):
    s = np.asarray(cal_scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return np.nan, np.nan
    covered, _, widths = run_aci(y_te, p_te, s, gamma=GAMMA_DEFAULT)
    return coverage_and_width(covered, widths)


def run_cell(domain, model, horizon, scores_pool, y_te, p_te):
    s_all = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s_all))[0]
    if len(valid) < 60:
        return
    trailing = s_all[valid[-N_FIX:]] if len(valid) >= N_FIX else s_all[valid]
    selected = s_all[SEL.support_width_selector(s_all, N_FIX)]

    base_cov, base_w = cov_of(trailing, y_te, p_te)
    sel_cov, sel_w = cov_of(selected, y_te, p_te)

    # ── Study 1: threshold stability (seed fixed at Task 11's value) ────────
    covs = []
    for q in THRESHOLDS:
        aug, info = AUG.augment_pool(trailing, synth_frac=SYNTH_FRAC,
                                     threshold_q=q, seed=3)
        c, w = cov_of(aug, y_te, p_te)
        covs.append(c)
        thr_rows.append(dict(domain=domain, model=model, horizon=horizon,
                             threshold_q=q, applied=bool(info.get("applied")),
                             xi=info.get("xi"),
                             coverage=round(c, 2), width=round(w, 3),
                             base_cov=round(base_cov, 2),
                             base_w=round(base_w, 3),
                             sel_cov=round(sel_cov, 2), sel_w=round(sel_w, 3),
                             beats_base=bool(c > base_cov),
                             reason=info.get("reason", "")))
    cv = np.array([c for c in covs if np.isfinite(c)])
    spread = float(cv.max() - cv.min()) if len(cv) else np.nan
    n_beat = int((cv > base_cov).sum())
    print(f"  {domain:11s} {model:14s} {horizon:12s} "
          f"base={base_cov:6.2f}  thresholds -> "
          f"[{cv.min():6.2f},{cv.max():6.2f}] spread={spread:5.2f}  "
          f"beats base in {n_beat}/{len(cv)}")

    # ── Study 2: seed variance (threshold fixed at 0.80) ───────────────────
    scov, swid = [], []
    for sd in SEEDS:
        aug, info = AUG.augment_pool(trailing, synth_frac=SYNTH_FRAC,
                                     threshold_q=0.80, seed=sd)
        c, w = cov_of(aug, y_te, p_te)
        scov.append(c); swid.append(w)
    scov = np.array(scov); swid = np.array(swid)
    seed_rows.append(dict(domain=domain, model=model, horizon=horizon,
                          n_seeds=len(SEEDS),
                          cov_mean=round(float(scov.mean()), 2),
                          cov_sd=round(float(scov.std()), 3),
                          cov_min=round(float(scov.min()), 2),
                          cov_max=round(float(scov.max()), 2),
                          width_mean=round(float(swid.mean()), 3),
                          width_sd=round(float(swid.std()), 3),
                          base_cov=round(base_cov, 2), base_w=round(base_w, 3),
                          sel_cov=round(sel_cov, 2), sel_w=round(sel_w, 3),
                          frac_seeds_beating_base=round(
                              float((scov > base_cov).mean()), 3)))


print("=" * 100)
print("TASK 11b — robustness of EVT pool augmentation")
print("=" * 100)
print(f"  Study 1: GPD thresholds {THRESHOLDS} (seed fixed)")
print(f"  Study 2: {len(SEEDS)} seeds (threshold fixed at 0.80)")
print()

import task_oof_and_probit as T  # noqa: E402
print("Recession:")
for h_idx, h in enumerate(["Current", "1M", "3M", "6M"]):
    run_cell("Recession", "stacking-chain", h,
             np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx]),
             T.y_test[:, h_idx], T.preds_test[:, h_idx])

import runpy  # noqa: E402
print("\nHealthcare:")
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    run_cell("Healthcare", name, "30-day", g["SCORES"][name],
             g["Y_TEST"], g["PREDS_TEST"][name])

print("\nClimate:")
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    run_cell("Climate", name, "region-month", gc["SCORES"][name],
             gc["Y_TEST"], gc["PREDS_TEST"][name])

thr = pd.DataFrame(thr_rows)
sd = pd.DataFrame(seed_rows)
thr.to_csv(f"{OUT}/task11b_threshold_stability.csv", index=False)
sd.to_csv(f"{OUT}/task11b_seed_variance.csv", index=False)

print("\n" + "=" * 100)
print("STUDY 1 — THRESHOLD STABILITY")
print("=" * 100)
summ = (thr.groupby(["domain", "model", "horizon"])
        .agg(base_cov=("base_cov", "first"),
             cov_min=("coverage", "min"), cov_max=("coverage", "max"),
             cov_spread=("coverage", lambda x: round(x.max() - x.min(), 2)),
             n_beat=("beats_base", "sum"), n_cfg=("beats_base", "size"))
        .reset_index())
print(summ.to_string(index=False))
print(f"\n  Median coverage spread across thresholds: "
      f"{summ['cov_spread'].median():.2f}pp")
print(f"  Cells beating baseline at ALL thresholds: "
      f"{int((summ.n_beat == summ.n_cfg).sum())} of {len(summ)}")
print(f"  Cells beating baseline at NO threshold:   "
      f"{int((summ.n_beat == 0).sum())} of {len(summ)}")

print("\n" + "=" * 100)
print("STUDY 2 — SEED VARIANCE")
print("=" * 100)
print(sd[["domain", "model", "horizon", "base_cov", "cov_mean", "cov_sd",
          "cov_min", "cov_max", "frac_seeds_beating_base"]].to_string(index=False))
print(f"\n  Median coverage sd across seeds: {sd['cov_sd'].median():.3f}pp")
print(f"  Cells beating baseline in >=95% of seeds: "
      f"{int((sd.frac_seeds_beating_base >= 0.95).sum())} of {len(sd)}")
print(f"  Cells beating baseline in <50% of seeds:  "
      f"{int((sd.frac_seeds_beating_base < 0.50).sum())} of {len(sd)}")
print(f"\nSaved {OUT}/task11b_threshold_stability.csv, task11b_seed_variance.csv")
