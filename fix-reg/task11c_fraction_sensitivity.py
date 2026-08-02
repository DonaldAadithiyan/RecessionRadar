"""
TASK 11c — Synthetic-fraction sensitivity for the trailing-set augmentation.

Task 11b resolved the threshold and seed questions. The remaining untested knob
is how many synthetic scores to add. Task 11 varied this only for the
selected-set arm (which does not work anyway); the arm that DOES work — trailing
set + synthetic tail — was run at synth_frac = 0.20 only.

This sweeps synth_frac over a wide range and asks three things:
  1. Is 0.20 near an optimum, or was it an arbitrary lucky choice?
  2. Is there a monotone coverage/width tradeoff, so a practitioner can dial it?
  3. Does the method degrade gracefully at extreme fractions, or break?

Seeds are averaged (5 per configuration) so the curve is not a single-draw
artifact — Task 11b established seed SD is small but non-zero.

Output: fix-reg/task11c_fraction_sensitivity.csv
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

from domain_common import run_aci, coverage_and_width, GAMMA_DEFAULT  # noqa: E402
import selector_lib as SEL  # noqa: E402
import augment_lib as AUG  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
FRACS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
SEEDS = [0, 1, 2, 3, 4]
THRESHOLD_Q = 0.80

rows = []


def cov_of(cal, y_te, p_te):
    s = np.asarray(cal, dtype=float)
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

    line = []
    for f in FRACS:
        cs, ws = [], []
        for sd in SEEDS:
            aug, info = AUG.augment_pool(trailing, synth_frac=f,
                                         threshold_q=THRESHOLD_Q, seed=sd)
            c, w = cov_of(aug, y_te, p_te)
            cs.append(c); ws.append(w)
        cs, ws = np.array(cs), np.array(ws)
        rows.append(dict(domain=domain, model=model, horizon=horizon,
                         synth_frac=f, n_seeds=len(SEEDS),
                         cov_mean=round(float(cs.mean()), 2),
                         cov_sd=round(float(cs.std()), 3),
                         width_mean=round(float(ws.mean()), 3),
                         base_cov=round(base_cov, 2), base_w=round(base_w, 3),
                         sel_cov=round(sel_cov, 2), sel_w=round(sel_w, 3),
                         width_vs_base=round(float(ws.mean()) / base_w, 3),
                         beats_base=bool(cs.mean() > base_cov)))
        line.append(f"{f:.2f}:{cs.mean():5.2f}")
    print(f"  {domain:11s} {model:14s} {horizon:12s} base={base_cov:6.2f} | "
          + "  ".join(line))


print("=" * 100)
print("TASK 11c — synthetic-fraction sensitivity (trailing-set augmentation)")
print("=" * 100)
print(f"  fractions {FRACS}, averaged over {len(SEEDS)} seeds, "
      f"threshold q={THRESHOLD_Q}")
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

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/task11c_fraction_sensitivity.csv", index=False)

print("\n" + "=" * 100)
print("COVERAGE vs SYNTHETIC FRACTION")
print("=" * 100)
piv = res.pivot_table(index=["domain", "model", "horizon"],
                      columns="synth_frac", values="cov_mean")
print(piv.round(2).to_string())

print("\n" + "=" * 100)
print("WIDTH (multiple of baseline) vs SYNTHETIC FRACTION")
print("=" * 100)
pw = res.pivot_table(index=["domain", "model", "horizon"],
                     columns="synth_frac", values="width_vs_base")
print(pw.round(2).to_string())

print("\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
# Is 0.20 near-optimal, and is the tradeoff monotone?
best = (res.loc[res.groupby(["domain", "model", "horizon"])["cov_mean"].idxmax()]
        [["domain", "model", "horizon", "synth_frac", "cov_mean", "base_cov"]])
print("\n  Coverage-maximising fraction per cell:")
print(best.to_string(index=False))

f020 = res[res.synth_frac == 0.20].set_index(["domain", "model", "horizon"])
bidx = best.set_index(["domain", "model", "horizon"])
gap = (bidx["cov_mean"] - f020["cov_mean"]).dropna()
print(f"\n  Coverage left on the table by using 0.20 instead of the per-cell "
      f"best: mean {gap.mean():.2f}pp, max {gap.max():.2f}pp")

mono = []
for k, gp in res.groupby(["domain", "model", "horizon"]):
    gp = gp.sort_values("synth_frac")
    mono.append(bool(np.all(np.diff(gp["width_vs_base"].values) >= -1e-9)))
print(f"  Width increases monotonically with fraction in "
      f"{sum(mono)} of {len(mono)} cells")

for f in FRACS:
    sub = res[res.synth_frac == f]
    print(f"  frac={f:.2f}: beats baseline in {int(sub.beats_base.sum())}/"
          f"{len(sub)} cells, mean width {sub.width_vs_base.mean():.2f}x base")
print(f"\nSaved {OUT}/task11c_fraction_sensitivity.csv")
