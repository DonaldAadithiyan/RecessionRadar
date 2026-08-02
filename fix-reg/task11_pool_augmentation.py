"""
TASK 11 — EVT pool augmentation vs the diversity-optimal selector.

Task 10 established that the selector is already at 100% of the achievable
quantile-reach ceiling for a FIXED pool. The only remaining lever is to change
what is in the pool. This tests that: fit a GPD to the score tail, draw
synthetic exceedances, add them to the calibration set, then run ORDINARY ACI.

Distinct from Phase 3b, which replaced the quantile estimator with a parametric
one and hurt (66% at 6M). Here the parametric model only adds scores above its
fitting threshold; the empirical body is untouched and ACI still takes empirical
quantiles.

Arms:
  pooled_trailing            reference
  support_width              existing selector
  augmented_trailing         trailing set + synthetic tail
  augmented_selected         selector's set + synthetic tail   <- the combination
  augmented_selected_f{X}    same, at synthetic fractions X

Step-1 diagnostics (task11_tail_diagnostics.py) found the tail is well-behaved
in climate and healthcare (xi 0.12-0.31, stable across thresholds) but NOT in
recession (xi-spread 1.4-2.8 across thresholds = unstable fit; 6M is bounded
with an endpoint only 1.18x the observed max). Recession cells are still run,
but their fit instability is carried into the output so the write-up can weight
them accordingly.

Output: fix-reg/task11_augmentation_comparison.csv
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
    run_aci, coverage_and_width, wilson_ci_from_indicator,
    block_bootstrap_ci, GAMMA_DEFAULT,
)
import selector_lib as SEL  # noqa: E402
import augment_lib as AUG  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
SYNTH_FRACS = [0.10, 0.20, 0.40]
SEED = 3

rows = []
diag_rows = []


def evaluate(domain, model, horizon, arm, cal_scores, y_te, p_te, block,
             base_width=None, info=None):
    s = np.asarray(cal_scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return None
    covered, _, widths = run_aci(y_te, p_te, s, gamma=GAMMA_DEFAULT)
    cov, w = coverage_and_width(covered, widths)
    wlo, whi = wilson_ci_from_indicator(covered)
    blo, bhi = block_bootstrap_ci(covered, block=block)
    r = dict(domain=domain, model=model, horizon=horizon, arm=arm,
             n=int(np.isfinite(np.asarray(covered, float)).sum()),
             coverage=round(cov, 2), wilson_lo=round(wlo, 2),
             wilson_hi=round(whi, 2), boot_lo=round(blo, 2),
             boot_hi=round(bhi, 2), mean_width=round(w, 3),
             width_vs_base=round(w / base_width, 3) if base_width else 1.0,
             cal_q90=round(float(np.quantile(s, 0.90)), 4),
             cal_n=int(len(s)))
    if info:
        r.update(aug_applied=info.get("applied", False),
                 aug_xi=info.get("xi"), aug_n_synth=info.get("n_synth", 0),
                 aug_reason=info.get("reason", ""))
    rows.append(r)
    return r


def run_cell(domain, model, horizon, scores_pool, y_te, p_te, block):
    s_all = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s_all))[0]
    if len(valid) < 60:
        print(f"  [skip] {domain}/{model}/{horizon}")
        return
    trailing = s_all[valid[-N_FIX:]] if len(valid) >= N_FIX else s_all[valid]
    sel_idx = SEL.support_width_selector(s_all, N_FIX)
    selected = s_all[sel_idx]

    base = evaluate(domain, model, horizon, "pooled_trailing", trailing,
                    y_te, p_te, block)
    if base is None:
        return
    bw = base["mean_width"]
    base["width_vs_base"] = 1.0

    evaluate(domain, model, horizon, "support_width", selected,
             y_te, p_te, block, base_width=bw)

    # augmenting the trailing set
    aug_t, info_t = AUG.augment_pool(trailing, synth_frac=0.20, seed=SEED)
    evaluate(domain, model, horizon, "augmented_trailing", aug_t,
             y_te, p_te, block, base_width=bw, info=info_t)

    # augmenting the SELECTED set at several synthetic fractions
    for f in SYNTH_FRACS:
        aug_s, info_s = AUG.augment_pool(selected, synth_frac=f, seed=SEED)
        evaluate(domain, model, horizon, f"augmented_selected_f{f}", aug_s,
                 y_te, p_te, block, base_width=bw, info=info_s)
        if f == 0.20:
            diag_rows.append(dict(domain=domain, model=model, horizon=horizon,
                                  **{k: v for k, v in info_s.items()
                                     if k != "reason"},
                                  reason=info_s.get("reason", "")))

    cell = [r for r in rows if r["domain"] == domain and r["model"] == model
            and r["horizon"] == horizon]
    print(f"\n  {domain}/{model}/{horizon}:")
    for r in cell:
        note = ""
        if r.get("aug_applied") is False and r["arm"].startswith("augmented"):
            note = f"   [NOT APPLIED: {r.get('aug_reason','')}]"
        print(f"    {r['arm']:28s} cov={r['coverage']:6.2f}% "
              f"[{r['wilson_lo']:5.2f},{r['wilson_hi']:5.2f}]  "
              f"w={r['mean_width']:9.3f} ({r['width_vs_base']:.2f}x) "
              f"q90={r['cal_q90']:8.3f}{note}")


print("=" * 96)
print("TASK 11 — EVT pool augmentation (extend the pool, don't replace the")
print("          quantile estimator — the Phase 3b distinction)")
print("=" * 96)

# ── Recession ───────────────────────────────────────────────────────────────
print("\n" + "-" * 96)
print("RECESSION  (NOTE: step-1 diagnostics found unstable GPD fits here,")
print("            xi-spread 1.4-2.8 across thresholds; 6M tail is BOUNDED)")
print("-" * 96)
import task_oof_and_probit as T  # noqa: E402
for h_idx, h in enumerate(["Current", "1M", "3M", "6M"]):
    s = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    run_cell("Recession", "stacking-chain", h, s,
             T.y_test[:, h_idx], T.preds_test[:, h_idx], block=12)

# ── Healthcare ──────────────────────────────────────────────────────────────
print("\n" + "-" * 96)
print("HEALTHCARE  (well-behaved tails: xi 0.00-0.12, stable)")
print("-" * 96)
import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    run_cell("Healthcare", name, "30-day", g["SCORES"][name],
             g["Y_TEST"], g["PREDS_TEST"][name], block=1)

# ── Climate ────────────────────────────────────────────────────────────────
print("\n" + "-" * 96)
print("CLIMATE  (well-behaved heavy tails: xi 0.29-0.31, very stable)")
print("-" * 96)
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    run_cell("Climate", name, "region-month", gc["SCORES"][name],
             gc["Y_TEST"], gc["PREDS_TEST"][name], block=12)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/task11_augmentation_comparison.csv", index=False)
if diag_rows:
    pd.DataFrame(diag_rows).to_csv(f"{OUT}/task11_augmentation_diagnostics.csv",
                                   index=False)

# ── Head-to-head vs the existing selector ──────────────────────────────────
print("\n" + "=" * 96)
print("HEAD-TO-HEAD — augmented arms vs the support-width selector")
print("=" * 96)
comp = []
for (d, m, h), grp in res.groupby(["domain", "model", "horizon"]):
    sw = grp[grp.arm == "support_width"]
    if sw.empty:
        continue
    sw = sw.iloc[0]
    for arm in grp.arm.unique():
        if not arm.startswith("augmented"):
            continue
        a = grp[grp.arm == arm].iloc[0]
        comp.append(dict(domain=d, model=m, horizon=h, arm=arm,
                         sw_cov=sw.coverage, aug_cov=a.coverage,
                         d_cov=round(a.coverage - sw.coverage, 2),
                         sw_w=sw.mean_width, aug_w=a.mean_width,
                         width_ratio=round(a.mean_width / sw.mean_width, 3),
                         sw_q90=sw.cal_q90, aug_q90=a.cal_q90,
                         applied=bool(a.get("aug_applied", False)),
                         # beat = higher coverage AND not wider
                         beats=bool(a.coverage > sw.coverage and
                                    a.mean_width <= sw.mean_width)))
c = pd.DataFrame(comp)
c.to_csv(f"{OUT}/task11_head_to_head.csv", index=False)
print(c.to_string(index=False))

print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
for arm in sorted(c.arm.unique()):
    sub = c[c.arm == arm]
    ap = sub[sub.applied]
    print(f"\n  {arm}:")
    print(f"    applied in {len(ap)} of {len(sub)} cells")
    if len(ap):
        print(f"    beats selector (higher cov, no wider): "
              f"{int(ap.beats.sum())} of {len(ap)}")
        print(f"    mean coverage change: {ap.d_cov.mean():+.2f}pp "
              f"(min {ap.d_cov.min():+.2f}, max {ap.d_cov.max():+.2f})")
        print(f"    median width ratio: {ap.width_ratio.median():.3f}x")
print(f"\nSaved {OUT}/task11_augmentation_comparison.csv")
