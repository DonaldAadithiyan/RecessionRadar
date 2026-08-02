"""
TASK 10 — Quantile-targeted calibration selection vs the support-width selector.

Hypothesis under test: the existing selector maximises p95-p5, but ACI only ever
consumes the (1-alpha_t) quantile. Measured alpha trajectories (step 1) show ACI
operates in alpha in [0.073, 0.132] across all domains — i.e. the operative
calibration quantile is always in [0.868, 0.927]. The entire lower half of the
support-width objective therefore cannot influence coverage, and the selection
budget spent there is pure width inflation.

Prediction: targeting quantile reach over the measured alpha window, subject to
a width budget, should hold coverage while cutting interval width.

This script reports the comparison honestly whether or not that prediction
holds. A negative result closes the idea; it does not get quietly reframed.

Arms compared, on identical scores/test streams:
  pooled_trailing          the reference (trailing N)
  support_width            the paper's existing selector
  quantile_targeted        NEW, unconstrained
  quantile_targeted_kX     NEW, with width budget kappa = X

Output: fix-reg/task10_selector_comparison.csv
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
import selector_lib as S  # noqa: E402

OUT = "fix-reg"
N_FIX = 254

# Measured empirically in task10_alpha_trajectories.py (all domains, all models).
ALPHA_LO, ALPHA_HI = 0.073, 0.132
KAPPAS = [0.75, 1.0, 1.5]      # width budgets, as a multiple of baseline width

rows = []


def evaluate(domain, model, horizon, arm, sel_scores, y_te, p_te, block,
             base_width=None, sel_desc=None):
    s = np.asarray(sel_scores, dtype=float)
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
             width_vs_base=round(w / base_width, 3) if base_width else 1.0)
    if sel_desc:
        r.update({f"sel_{k}": round(v, 4) for k, v in sel_desc.items()})
    rows.append(r)
    return r


def run_cell(domain, model, horizon, scores_pool, y_te, p_te, block):
    s_all = np.asarray(scores_pool, dtype=float)
    valid = np.where(np.isfinite(s_all))[0]
    if len(valid) < 40:
        print(f"  [skip] {domain}/{model}/{horizon}: too few scored points")
        return
    trailing = s_all[valid[-N_FIX:]] if len(valid) >= N_FIX else s_all[valid]

    base = evaluate(domain, model, horizon, "pooled_trailing", trailing,
                    y_te, p_te, block)
    if base is None:
        return
    bw = base["mean_width"]
    base["width_vs_base"] = 1.0

    # existing selector
    sel_sw = S.support_width_selector(s_all, N_FIX)
    evaluate(domain, model, horizon, "support_width", s_all[sel_sw],
             y_te, p_te, block, base_width=bw,
             sel_desc=S.describe_selection(s_all, sel_sw))

    # new selector, unconstrained
    sel_q = S.quantile_targeted_selector(s_all, N_FIX,
                                         alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI)
    evaluate(domain, model, horizon, "quantile_targeted", s_all[sel_q],
             y_te, p_te, block, base_width=bw,
             sel_desc=S.describe_selection(s_all, sel_q))

    # new selector, width-budgeted
    for k in KAPPAS:
        sel_k = S.quantile_targeted_selector(
            s_all, N_FIX, alpha_lo=ALPHA_LO, alpha_hi=ALPHA_HI,
            kappa=k, baseline_scores=trailing)
        evaluate(domain, model, horizon, f"quantile_targeted_k{k}",
                 s_all[sel_k], y_te, p_te, block, base_width=bw,
                 sel_desc=S.describe_selection(s_all, sel_k))

    cell = [r for r in rows if r["domain"] == domain and r["model"] == model
            and r["horizon"] == horizon]
    print(f"\n  {domain}/{model}/{horizon}:")
    for r in cell:
        print(f"    {r['arm']:26s} cov={r['coverage']:6.2f}% "
              f"[{r['wilson_lo']:5.2f},{r['wilson_hi']:5.2f}]  "
              f"w={r['mean_width']:9.3f}  ({r['width_vs_base']:.2f}x base)")


print("=" * 90)
print("TASK 10 — quantile-targeted vs support-width calibration selection")
print("=" * 90)
print(f"  Targeting the empirically measured ACI window: "
      f"alpha in [{ALPHA_LO}, {ALPHA_HI}]")
print(f"  i.e. operative calibration quantiles in "
      f"[{1-ALPHA_HI:.3f}, {1-ALPHA_LO:.3f}]")
print(f"  Width budgets tested: kappa = {KAPPAS}")

# ── Recession ───────────────────────────────────────────────────────────────
print("\n" + "-" * 90)
print("RECESSION (out-of-fold)")
print("-" * 90)
import task_oof_and_probit as T  # noqa: E402
for h_idx, h in enumerate(["Current", "1M", "3M", "6M"]):
    s = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    run_cell("Recession", "stacking-chain", h, s,
             T.y_test[:, h_idx], T.preds_test[:, h_idx], block=12)

# ── Healthcare ──────────────────────────────────────────────────────────────
print("\n" + "-" * 90)
print("HEALTHCARE")
print("-" * 90)
import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    run_cell("Healthcare", name, "30-day", g["SCORES"][name],
             g["Y_TEST"], g["PREDS_TEST"][name], block=1)

# ── Climate ────────────────────────────────────────────────────────────────
print("\n" + "-" * 90)
print("CLIMATE")
print("-" * 90)
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    run_cell("Climate", name, "region-month", gc["SCORES"][name],
             gc["Y_TEST"], gc["PREDS_TEST"][name], block=12)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/task10_selector_comparison.csv", index=False)

# ── Head-to-head: new vs existing selector ─────────────────────────────────
print("\n" + "=" * 90)
print("HEAD-TO-HEAD — quantile-targeted vs support-width")
print("=" * 90)
comp = []
for (d, m, h), grp in res.groupby(["domain", "model", "horizon"]):
    sw = grp[grp.arm == "support_width"]
    if sw.empty:
        continue
    sw = sw.iloc[0]
    for arm in ["quantile_targeted"] + [f"quantile_targeted_k{k}" for k in KAPPAS]:
        q = grp[grp.arm == arm]
        if q.empty:
            continue
        q = q.iloc[0]
        comp.append(dict(domain=d, model=m, horizon=h, arm=arm,
                         sw_cov=sw.coverage, q_cov=q.coverage,
                         d_cov=round(q.coverage - sw.coverage, 2),
                         sw_w=sw.mean_width, q_w=q.mean_width,
                         width_ratio=round(q.mean_width / sw.mean_width, 3),
                         # a strict improvement: no worse coverage, less width
                         strict_win=bool(q.coverage >= sw.coverage - 1e-9 and
                                         q.mean_width < sw.mean_width)))
c = pd.DataFrame(comp)
c.to_csv(f"{OUT}/task10_head_to_head.csv", index=False)
print(c.to_string(index=False))

print("\n" + "=" * 90)
print("VERDICT")
print("=" * 90)
for arm in ["quantile_targeted"] + [f"quantile_targeted_k{k}" for k in KAPPAS]:
    sub = c[c.arm == arm]
    if sub.empty:
        continue
    print(f"\n  {arm}:")
    print(f"    strict wins (>= coverage AND < width): "
          f"{int(sub.strict_win.sum())} of {len(sub)}")
    print(f"    median width vs support-width selector: "
          f"{sub.width_ratio.median():.3f}x")
    print(f"    mean coverage change: {sub.d_cov.mean():+.2f}pp "
          f"(min {sub.d_cov.min():+.2f}, max {sub.d_cov.max():+.2f})")
print(f"\nSaved {OUT}/task10_selector_comparison.csv, task10_head_to_head.csv")
