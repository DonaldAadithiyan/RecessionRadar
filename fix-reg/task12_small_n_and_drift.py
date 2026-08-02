"""
TASK 12 — Two boundary tests on the selector's optimality claim.

STUDY A (#1) — Does the 100%-of-ceiling result survive at small N?
    Task 10 proved the support-width selector attains exactly the maximum
    achievable quantile reach for a fixed pool, but explicitly hedged: "the
    negative result is specific to this N/pool regime... at much smaller N,
    selection could matter again. Not tested." That hedge sits directly under an
    otherwise unqualified optimality claim, so it is closed here.

    Method: sweep N from 254 down to 20. At each N compare the selector's
    quantile at the operative levels against the ceiling — the same quantile of
    the N LARGEST pool scores, which no N-subset can exceed. Report the ratio.
    A ratio < 1 anywhere marks the boundary of the theorem.

    Also reported: whether ACI coverage itself degrades with N, so the practical
    consequence of any theoretical gap is visible rather than inferred.

STUDY B (#5) — Does re-selecting the calibration set as alpha drifts help?
    Task 10 called this "much less promising" given the narrow measured alpha
    window ([0.073, 0.132]) but left it untested. Untested is not falsified, so
    it is run here rather than hand-waved.

    Method: an ACI variant that, at each step, re-selects its calibration subset
    to target the CURRENT alpha_t rather than using one fixed set throughout.
    Compared against the static selector on identical streams.

Both studies report whichever answer the data gives.

Outputs:
  fix-reg/task12a_small_n_ceiling.csv
  fix-reg/task12b_alpha_drift.csv
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

OUT = "fix-reg"
N_GRID = [20, 30, 50, 80, 120, 180, 254]
OPERATIVE_QS = [0.868, 0.900, 0.927]   # measured ACI window, Task 10
ALPHA_TARGET = 0.10

a_rows, b_rows = [], []


# ── STUDY A ─────────────────────────────────────────────────────────────────

def ceiling_ratio(pool, N, q):
    """selector's q-quantile / the max achievable q-quantile for any N-subset."""
    p = np.asarray(pool, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) < N + 5:
        return None
    order = np.argsort(p)
    top = p[order[-N:]]                      # the ceiling-attaining subset
    sel = p[SEL.support_width_selector(p, N)]
    c_top = float(np.quantile(top, q))
    c_sel = float(np.quantile(sel, q))
    return c_sel, c_top, (c_sel / c_top if c_top > 0 else np.nan)


def study_a(domain, model, horizon, pool, y_te, p_te):
    p = np.asarray(pool, dtype=float)
    p = p[np.isfinite(p)]
    for N in N_GRID:
        if len(p) < N + 5:
            continue
        ratios = {}
        for q in OPERATIVE_QS:
            r = ceiling_ratio(p, N, q)
            if r is None:
                continue
            ratios[q] = r[2]
        if not ratios:
            continue
        sel_idx = SEL.support_width_selector(p, N)
        covered, _, widths = run_aci(y_te, p_te, p[sel_idx],
                                     gamma=GAMMA_DEFAULT)
        cov, w = coverage_and_width(covered, widths)
        a_rows.append(dict(
            domain=domain, model=model, horizon=horizon, N=N,
            ratio_q868=round(ratios.get(0.868, np.nan), 6),
            ratio_q900=round(ratios.get(0.900, np.nan), 6),
            ratio_q927=round(ratios.get(0.927, np.nan), 6),
            min_ratio=round(float(np.nanmin(list(ratios.values()))), 6),
            coverage=round(cov, 2), width=round(w, 3)))
    sub = [r for r in a_rows if r["domain"] == domain and r["model"] == model
           and r["horizon"] == horizon]
    if sub:
        line = "  ".join(f"N={r['N']}:{r['min_ratio']:.4f}" for r in sub)
        print(f"  {domain:11s} {model:14s} {horizon:12s} min-ratio  {line}")


# ── STUDY B ─────────────────────────────────────────────────────────────────

def run_aci_reselect(y_te, p_te, pool, N, gamma=GAMMA_DEFAULT,
                     alpha_init=ALPHA_TARGET, alpha_target=ALPHA_TARGET):
    """
    ACI that re-selects its calibration subset at each step to target the
    CURRENT alpha_t, instead of holding one subset fixed for the whole stream.

    The subset is chosen by the quantile-targeted rule aimed at a narrow window
    around alpha_t. If re-selection helps, it should show up as better coverage
    or narrower intervals than the static selector on the same stream.
    """
    p = np.asarray(pool, dtype=float)
    p = p[np.isfinite(p)]
    alpha_t = alpha_init
    covered, widths = [], []
    cache = {}

    for t in range(len(y_te)):
        key = round(alpha_t, 3)          # cache: re-selection is deterministic
        if key not in cache:
            lo = max(0.01, alpha_t * 0.8)
            hi = min(0.99, alpha_t * 1.2)
            idx = SEL.quantile_targeted_selector(p, N, alpha_lo=lo, alpha_hi=hi)
            cache[key] = p[idx]
        cal = cache[key]

        q = np.quantile(cal, np.clip(1 - alpha_t, 0.0, 1.0))
        lo_b, hi_b = p_te[t] - q, p_te[t] + q
        widths.append(2 * q)

        yt = y_te[t]
        if np.isnan(yt):
            covered.append(np.nan)
            continue
        miss = 1 if (yt < lo_b or yt > hi_b) else 0
        covered.append(1 - miss)
        alpha_t = float(np.clip(alpha_t + gamma * (alpha_target - miss),
                                0.01, 0.99))
    return np.array(covered), np.array(widths)


def study_b(domain, model, horizon, pool, y_te, p_te, N=254):
    p = np.asarray(pool, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) < N + 5:
        return
    static = p[SEL.support_width_selector(p, N)]
    cs, _, ws = run_aci(y_te, p_te, static, gamma=GAMMA_DEFAULT)
    cov_s, w_s = coverage_and_width(cs, ws)
    lo_s, hi_s = wilson_ci_from_indicator(cs)

    cd, wd = run_aci_reselect(y_te, p_te, p, N)
    cov_d, w_d = coverage_and_width(cd, wd)
    lo_d, hi_d = wilson_ci_from_indicator(cd)

    b_rows.append(dict(domain=domain, model=model, horizon=horizon,
                       static_cov=round(cov_s, 2), static_w=round(w_s, 3),
                       static_lo=round(lo_s, 2), static_hi=round(hi_s, 2),
                       reselect_cov=round(cov_d, 2), reselect_w=round(w_d, 3),
                       reselect_lo=round(lo_d, 2), reselect_hi=round(hi_d, 2),
                       d_cov=round(cov_d - cov_s, 2),
                       width_ratio=round(w_d / w_s, 3) if w_s else np.nan,
                       helps=bool(cov_d > cov_s and w_d <= w_s)))
    r = b_rows[-1]
    print(f"  {domain:11s} {model:14s} {horizon:12s} "
          f"static={r['static_cov']:6.2f}% (w={r['static_w']:8.2f})  "
          f"re-select={r['reselect_cov']:6.2f}% (w={r['reselect_w']:8.2f})  "
          f"d={r['d_cov']:+6.2f}pp")


# ── Run both studies over every cell ────────────────────────────────────────
print("=" * 100)
print("TASK 12 — small-N optimality boundary (A) and alpha-drift re-selection (B)")
print("=" * 100)

CELLS = []
import task_oof_and_probit as T  # noqa: E402
for h_idx, h in enumerate(["Current", "1M", "3M", "6M"]):
    CELLS.append(("Recession", "stacking-chain", h,
                  np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx]),
                  T.y_test[:, h_idx], T.preds_test[:, h_idx]))

import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    CELLS.append(("Healthcare", name, "30-day", g["SCORES"][name],
                  g["Y_TEST"], g["PREDS_TEST"][name]))

gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    CELLS.append(("Climate", name, "region-month", gc["SCORES"][name],
                  gc["Y_TEST"], gc["PREDS_TEST"][name]))

print("\n" + "-" * 100)
print("STUDY A — ceiling ratio as N shrinks (1.0000 = selector is optimal)")
print("-" * 100)
for dom, mod, hor, pool, y_te, p_te in CELLS:
    study_a(dom, mod, hor, pool, y_te, p_te)

print("\n" + "-" * 100)
print("STUDY B — alpha-drift re-selection vs the static selector (N=254)")
print("-" * 100)
for dom, mod, hor, pool, y_te, p_te in CELLS:
    study_b(dom, mod, hor, pool, y_te, p_te)

A = pd.DataFrame(a_rows)
B = pd.DataFrame(b_rows)
A.to_csv(f"{OUT}/task12a_small_n_ceiling.csv", index=False)
B.to_csv(f"{OUT}/task12b_alpha_drift.csv", index=False)

print("\n" + "=" * 100)
print("STUDY A VERDICT — does optimality survive small N?")
print("=" * 100)
piv = A.pivot_table(index=["domain", "model", "horizon"], columns="N",
                    values="min_ratio")
print(piv.round(4).to_string())
worst = A["min_ratio"].min()
n_sub1 = int((A["min_ratio"] < 0.9999).sum())
print(f"\n  Worst ratio anywhere: {worst:.6f}")
print(f"  Configurations below 1.0: {n_sub1} of {len(A)}")
if n_sub1 == 0:
    print("  -> Optimality HOLDS at every N tested. The Task 10 hedge can be")
    print("     replaced with a tested claim.")
else:
    bad = A[A["min_ratio"] < 0.9999]
    print("  -> Optimality BREAKS below some N. Boundary cases:")
    print(bad[["domain", "model", "horizon", "N", "min_ratio",
               "coverage"]].to_string(index=False))

print("\n  Coverage as N shrinks (practical consequence):")
cp = A.pivot_table(index=["domain", "model", "horizon"], columns="N",
                   values="coverage")
print(cp.round(2).to_string())

print("\n" + "=" * 100)
print("STUDY B VERDICT — does re-selection on alpha drift help?")
print("=" * 100)
print(B[["domain", "model", "horizon", "static_cov", "reselect_cov", "d_cov",
         "width_ratio", "helps"]].to_string(index=False))
print(f"\n  Cells where re-selection helps (better cov, no wider): "
      f"{int(B['helps'].sum())} of {len(B)}")
print(f"  Mean coverage change: {B['d_cov'].mean():+.2f}pp "
      f"(min {B['d_cov'].min():+.2f}, max {B['d_cov'].max():+.2f})")
print(f"  Median width ratio: {B['width_ratio'].median():.3f}x")
print(f"\nSaved {OUT}/task12a_small_n_ceiling.csv, task12b_alpha_drift.csv")
