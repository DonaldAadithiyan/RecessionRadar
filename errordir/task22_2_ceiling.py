"""
TASK 22, Item 2 — Raise the bounded-compensation ceiling.

THE BUG (Task 21 Item 4): the Jan-Apr 2020 cluster got the CORRECT signal —
beta flagged them with high multipliers (1.21-1.24) — but the ceiling of 1.25
could not stretch against misses of that size (predicted 24-47 vs actual ~0).
The signal worked; the response range didn't.

THE NEW CEILING, DERIVED FROM FIT-PERIOD DATA ONLY
--------------------------------------------------
The multiplier scales the ACI interval half-width Q_C. For the scaling to be
able to cover an error of size E when the base interval is Q_C, we need

        HI * Q_C  >=  E          ->  HI >= E / Q_C

So the natural fit-period-derived ceiling is the ratio of the fit period's own
upper-tail error to the fit period's own typical interval half-width:

        HI_new = p99(fit-period |error|) / p90(fit-period |error|)

RATIONALE for this exact form, stated before computing it:
  - NUMERATOR p99: the ceiling must be able to reach the fit period's own
    near-worst errors. p99 rather than max, because max is a single point and
    not a stable statistic at n=291.
  - DENOMINATOR p90: ACI at alpha=0.10 operates at roughly the 90th percentile
    of the score distribution, so p90 is what Q_C is typically near. The ratio
    therefore answers "how many typical interval half-widths do I need to reach
    a near-worst fit-period error?" — dimensionless and scale-free.
  - Both quantities come from the 291-month FIT split ONLY. No CAL data, no
    test data, no knowledge of the Jan-Apr 2020 cluster's size.

LO is left at 0.75, unchanged. Only the ceiling was diagnosed as the bug;
the floor bounds the confidently-wrong-narrow risk and Task 21 Item 4 found
that risk correctly targeted (narrowed months had 10x lower realized error),
so there is no evidence-based reason to move it.

NOTE ON SYMMETRY: raising HI without moving LO breaks the old design's
"mean multiplier = 1.0 under uniform ranks" property BY CONSTRUCTION. That
property is reported explicitly in Item 4 rather than silently assumed, and
the width column of the comparison is where any resulting inflation shows up.

Outputs: errordir/task22_2_ceiling.csv
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
from task_oof_and_probit import y_train as y_pool, oof_pred as preds_pool_oof, n_pool  # noqa: E402

OUT = "errordir"
H_IDX = 3
FIT_FRAC = 0.60
LO = 0.75          # unchanged from Task 21


def derive_ceiling():
    """HI = p99(fit |error|) / p90(fit |error|). FIT SPLIT ONLY."""
    abs_err = np.abs(preds_pool_oof[:, H_IDX] - y_pool[:, H_IDX])
    n_fit = int(round(FIT_FRAC * n_pool))
    fit_idx = np.arange(0, n_fit)
    e = abs_err[fit_idx]
    e = e[np.isfinite(e)]
    p99 = float(np.percentile(e, 99))
    p90 = float(np.percentile(e, 90))
    hi = p99 / p90
    return hi, p99, p90, len(e)


HI_NEW, P99, P90, N_FIT_USED = derive_ceiling()


def multiplier(u, lo=LO, hi=None):
    hi = HI_NEW if hi is None else hi
    return lo + (hi - lo) * np.clip(np.asarray(u, dtype=float), 0.0, 1.0)


if __name__ == "__main__":
    print("=" * 96)
    print("TASK 22 Item 2 — new ceiling from FIT-PERIOD data only")
    print("=" * 96)
    print(f"  fit-period months used: {N_FIT_USED}")
    print(f"  p99(fit |error|) = {P99:.4f}")
    print(f"  p90(fit |error|) = {P90:.4f}")
    print(f"  HI_new = p99/p90 = {HI_NEW:.4f}   (old HI = 1.25)")
    print(f"  LO unchanged = {LO}")
    print(f"  band: [{LO}, {HI_NEW:.4f}]  vs old [0.75, 1.25]")

    # Observation only (NOT used to choose the ceiling): what would Jan-Apr 2020
    # have needed? Reported per the guardrail as an observation, after the fact.
    print("\n  OBSERVATION (not used to choose the ceiling):")
    print("    Jan-Apr 2020 realized errors were 24.27, 23.69, 46.98, 14.14.")
    print(f"    With a 6M ACI half-width around Q_C ~ 26.6 (Task 21 mean width")
    print(f"    53.28 / 2), covering a 46.98 error needs HI ~ {46.98/26.64:.2f}.")
    print(f"    The fit-derived ceiling is {HI_NEW:.2f} — "
          f"{'ENOUGH' if HI_NEW >= 46.98/26.64 else 'STILL NOT ENOUGH'} for the worst case.")
    print("    This is reported as an observation, not as validation: the")
    print("    ceiling was derived independently of these four months.")

    pd.DataFrame([dict(n_fit_months=N_FIT_USED, p99_fit_error=round(P99, 4),
                       p90_fit_error=round(P90, 4), HI_new=round(HI_NEW, 4),
                       HI_old=1.25, LO=LO,
                       worst_2020_error=46.98,
                       hi_needed_for_worst_2020=round(46.98 / 26.64, 4),
                       ceiling_sufficient_for_worst=bool(HI_NEW >= 46.98 / 26.64)
                       )]).to_csv(f"{OUT}/task22_2_ceiling.csv", index=False)
    print(f"\nSaved {OUT}/task22_2_ceiling.csv")
