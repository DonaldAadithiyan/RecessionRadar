"""
TASK 18, Item 6 — parameter-free online conformal baseline (UP-OCP style).

The lowest-priority item in the spec, attempted because Items 1-5 and 7-9 are
complete. Same protocol as Item 1: primary recession testbed, all four horizons,
out-of-fold.

WHAT "PARAMETER-FREE" MEANS HERE. Standard ACI needs a step size gamma chosen in
advance; DtACI removes that by aggregating experts but introduces eta and sigma
in its place. Parameter-free online conformal methods (in the spirit of
UP-OCP / parameter-free online learning) instead use a step size that adapts
from the observed gradient history, so nothing needs setting by the user.

The update implemented here is the standard parameter-free online-gradient form:

    alpha_{t+1} = alpha_t + eta_t * (alpha_target - err_t)
    eta_t = D / sqrt(1 + sum_{s<=t} g_s^2)          (AdaGrad-style scaling)

with g_s the observed gradient (alpha_target - err_s) and D the diameter of the
alpha domain (D = 1, since alpha is a probability). There is nothing to tune:
eta_t is determined entirely by the data seen so far.

Reported honestly against the same reference points as every other baseline.

Outputs:
  fix-reg/task18_6_upocp.csv
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
    ALPHA_TARGET,
)
import selector_lib as SEL  # noqa: E402

OUT = "fix-reg"
N_FIX = 254
LABELS = ["Current", "1M", "3M", "6M"]

import task_oof_and_probit as T  # noqa: E402


def run_upocp(y_te, p_te, cal, at=ALPHA_TARGET, D=1.0):
    """
    Parameter-free online conformal: AdaGrad-scaled step, nothing to tune.
    """
    a = at
    grad_sq_sum = 0.0
    covered, widths, etas = [], [], []
    for t in range(len(y_te)):
        q = np.quantile(cal, np.clip(1 - a, 0, 1))
        lo, hi = p_te[t] - q, p_te[t] + q
        widths.append(2 * q)
        yv = y_te[t]
        if np.isnan(yv):
            covered.append(np.nan); continue
        miss = 1 if (yv < lo or yv > hi) else 0
        covered.append(1 - miss)

        g = at - miss                     # observed gradient
        grad_sq_sum += g * g
        eta = D / np.sqrt(1.0 + grad_sq_sum)   # adapts from history alone
        etas.append(eta)
        a = float(np.clip(a + eta * g, 0.01, 0.99))
    return np.array(covered), np.array(widths), etas


print("=" * 96)
print("TASK 18 Item 6 — parameter-free online conformal (UP-OCP style)")
print("=" * 96)
print("  step size eta_t = D / sqrt(1 + sum g_s^2): determined by the data,")
print("  not set by the user. Compared against the same reference points.")

rows = []
for h_idx, h in enumerate(LABELS):
    s = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    valid = np.where(np.isfinite(s))[0]
    y_te, p_te = T.y_test[:, h_idx], T.preds_test[:, h_idx]
    trailing = s[valid[-N_FIX:]]
    sel = s[SEL.support_width_selector(s, N_FIX)]
    sel = sel[np.isfinite(sel)]

    cb, _, wb = run_aci(y_te, p_te, trailing, gamma=GAMMA_DEFAULT)
    cov_b, w_b = coverage_and_width(cb, wb)
    cs, _, ws = run_aci(y_te, p_te, sel, gamma=GAMMA_DEFAULT)
    cov_s, w_s = coverage_and_width(cs, ws)

    cu, wu, etas = run_upocp(y_te, p_te, trailing)
    cov_u, w_u = coverage_and_width(cu, wu)
    lo, hi = wilson_ci_from_indicator(cu)

    rows.append(dict(horizon=h,
                     trailing_cov=round(cov_b, 2), trailing_w=round(w_b, 2),
                     upocp_cov=round(cov_u, 2), upocp_w=round(w_u, 2),
                     upocp_lo=round(lo, 2), upocp_hi=round(hi, 2),
                     selector_cov=round(cov_s, 2), selector_w=round(w_s, 2),
                     eta_first=round(float(etas[0]), 4) if etas else None,
                     eta_last=round(float(etas[-1]), 4) if etas else None,
                     beats_trailing=bool(cov_u > cov_b),
                     closes_gap_frac=round((cov_u - cov_b) / (cov_s - cov_b), 3)
                     if abs(cov_s - cov_b) > 1e-9 else None))
    r = rows[-1]
    print(f"  {h:8s} trailing={r['trailing_cov']:6.2f}  "
          f"UP-OCP={r['upocp_cov']:6.2f} (w={r['upocp_w']:7.2f})  "
          f"selector={r['selector_cov']:6.2f}  "
          f"eta {r['eta_first']:.3f}->{r['eta_last']:.3f}  "
          f"closes {r['closes_gap_frac']}")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/task18_6_upocp.csv", index=False)

print("\n" + "=" * 96)
print("VERDICT")
print("=" * 96)
print(f"  beats the trailing baseline at "
      f"{int(R['beats_trailing'].sum())} of {len(R)} horizons")
gaps = R["closes_gap_frac"].dropna()
if len(gaps):
    print(f"  median fraction of the selector's gain closed: {gaps.median():.3f}")
print(f"\nSaved {OUT}/task18_6_upocp.csv")
