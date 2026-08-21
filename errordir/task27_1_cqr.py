"""
TASK 27, Item 1 — Conformalized Quantile Regression (CQR) baseline.

Romano, Patterson & Candes (NeurIPS 2019). Standard reference method for this
problem; its absence weakened every prior comparison table in this project.

CONSTRUCTION (same fit/cal/test discipline as every other baseline here):
  1. On the FIT split, fit two quantile regressors at alpha/2 and 1-alpha/2.
  2. On the CALIBRATION split, form the conformity score
         E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
     which is negative when the initial interval already covers.
  3. Take Q = the ceil((n+1)(1-alpha))/n empirical quantile of E (the finite-
     sample-valid conformal correction) and widen:
         [q_lo(x) - Q,  q_hi(x) + Q]

CRITICAL SPLIT DISCIPLINE (Task 24 Finding 3): the quantile regressors are fit
on FIT only and conformalized on CAL only. CQR never sees data another method
was denied, and never sees CAL when fitting its quantile functions -- doing so
would flatter CQR the way the selector's full-pool calibration initially
flattered it in Task 24.

Exposed as cqr_intervals() for reuse by task27_3.
"""
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor

ALPHA = 0.10


def cqr_intervals(X_fit, y_fit, X_cal, y_cal, X_test, alpha=ALPHA,
                  kind="linear", seed=5):
    """
    Returns (lo, hi) for X_test. Quantile regressors fit on FIT only;
    conformal correction computed on CAL only.
    """
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    if kind == "linear":
        m_lo = QuantileRegressor(quantile=lo_q, alpha=1e-3, solver="highs")
        m_hi = QuantileRegressor(quantile=hi_q, alpha=1e-3, solver="highs")
    else:
        m_lo = GradientBoostingRegressor(loss="quantile", alpha=lo_q,
                                         n_estimators=200, max_depth=3,
                                         learning_rate=0.06, random_state=seed)
        m_hi = GradientBoostingRegressor(loss="quantile", alpha=hi_q,
                                         n_estimators=200, max_depth=3,
                                         learning_rate=0.06, random_state=seed)
    ok = np.isfinite(y_fit)
    m_lo.fit(X_fit[ok], y_fit[ok])
    m_hi.fit(X_fit[ok], y_fit[ok])

    # conformity scores on CAL
    c_lo, c_hi = m_lo.predict(X_cal), m_hi.predict(X_cal)
    okc = np.isfinite(y_cal)
    E = np.maximum(c_lo[okc] - y_cal[okc], y_cal[okc] - c_hi[okc])
    n = len(E)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    Q = float(np.sort(E)[k - 1]) if n > 0 else 0.0

    t_lo, t_hi = m_lo.predict(X_test), m_hi.predict(X_test)
    return t_lo - Q, t_hi + Q
