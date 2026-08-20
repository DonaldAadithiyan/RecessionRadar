"""
TASK 20, Item 2 — Signal B: recent realized-error drift.

Tracks the last K RESOLVED nonconformity scores as they accumulate during the
test period, entirely separate from whatever calibration set is active, and
compares that recent window's own spread against the calibration set's current
Q_C(q_t).

    drift_ratio_t = spread(last K resolved scores, strictly before t) / Q_C(q_t)

    ratio << 1 : recent errors are small relative to what the calibration set
                 is provisioned for — comfortable.
    ratio -> 1 : recent errors are already brushing the calibration set's own
                 (1-alpha)-quantile — the set may be falling behind the current
                 volatility regime.
    ratio > 1  : recent reality has already exceeded it.

Two spread definitions are reported (both named in the spec: "its own max, or
its own p90"). BOTH are computed and reported; neither is selected by looking at
Item 3's outcome.

*** K IS FIXED IN ADVANCE: K = 6. ***
Rationale, stated before any validation run: the primary testbed is monthly
macroeconomic data and the headline horizon is 6M, so K=6 is exactly one
forecast horizon of resolved history — the shortest window that covers a full
horizon's worth of realized error without reaching back into a prior regime.
This is the value the spec itself suggests ("start with K=6, one prior year of
monthly data"). It is NOT tuned: no other K is evaluated against Item 3's
outcome anywhere in this task. A K-sensitivity sweep would be a separate,
clearly-labelled robustness study, not part of this validation.

NO-LEAKAGE STATEMENT: at step t the window uses resolved scores from steps
strictly < t only. The score at step t is excluded, because at prediction time
it has not happened yet. Enforced by construction in recent_window() and
re-verified independently in Item 3.

Outputs: none directly — this module is imported by Item 3.
"""
import numpy as np

# Fixed in advance. See rationale in the module docstring.
K_DEFAULT = 6


def recent_window(resolved_scores, t, K=K_DEFAULT):
    """
    The last K resolved nonconformity scores STRICTLY BEFORE index t.

    resolved_scores[i] is the realized |error| at test step i. At prediction
    time t, steps 0..t-1 have resolved and step t has not. Returns None until
    at least K scores have accumulated, so the signal is undefined (not
    guessed) during warm-up.
    """
    if t <= 0:
        return None
    hist = np.asarray(resolved_scores[:t], dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < K:
        return None
    return hist[-K:]


def drift_ratio(resolved_scores, t, q_c, K=K_DEFAULT, stat="max"):
    """
    spread(recent window) / Q_C(q_t).

    stat="max" : the window's own max          (spec's first suggestion)
    stat="p90" : the window's own 90th pctile  (spec's second suggestion)
    """
    w = recent_window(resolved_scores, t, K=K)
    if w is None or not np.isfinite(q_c) or q_c <= 0:
        return None
    spread = float(np.max(w)) if stat == "max" else float(np.percentile(w, 90))
    return spread / float(q_c)
