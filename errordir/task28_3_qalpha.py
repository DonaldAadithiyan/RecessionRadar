"""
TASK 28, Item 3 — q_alpha(z): a learned conditional quantile function replacing
the fixed +/-25% multiplier.

MOTIVATION (Task 27): the fixed rank-based multiplier, not the fitting target,
is the bottleneck. beta_mean and beta_tail produced identical coverage gaps
(3.25pp on energy) because the multiplier is a monotone rank map -- only the
ORDERING matters, and both betas order test points nearly identically. So the
allocation function itself must change.

THE ESTIMATOR — stated in full, with every parameter fixed from FIT-PERIOD
reasoning BEFORE any test-period result is seen:

    q_alpha(z) = the (1-alpha) conditional quantile of the nonconformity score,
                 as a function of the difficulty coordinate z = beta'x.

    Estimator: ISOTONIC (monotone non-decreasing) regression of the indicator
    1{s_i > tau} on z_i, inverted to a quantile via a local-window quantile,
    then smoothed. Concretely:

      1. Sort calibration points by z.
      2. Over a centred sliding window of K neighbours in z-order, take the
         empirical (1-alpha) quantile of the scores in that window.
      3. Enforce monotonicity in z with isotonic regression (PAVA) on those
         window quantiles.
      4. Interpolate linearly between knots; clamp outside the observed z-range
         to the terminal knot values (no extrapolation).

    WHY ISOTONIC + WINDOW rather than, say, linear quantile regression in z:
      - Monotonicity is REQUIRED by the task (z1 < z2 => q(z1) <= q(z2)) and
        isotonic enforces it exactly rather than approximately.
      - It is non-parametric in shape: the whole point is that the fixed
        multiplier imposed a shape (linear in rank); imposing a different fixed
        shape would repeat the mistake.
      - PAVA is the standard, parameter-free monotone fit -- no smoothing
        constant hidden in the monotonisation step.

    THE ONE FREE PARAMETER — window size K:
        K = max(30, ceil(n_cal / 10))
      Rationale, fixed in advance from fit-period reasoning ONLY:
        * A window must contain enough points to estimate a 0.90 quantile at
          all. With fewer than ~30 points the 90th percentile is determined by
          the top 3 observations and is pure noise; 30 is the smallest window
          where it rests on >=3 order statistics with any stability. This is the
          same reasoning Task 22 used to prefer p99 over max at n=291.
        * n_cal/10 gives 10 effective difficulty levels, matching the resolution
          the fixed multiplier's rank map effectively had, so the comparison
          isolates SHAPE (learned vs imposed), not resolution.
      K is NOT selected by checking which value produces the best test-period
      frontier. No sweep over K against test data is run anywhere in this task.

NO-LEAKAGE: q_alpha(z) is estimated on the CALIBRATION split only. beta comes
from the FIT split only. The TEST split is never touched during estimation.
Item 4 validates on data disjoint from whatever fit q_alpha.

Exposed as fit_qalpha() / apply_qalpha() for Items 4 and 5.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

ALPHA = 0.10


def window_size(n_cal):
    """K fixed in advance from the stated rationale. Not tuned."""
    return int(max(30, np.ceil(n_cal / 10)))


def fit_qalpha(z_cal, s_cal, alpha=ALPHA):
    """
    Estimate the monotone conditional (1-alpha) quantile of s given z.
    Returns a callable q(z_new) -> width half-length.
    """
    z = np.asarray(z_cal, dtype=float)
    s = np.asarray(s_cal, dtype=float)
    ok = np.isfinite(z) & np.isfinite(s)
    z, s = z[ok], s[ok]
    order = np.argsort(z)
    z, s = z[order], s[order]
    n = len(z)
    K = window_size(n)
    if n < 5:
        const = float(np.quantile(s, 1 - alpha)) if n else 0.0
        return lambda zz: np.full(np.shape(zz), const, dtype=float), np.array([]), np.array([])

    # centred sliding-window empirical quantile
    half = K // 2
    knot_z, knot_q = [], []
    step = max(1, n // 200)                     # cap knots for tractability
    for i in range(0, n, step):
        lo = max(0, i - half)
        hi = min(n, lo + K)
        lo = max(0, hi - K)
        knot_z.append(z[i])
        knot_q.append(float(np.quantile(s[lo:hi], 1 - alpha)))
    knot_z = np.asarray(knot_z, dtype=float)
    knot_q = np.asarray(knot_q, dtype=float)

    # enforce monotone non-decreasing in z (PAVA)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    knot_q_mono = iso.fit_transform(knot_z, knot_q)

    lo_val, hi_val = float(knot_q_mono[0]), float(knot_q_mono[-1])

    def q(z_new):
        zz = np.asarray(z_new, dtype=float)
        out = np.interp(zz, knot_z, knot_q_mono, left=lo_val, right=hi_val)
        return out

    return q, knot_z, knot_q_mono


def apply_qalpha(qfun, z_test):
    """Half-widths for the test points."""
    return np.asarray(qfun(z_test), dtype=float)
