"""
TASK 21, Item 2 — Width-scaling design.

*** WRITTEN AND FIXED BEFORE ANY ITEM 3 RESULT EXISTS. ***
Every constant below is set from the stated rationale in this file. None is
tuned against the comparison outcome.

THE SCALING FUNCTION (quantile-mapped, not linear in the raw projection)

    s_t   = beta . x_t                       raw difficulty projection
    u_t   = F_CAL(s_t)                       mapped to [0,1] by the CAL-split
                                             projection's own empirical CDF
    m_t   = LO + (HI - LO) * u_t             multiplier, linear in the RANK
    width_t = m_t * 2 * Q_C(q_t)             the ACI interval, scaled

WHY QUANTILE-MAPPED RATHER THAN LINEAR IN s_t:
  A linear map is not scale-free — its effect depends on the arbitrary units and
  spread of beta.x, which vary by horizon, so one shared parameter set could not
  mean the same thing across horizons. The rank map is invariant to any monotone
  transform of the projection, so LO/HI carry identical meaning everywhere.
  It is also robust to outliers in beta.x, which a linear map is not.

WHY F_CAL AND NOT F_TEST:
  The CDF is estimated on the CAL split ONLY. Using the test projections' own
  CDF would leak the test distribution into the interval construction. F_CAL is
  frozen before any test point is seen.

PARAMETERS, FIXED IN ADVANCE:
  LO = 0.75, HI = 1.25
    Rationale: a symmetric +/-25% band around the unscaled interval. Symmetric
    about 1.0 so the scaling is width-neutral ON AVERAGE at a uniform rank
    distribution (mean multiplier = (0.75+1.25)/2 = 1.0) — the method must earn
    any width reduction by allocating width differently across points, NOT by
    uniformly shrinking every interval, which would be a trivial and dishonest
    way to "win" on width while losing coverage. +/-25% is a moderate band: wide
    enough for the signal to act, narrow enough that a wrong beta cannot produce
    catastrophically narrow intervals (floor is 0.75x, not 0x).

  A NOTE ON WHAT THIS DESIGN DELIBERATELY GIVES UP:
    Because the mean multiplier is pinned to 1.0, this method CANNOT win on
    width by simple uniform shrinkage. If it produces narrower mean width, that
    can only come from correlation between the multiplier and the realized
    error. This makes the test harder to pass and the result harder to fake.
"""
import numpy as np

LO = 0.75
HI = 1.25


def fit_cdf(cal_projections):
    """Freeze the projection CDF on the CAL split. Returns a mapping function."""
    ref = np.sort(np.asarray(cal_projections, dtype=float))
    ref = ref[np.isfinite(ref)]

    def to_rank(s):
        s = np.asarray(s, dtype=float)
        return np.searchsorted(ref, s, side="right") / max(len(ref), 1)

    return to_rank


def multiplier(u, lo=LO, hi=HI):
    """Rank u in [0,1] -> width multiplier in [lo,hi], linear in rank."""
    return lo + (hi - lo) * np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
