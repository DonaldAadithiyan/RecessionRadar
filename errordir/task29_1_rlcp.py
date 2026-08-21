"""
TASK 29, Item 1 — RLCP implemented per the published method.

Hore & Barber, "Conformal prediction with local weights: randomization enables
robust guarantees" (arXiv:2310.07850).

WHY THIS IS A REIMPLEMENTATION RATHER THAN A REUSE
--------------------------------------------------
Task 28 built a kernel-weighted localized conformal function and called it RLCP.
Checked against the paper, it is missing the two features that DEFINE RLCP and
that deliver its guarantees:

  1. THE RANDOMIZATION STEP. RLCP samples a synthetic localization point
     Xtilde_{n+1} ~ H(X_{n+1}, .) and centres the kernel weights at Xtilde,
     NOT at the test point itself. This randomization is exactly what the
     paper's title refers to and what buys marginal validity. Task 28's version
     centres deterministically at the test point -- that is baseLCP, the
     un-randomized variant the paper contrasts RLCP against.

  2. THE +INFINITY MASS. The weighted quantile is taken over
         sum_i wtilde_i * delta_{s_i}  +  wtilde_{n+1} * delta_{+inf}
     i.e. the test point contributes its own weight as mass at +infinity, and
     the denominator normalises over n+1 points, not n. Task 28 normalised over
     n calibration points only and omitted the +inf atom, which makes the
     interval anti-conservative.

So Task 28's numbers are baseLCP numbers, not RLCP numbers. Per the Item 1
guardrail ("if RLCP's step differs materially, implement it as published, not as
a convenient stand-in"), this file implements the published algorithm. Task 28's
function is retained here as `baselcp_intervals` so the difference is visible and
both can be reported.

ALGORITHM (per the paper)
-------------------------
  Given calibration (X_i, s_i) i=1..n, test point X_{n+1}, level alpha:
    1. Sample Xtilde ~ H(X_{n+1}, .).  For a Gaussian kernel
       H(x, x') propto exp(-gamma ||x - x'||^2), this is
       Xtilde ~ N(X_{n+1}, 1/(2*gamma)) per coordinate.
    2. wtilde_i propto H(X_i, Xtilde) for i in 1..n+1, normalised over ALL n+1.
    3. qhat = Quantile_{1-alpha} of  sum_{i<=n} wtilde_i delta_{s_i}
                                     + wtilde_{n+1} delta_{+inf}
    4. Interval = pred(X_{n+1}) +/- qhat.
  If the cumulative weight of the finite atoms never reaches 1-alpha, the
  quantile IS +infinity and the interval is unbounded -- handled explicitly.

The kernel operates in the SAME 1-D coordinate z = beta'x that errordir uses.
This is deliberate and is the fairest possible comparison: it isolates
"kernel-weighted local quantile" vs "rank-based fixed multiplier" on identical
geometry, rather than confounding the mechanism with a different representation.

BANDWIDTH: gamma is a free parameter of RLCP. It is set by a FIT-SPLIT-ONLY
median-heuristic (gamma = 1 / (2 * median pairwise squared distance among FIT
projections)), the standard kernel default. It is NOT tuned against test-period
results -- the same discipline every parameter in this project has followed
since Task 22. A sensitivity band is reported in Item 2 so RLCP is not
disadvantaged by one unlucky bandwidth.

Outputs: errordir/task29_1_rlcp_verify.csv
"""
import numpy as np

ALPHA = 0.10


def median_heuristic_gamma(z_fit):
    """Standard kernel bandwidth default, computed on FIT projections only."""
    z = np.asarray(z_fit, dtype=float)
    z = z[np.isfinite(z)]
    if len(z) > 2000:                      # subsample for tractability
        z = np.random.default_rng(0).choice(z, 2000, replace=False)
    d2 = (z[:, None] - z[None, :]) ** 2
    med = float(np.median(d2[np.triu_indices_from(d2, k=1)]))
    return 1.0 / (2.0 * max(med, 1e-12))


def rlcp_intervals(z_cal, s_cal, z_test, pred_test, gamma, alpha=ALPHA, seed=29):
    """
    RLCP per Hore & Barber: randomized localization point + (n+1) normalisation
    + point mass at +infinity for the test point.
    """
    rng = np.random.default_rng(seed)
    zc = np.asarray(z_cal, dtype=float)
    sc = np.asarray(s_cal, dtype=float)
    zt = np.asarray(z_test, dtype=float)
    ok = np.isfinite(zc) & np.isfinite(sc)
    zc, sc = zc[ok], sc[ok]
    order = np.argsort(sc)
    zc_o, sc_o = zc[order], sc[order]

    # H(x,x') ∝ exp(-gamma (x-x')^2)  ->  Xtilde ~ N(x, 1/(2 gamma))
    sd = np.sqrt(1.0 / (2.0 * gamma))
    z_tilde = zt + rng.normal(0.0, sd, size=len(zt))

    los, his, unbounded = [], [], 0
    for t in range(len(zt)):
        w_cal = np.exp(-gamma * (zc_o - z_tilde[t]) ** 2)      # i = 1..n
        w_test = float(np.exp(-gamma * (zt[t] - z_tilde[t]) ** 2))  # i = n+1
        tot = float(w_cal.sum()) + w_test
        if tot <= 0:
            q = float(np.quantile(sc_o, 1 - alpha))
        else:
            cw = np.cumsum(w_cal) / tot            # normalised over ALL n+1
            # the remaining mass w_test/tot sits at +infinity
            idx = int(np.searchsorted(cw, 1 - alpha))
            if idx >= len(sc_o):
                # finite atoms never reach 1-alpha -> quantile is +infinity
                q = np.inf
                unbounded += 1
            else:
                q = float(sc_o[idx])
        los.append(pred_test[t] - q); his.append(pred_test[t] + q)
    return np.array(los), np.array(his), unbounded


def baselcp_intervals(z_cal, s_cal, z_test, pred_test, gamma, alpha=ALPHA):
    """
    Task 28's construction: deterministic centring at the test point, weights
    normalised over n calibration points only, no +inf atom. This is baseLCP,
    retained so Task 28's numbers can be reproduced and the difference shown.
    """
    zc = np.asarray(z_cal, float); sc = np.asarray(s_cal, float)
    zt = np.asarray(z_test, float)
    order = np.argsort(sc)
    zc_o, sc_o = zc[order], sc[order]
    los, his = [], []
    for t in range(len(zt)):
        w = np.exp(-gamma * (zc_o - zt[t]) ** 2)
        cw = np.cumsum(w) / max(w.sum(), 1e-12)
        idx = min(max(int(np.searchsorted(cw, 1 - alpha)), 0), len(sc_o) - 1)
        q = float(sc_o[idx])
        los.append(pred_test[t] - q); his.append(pred_test[t] + q)
    return np.array(los), np.array(his)
