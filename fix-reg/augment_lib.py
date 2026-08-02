"""
EVT pool augmentation for conformal calibration.

THE DISTINCTION FROM PHASE 3b (which failed, 66% at 6M)
-------------------------------------------------------
Phase 3b used a GPD fit to REPLACE the quantile estimator: at each ACI step it
computed the (1-alpha) quantile from the fitted tail instead of from the
empirical scores. That makes every interval depend on a parametric extrapolation,
including at alphas well inside the observed data, where the empirical quantile
was already correct. It hurt.

This module does something different: the GPD is used only to ADD synthetic
scores to the calibration pool, above the fitting threshold. ACI then runs
ordinarily, taking empirical quantiles of the augmented pool. Below the
threshold the pool is untouched, so the parametric model can only influence the
tail region it was fitted for — never the body of the distribution.

Motivation (Task 10): the existing selector is already at 100% of the achievable
quantile-reach ceiling for a fixed pool. Selection cannot go further. The only
remaining lever is to change WHAT IS IN THE POOL.

GUARDS (each addresses a specific way this can go wrong)
--------------------------------------------------------
1. Bounded-tail check. If the fitted shape xi < 0 the GPD has a finite endpoint;
   augmentation cannot reach past it. Cells like that are reported, not forced.
2. Extrapolation cap. Synthetic scores are capped at `max_extrap` times the
   observed maximum, so a near-zero xi cannot produce absurd draws.
3. Body preservation. Only scores above the threshold are added; the empirical
   body is never resampled or reweighted.
4. Fraction cap. Synthetic scores are limited to a fraction of the pool, so the
   calibration set never becomes majority-synthetic.
"""

import numpy as np
from scipy import stats


def fit_tail(scores, threshold_q=0.80, min_exceedances=15):
    """
    Fit a GPD to exceedances above the threshold_q quantile.
    Returns (u, xi, sigma, n_exc) or None if the tail is not fittable.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 40:
        return None
    u = float(np.quantile(s, threshold_q))
    exc = s[s > u] - u
    if len(exc) < min_exceedances:
        return None
    try:
        xi, _, sigma = stats.genpareto.fit(exc, floc=0)
    except Exception:
        return None
    if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
        return None
    return float(u), float(xi), float(sigma), int(len(exc))


def tail_endpoint(u, xi, sigma):
    """Finite upper endpoint of the fitted GPD, or inf when xi >= 0."""
    return (u - sigma / xi) if xi < -1e-9 else np.inf


def augment_pool(scores, n_synth=None, synth_frac=0.20, threshold_q=0.80,
                 max_extrap=2.0, seed=0, min_exceedances=15):
    """
    Return (augmented_scores, info). Adds synthetic tail draws to the pool.

    n_synth      : number of synthetic scores; defaults to synth_frac * len(pool)
    max_extrap   : cap synthetic draws at max_extrap * observed max
    threshold_q  : GPD fitting threshold

    info records what actually happened, including refusals, so the caller can
    report cells where augmentation was declined rather than silently skipped.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    info = dict(n_original=int(len(s)), n_synth=0, applied=False, reason="")

    fit = fit_tail(s, threshold_q=threshold_q, min_exceedances=min_exceedances)
    if fit is None:
        info["reason"] = "tail not fittable"
        return s, info
    u, xi, sigma, n_exc = fit
    obs_max = float(s.max())
    ep = tail_endpoint(u, xi, sigma)

    info.update(u=round(u, 4), xi=round(xi, 4), sigma=round(sigma, 4),
                n_exceedances=n_exc, obs_max=round(obs_max, 4),
                endpoint=(None if not np.isfinite(ep) else round(ep, 4)))

    # GUARD 1: a bounded tail whose endpoint is at or below the observed max
    # cannot extend anything.
    if np.isfinite(ep) and ep <= obs_max:
        info["reason"] = ("bounded tail with endpoint <= observed max "
                          f"({ep:.3f} <= {obs_max:.3f})")
        return s, info

    if n_synth is None:
        n_synth = int(round(synth_frac * len(s)))
    n_synth = max(0, int(n_synth))
    if n_synth == 0:
        info["reason"] = "n_synth resolved to 0"
        return s, info

    rng = np.random.default_rng(seed)
    # Draw exceedances from the fitted GPD and place them above the threshold.
    draws = u + stats.genpareto.rvs(xi, loc=0, scale=sigma, size=n_synth,
                                    random_state=rng)

    # GUARD 2: cap extrapolation so a near-zero xi cannot yield absurd values.
    cap = max_extrap * obs_max
    n_capped = int((draws > cap).sum())
    draws = np.clip(draws, u, cap)

    # GUARD 3 is structural: only values above u are added; the body is untouched.
    out = np.concatenate([s, draws])
    info.update(applied=True, n_synth=int(n_synth), n_capped=n_capped,
                cap=round(float(cap), 4),
                synth_max=round(float(draws.max()), 4),
                pool_q90_before=round(float(np.quantile(s, 0.90)), 4),
                pool_q90_after=round(float(np.quantile(out, 0.90)), 4),
                reason="ok")
    return out, info


def augment_selected(scores_pool, selector_idx, n_synth=None, synth_frac=0.20,
                     threshold_q=0.80, max_extrap=2.0, seed=0):
    """
    Augment the SELECTED calibration set rather than the whole pool.

    This is the combination worth testing: the selector already extracts the
    maximum achievable tail reach from the finite pool (Task 10), and
    augmentation then extends beyond that ceiling. Fitting the GPD on the
    selected set is deliberate — that set is tail-loaded by construction, so the
    exceedances above its own 80th percentile are the relevant ones.
    """
    s = np.asarray(scores_pool, dtype=float)[selector_idx]
    s = s[np.isfinite(s)]
    return augment_pool(s, n_synth=n_synth, synth_frac=synth_frac,
                        threshold_q=threshold_q, max_extrap=max_extrap,
                        seed=seed)
