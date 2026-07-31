"""
Shared calibration-diversity diagnostic pipeline, domain-agnostic.

This module factors out the exact machinery already used for the recession
testbed (task_phase1.py / task_phase2.py) so that every new domain added under
Task 1 of the multidomain revision runs *identical* diagnostics rather than a
lookalike reimplementation. Nothing here is recession-specific.

Core objects:
  run_aci            — Gibbs & Candes adaptive conformal inference runner
                       (byte-for-byte the same update rule as task_phase1.py)
  support_width      — the paper's headline diversity statistic, p95 - p5
  fixed_size_ablation— vary ONLY rare-event count at fixed calibration size N
  random_draw_sweep  — 200 random fixed-size calibration draws, then
                       Spearman rho / R^2 of coverage against diversity vs
                       rare-count, plus the within-tertile redundancy check
  wilson_ci          — binomial (Wilson score) interval for a coverage %
  block_bootstrap_ci — moving-block bootstrap CI for autocorrelated series

Convention shared with the recession testbed: nonconformity score is the
absolute residual |prediction - actual|, and a test point is "covered" when the
actual falls in prediction +/- q_t.
"""

import numpy as np
import pandas as pd
from scipy import stats as _st

GAMMA_DEFAULT = 0.005
ALPHA_INIT = 0.10
ALPHA_TARGET = 0.10


# ── ACI ────────────────────────────────────────────────────────────────────────

def run_aci(y_test_h, pred_test_h, cal_scores,
            gamma=GAMMA_DEFAULT, alpha_init=ALPHA_INIT, alpha_target=ALPHA_TARGET):
    """Standard Gibbs & Candes ACI runner. Identical to task_phase1.py."""
    T = len(y_test_h)
    alpha_t = alpha_init
    covered, alpha_traj, widths = [], [], []

    for t in range(T):
        q_t = np.quantile(cal_scores, np.clip(1 - alpha_t, 0.0, 1.0))
        lo = pred_test_h[t] - q_t
        hi = pred_test_h[t] + q_t
        widths.append(2 * q_t)
        alpha_traj.append(alpha_t)

        y_t = y_test_h[t]
        if np.isnan(y_t):
            covered.append(np.nan)  # no alpha update when actual is unknown
        else:
            miss = 1 if (y_t < lo or y_t > hi) else 0
            covered.append(1 - miss)
            alpha_t = np.clip(alpha_t + gamma * (alpha_target - miss), 0.01, 0.99)

    return np.array(covered), alpha_traj, np.array(widths)


def coverage_and_width(covered, widths):
    """Empirical coverage (%) and mean interval width."""
    valid = ~np.isnan(covered)
    cov = float(np.nanmean(covered[valid]) * 100) if valid.sum() > 0 else np.nan
    return cov, float(np.mean(widths))


# ── Diversity statistics ───────────────────────────────────────────────────────

def support_width(s):
    """p95 - p5 of the nonconformity scores: the paper's headline diversity stat."""
    s = np.asarray(s)
    s = s[~np.isnan(s)]
    return float(np.percentile(s, 95) - np.percentile(s, 5)) if len(s) >= 2 else 0.0


def iqr(s):
    s = np.asarray(s); s = s[~np.isnan(s)]
    return float(np.percentile(s, 75) - np.percentile(s, 25)) if len(s) >= 2 else 0.0


def shannon(a, bins=20):
    a = np.asarray(a); a = a[~np.isnan(a)]
    if len(a) < 2:
        return 0.0
    h, _ = np.histogram(a, bins=bins)
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ── Confidence intervals (Task 4) ──────────────────────────────────────────────

def wilson_ci(k, n, conf=0.95):
    """Wilson score interval for a binomial proportion, returned in percent."""
    if n == 0:
        return (np.nan, np.nan)
    z = _st.norm.ppf(1 - (1 - conf) / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half) * 100), float(min(1.0, centre + half) * 100))


def wilson_ci_from_indicator(covered, conf=0.95):
    """Wilson CI directly from an ACI coverage indicator array (NaNs dropped)."""
    c = np.asarray(covered, dtype=float)
    c = c[~np.isnan(c)]
    return wilson_ci(int(c.sum()), int(len(c)), conf=conf)


def block_bootstrap_ci(covered, block=12, n_boot=2000, conf=0.95, seed=11):
    """
    Moving-block bootstrap CI (percent) for the mean of an autocorrelated 0/1
    coverage series. Block length defaults to 12 (one year of monthly data);
    callers with non-temporal units should pass block=1.
    """
    c = np.asarray(covered, dtype=float)
    c = c[~np.isnan(c)]
    n = len(c)
    if n == 0:
        return (np.nan, np.nan)
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block + 1
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        samp = np.concatenate([c[s:s + block] for s in starts])[:n]
        means[b] = samp.mean()
    lo, hi = np.quantile(means, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return (float(lo * 100), float(hi * 100))


# ── Fixed-size ablation (mirrors Table 1) ──────────────────────────────────────

def fixed_size_ablation(scores_all, is_rare, y_test, pred_test, rare_counts,
                        N, gamma=GAMMA_DEFAULT, seed=3, label="value"):
    """
    Hold calibration size N fixed and vary ONLY the number of rare-event units in
    it. This is the domain-general form of the recession testbed's Table 1.

    scores_all : (n_pool,) precomputed out-of-fold nonconformity scores
    is_rare    : (n_pool,) bool, whether each pool unit is a rare event
    rare_counts: list of rare-event counts k to place in the calibration set
    """
    rng = np.random.default_rng(seed)
    scores_all = np.asarray(scores_all, dtype=float)
    valid = np.where(~np.isnan(scores_all))[0]
    rare_idx = valid[is_rare[valid]]
    exp_idx = valid[~is_rare[valid]]

    rows = []
    for k in rare_counts:
        k_eff = min(k, len(rare_idx))
        n_exp = N - k_eff
        if n_exp > len(exp_idx):
            continue
        sel = np.concatenate([
            rng.choice(rare_idx, size=k_eff, replace=False) if k_eff > 0 else np.array([], dtype=int),
            rng.choice(exp_idx, size=n_exp, replace=False),
        ]).astype(int)
        s = scores_all[sel]
        s = s[~np.isnan(s)]
        covered, _, widths = run_aci(y_test, pred_test, s, gamma=gamma)
        cov, w = coverage_and_width(covered, widths)
        lo, hi = wilson_ci_from_indicator(covered)
        rows.append(dict(Rare_Units=k_eff, N=N,
                         Composition_pct=round(100 * k_eff / N, 2),
                         Coverage=round(cov, 2),
                         Wilson_lo=round(lo, 2), Wilson_hi=round(hi, 2),
                         Width=round(w, 3),
                         Support_width=round(support_width(s), 4)))
    return pd.DataFrame(rows)


# ── 200-draw diversity sweep (mirrors "Diversity Explains It") ─────────────────

def random_draw_sweep(scores_all, is_rare, y_test, pred_test, N,
                      n_draws=200, gamma=GAMMA_DEFAULT, seed=7):
    """
    Draw `n_draws` random fixed-size (N) calibration sets from the pool, and for
    each record diversity statistics, rare-event count, and realised ACI
    coverage. Returns the per-draw frame.
    """
    rng = np.random.default_rng(seed)
    scores_all = np.asarray(scores_all, dtype=float)
    valid = np.where(~np.isnan(scores_all))[0]
    N = min(N, len(valid))

    rows = []
    for d in range(n_draws):
        sel = rng.choice(valid, size=N, replace=False)
        s = scores_all[sel]
        covered, _, widths = run_aci(y_test, pred_test, s, gamma=gamma)
        cov, w = coverage_and_width(covered, widths)
        rows.append(dict(draw=d,
                         rare_count=int(is_rare[sel].sum()),
                         supp=round(support_width(s), 5),
                         IQR=round(iqr(s), 5),
                         ent=round(shannon(s), 5),
                         cov=round(cov, 4),
                         width=round(w, 4)))
    return pd.DataFrame(rows)


def sweep_diagnostics(sweep, domain, extra=None):
    """
    Reduce a random_draw_sweep frame to the row format of the cross-series table
    (Table 2), i.e. rho(diversity), rho(rare-count), R^2 for each, and the
    within-tertile redundancy check.
    """
    cov = sweep["cov"].values
    supp = sweep["supp"].values
    rare = sweep["rare_count"].values.astype(float)

    def _rho(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(_st.spearmanr(x, y).correlation)

    def _r2(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return float(_st.pearsonr(x, y).statistic ** 2)

    # Redundancy check: within support-width tertiles, does rare-count still
    # predict coverage? Averaged over tertiles, as in Phase 1.
    within = []
    if np.std(supp) > 0:
        terts = np.quantile(supp, [0, 1 / 3, 2 / 3, 1.0])
        for ti in range(3):
            lo, hi = terts[ti], terts[ti + 1]
            m = (supp >= lo) & (supp <= hi if ti == 2 else supp < hi)
            if m.sum() >= 6 and np.std(rare[m]) > 0 and np.std(cov[m]) > 0:
                within.append(_st.spearmanr(rare[m], cov[m]).correlation)
    within_rho = float(np.nanmean(within)) if within else np.nan

    row = dict(domain=domain,
               draws=int(len(sweep)),
               rho_supp=round(_rho(supp, cov), 3),
               R2_supp=round(_r2(supp, cov), 3),
               rho_rare=round(_rho(rare, cov), 3),
               R2_rare=round(_r2(rare, cov), 3),
               within_tertile_rho_rare=None if np.isnan(within_rho) else round(within_rho, 3),
               cov_mean=round(float(np.mean(cov)), 2),
               cov_std=round(float(np.std(cov)), 2))
    if extra:
        row.update(extra)
    return row


# ── Out-of-fold scoring helper ─────────────────────────────────────────────────

def oof_predictions(model_factory, X, y, folds):
    """
    Generic out-of-fold prediction generator. `folds` is an iterable of
    (train_idx, test_idx). Returns an array of OOF predictions (NaN where a unit
    was never in a test fold). Used so every new domain has honest out-of-fold
    nonconformity scores from the start, per Task 1/Task 3.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        m = model_factory()
        m.fit(X[tr], y[tr])
        oof[te] = _predict_scalar(m, X[te])
    return oof


def _predict_scalar(m, X):
    """Probability of the positive class if available, else raw prediction."""
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    return m.predict(X)


def blocked_folds(n, n_folds=5):
    """Contiguous (blocked) folds — the temporal-safe scheme for ordered data."""
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.concatenate([np.arange(0, edges[i]), np.arange(edges[i + 1], n)])
        if len(te) and len(tr):
            yield tr, te


def rolling_origin_folds(n, n_folds=5, min_train=None):
    """
    Forward-chaining folds: train only on the past, predict the next block.
    This is the leakage-safe scheme for the recession testbed's Task 3 re-run.
    """
    min_train = min_train or max(1, n // (n_folds + 1))
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    for i in range(n_folds):
        tr = np.arange(0, edges[i])
        te = np.arange(edges[i], edges[i + 1])
        if len(te) and len(tr):
            yield tr, te
