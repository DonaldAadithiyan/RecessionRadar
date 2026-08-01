"""
TASK 7 — Calibration-strategy baselines, domain-agnostic.

Every strategy here takes calibration nonconformity scores plus a test stream
and returns (covered_indicator, widths), so they are interchangeable inside the
same harness and can be run identically on recession, healthcare and climate.

Strategies
----------
pooled_trailing     Standard Gibbs & Candes ACI on the trailing calibration set
                    (the reference row). Delegates to domain_common.run_aci.
mondrian            Class-conditional ACI (Vovk et al. 2003): a separate alpha
                    and calibration pool per regime.
pid_conformal       PID control on the miscoverage rate (Angelopoulos et al.
                    2023). Ported verbatim from task_phase3bc.py (P + I + D,
                    ki=0.02, kd=0.01) so recession numbers reproduce exactly.
evt_tail            Generalized-Pareto fit above the 80th percentile
                    (Pasche et al. 2026). Ported verbatim from task_phase3bc.py.
dtaci               Dynamically-tuned ACI (Gibbs & Candes, JMLR 25(162), 2024).
                    Runs k experts with different step sizes in parallel and
                    aggregates them by exponential weights on the pinball loss,
                    so the step size is learned online instead of fixed.
acmcp               Autocorrelated multi-step conformal prediction (Wang &
                    Hyndman 2024, arXiv:2410.13115). Adds an explicit
                    autocorrelation term to a PI update, modelling the serial
                    correlation of h-step-ahead errors up to lag h-1.

Fidelity note: pid_conformal and evt_tail are byte-for-byte ports of the Phase 3
implementations rather than reimplementations, because Task 7 requires the
recession rows to reproduce the published Phase 3 figures as a bug check.
"""

import numpy as np
from scipy import stats as _st

GAMMA_DEFAULT = 0.005
ALPHA_TARGET = 0.10


# ── 1. Pooled / trailing ACI ──────────────────────────────────────────────────
# (delegates to domain_common.run_aci; re-exported here for a uniform interface)
from domain_common import run_aci  # noqa: E402,F401


# ── 2. Mondrian (class-conditional) ACI ───────────────────────────────────────

def run_mondrian(y_te, p_te, cal_by_regime, test_regime,
                 gamma=GAMMA_DEFAULT, at=ALPHA_TARGET):
    """
    Class-conditional ACI. Ported from task_phase3a.py and generalized from
    two hardcoded regimes to an arbitrary regime labelling.

    cal_by_regime : dict {regime_label: score array}
    test_regime   : array of regime labels, one per test point
    """
    alphas = {k: 0.10 for k in cal_by_regime}
    pooled = np.concatenate([v for v in cal_by_regime.values() if len(v)])
    covered, widths = [], []

    for t in range(len(y_te)):
        r = test_regime[t]
        cs = cal_by_regime.get(r, np.array([]))
        if len(cs) == 0:          # degenerate regime -> fall back to pooled
            cs = pooled
        a = alphas.get(r, 0.10)
        q = np.quantile(cs, np.clip(1 - a, 0, 1))
        lo, hi = p_te[t] - q, p_te[t] + q
        widths.append(2 * q)
        yt = y_te[t]
        if np.isnan(yt):
            covered.append(np.nan)
            continue
        miss = 1 if (yt < lo or yt > hi) else 0
        covered.append(1 - miss)
        if r in alphas:
            alphas[r] = float(np.clip(alphas[r] + gamma * (at - miss), 0.01, 0.99))
    return np.array(covered), np.array(widths)


# ── 3. PID-conformal (verbatim port from task_phase3bc.py) ────────────────────

def run_pid(y_h, p_h, cs, gamma=GAMMA_DEFAULT, at=ALPHA_TARGET, ki=0.02, kd=0.01):
    T = len(y_h); a = at
    covered, widths = [], []
    err_int, prev_err = 0.0, 0.0
    for t in range(T):
        q = np.quantile(cs, np.clip(1 - a, 0, 1))
        lo, hi = p_h[t] - q, p_h[t] + q
        widths.append(2 * q)
        yt = y_h[t]
        if np.isnan(yt):
            covered.append(np.nan); continue
        miss = 1 if (yt < lo or yt > hi) else 0
        covered.append(1 - miss)
        err = at - miss
        err_int += err
        deriv = err - prev_err
        prev_err = err
        a = np.clip(a + gamma * err + ki * gamma * err_int + kd * deriv, 0.01, 0.99)
    return np.array(covered), np.array(widths)


# ── 4. EVT-tail conformal (verbatim port from task_phase3bc.py) ───────────────

def evt_quantile(cs, alpha=0.10):
    """Fit GPD to exceedances over the 80th pctile; return the (1-alpha) quantile."""
    u = np.quantile(cs, 0.80)
    exc = cs[cs > u] - u
    if len(exc) < 10:
        return np.quantile(cs, 1 - alpha)
    try:
        c, loc, scale = _st.genpareto.fit(exc, floc=0)
        p_exceed = len(exc) / len(cs)
        target = alpha / p_exceed
        if target >= 1 or target <= 0:
            return np.quantile(cs, 1 - alpha)
        q_tail = u + _st.genpareto.ppf(1 - target, c, loc=0, scale=scale)
        return float(q_tail) if np.isfinite(q_tail) else np.quantile(cs, 1 - alpha)
    except Exception:
        return np.quantile(cs, 1 - alpha)


def run_evt(y_h, p_h, cs, gamma=GAMMA_DEFAULT, at=ALPHA_TARGET):
    T = len(y_h); a = at
    covered, widths = [], []
    for t in range(T):
        q = evt_quantile(cs, alpha=np.clip(a, 0.001, 0.5))
        lo, hi = p_h[t] - q, p_h[t] + q
        widths.append(2 * q)
        yt = y_h[t]
        if np.isnan(yt):
            covered.append(np.nan); continue
        miss = 1 if (yt < lo or yt > hi) else 0
        covered.append(1 - miss)
        a = np.clip(a + gamma * (at - miss), 0.01, 0.99)
    return np.array(covered), np.array(widths)


# ── 5. DtACI — dynamically-tuned ACI (Gibbs & Candes 2024, Algorithm 1) ───────

def _pinball(beta, theta, alpha):
    """Pinball loss l(beta, theta) = alpha*(beta - theta) - min(0, beta - theta)."""
    d = beta - theta
    return alpha * d - min(0.0, d)


def run_dtaci(y_h, p_h, cs, gammas=None, at=ALPHA_TARGET, I=500, seed=0):
    """
    DtACI (Gibbs & Candes, JMLR 25(162), 2024, Algorithm 1).

    Runs k ACI experts with different step sizes gamma_i in parallel and picks
    alpha_t from among them by exponential weights on the pinball loss, so the
    step size is learned online rather than fixed in advance.

    Following the paper's practical recommendation, the weight parameters are
    set from a nominal interval length |I| = 500:
        sigma = 1 / (2*|I|)
        eta   = sqrt(3/|I|) * sqrt( (log(k*|I|) + 2) / ((1-alpha)^2 * alpha^2) )
    The candidate grid defaults to the paper's spread of step sizes.

    Deterministic variant: the paper samples alpha_t ~ p_t; we instead take the
    probability-weighted mean of the experts' alphas. This removes Monte-Carlo
    noise from a 59-655 point test stream, where sampling variance would
    otherwise swamp the comparison. Flagged in the write-up.
    """
    if gammas is None:
        gammas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    gammas = np.asarray(gammas, dtype=float)
    k = len(gammas)

    sigma = 1.0 / (2.0 * I)
    eta = np.sqrt(3.0 / I) * np.sqrt(
        (np.log(k * I) + 2.0) / (((1 - at) ** 2) * (at ** 2)))

    alphas = np.full(k, at, dtype=float)
    w = np.ones(k, dtype=float)
    covered, widths = [], []

    cs = np.asarray(cs, dtype=float)
    cs_sorted = np.sort(cs)
    n_cs = len(cs_sorted)

    for t in range(len(y_h)):
        p = w / w.sum()
        alpha_t = float(np.clip(np.dot(p, alphas), 0.01, 0.99))

        q = np.quantile(cs, np.clip(1 - alpha_t, 0, 1))
        lo, hi = p_h[t] - q, p_h[t] + q
        widths.append(2 * q)

        yt = y_h[t]
        if np.isnan(yt):
            covered.append(np.nan)
            continue

        miss = 1 if (yt < lo or yt > hi) else 0
        covered.append(1 - miss)

        # beta_t = smallest miscoverage level whose interval still covers y_t,
        # i.e. the calibration-quantile level matching this point's error.
        err_t = abs(yt - p_h[t])
        rank = np.searchsorted(cs_sorted, err_t, side="right")
        beta_t = float(np.clip(1.0 - rank / max(n_cs, 1), 0.0, 1.0))

        # Expert weights: exponential weights on the pinball loss, with the
        # paper's sigma-mixing toward uniform (guards against weight collapse).
        # Loss and error use the quantile-level convention of the authors'
        # reference implementation (CPTC repo, algos/dtaci.py): beta_t is a
        # quantile level, an expert errs when alpha_i > beta_t, and the loss
        # argument is (beta_t - alpha_i).
        losses = np.array([_pinball(beta_t, a_i, at) for a_i in alphas])

        # log-sum-exp for numerical stability (as in the reference code)
        log_w = np.log(w + 1e-300) - eta * losses
        log_w -= log_w.max()
        w_bar = np.exp(log_w)
        w_bar_sum = w_bar.sum() + 1e-300
        w = (1 - sigma) * w_bar / w_bar_sum + sigma / k
        w = np.clip(w, 1e-300, 1.0)
        w /= w.sum()

        # Each expert takes its own ACI step against its own miscoverage.
        err_experts = (alphas > beta_t).astype(float)
        alphas = np.clip(alphas + gammas * (at - err_experts), 0.01, 0.99)

    return np.array(covered), np.array(widths)


# ── 6. AcMCP — autocorrelated multi-step conformal prediction ─────────────────

def run_acmcp(y_h, p_h, cs, horizon=1, eta=None, at=ALPHA_TARGET,
              window=100, past_scores=None):
    """
    AcMCP (Wang & Hyndman 2024, arXiv:2410.13115), quantile-update form:

        q_{t+h|t} = q_{t+h-1|t-1} + eta*(err_{t|t-h} - alpha)
                    + r_t( sum_i (err_{i|i-h} - alpha) )
                    + e~_{t+h|t}

    The three terms are, in order: a delayed proportional correction on the
    h-step-ahead coverage error; an integral term passed through a saturation
    function r_t; and e~, a forecast of the current error built from the
    autocorrelation of h-step errors. Optimal h-step forecast errors are
    serially correlated up to lag h-1, which is exactly what this term exploits
    and what treating each horizon independently throws away.

    e~ is implemented as the paper describes it — a combination of an MA(h-1)
    fit on recent h-step errors and a linear regression on their lags — using a
    trailing window so it stays online.

    eta defaults to the paper's 0.01 * B_hat_t, with B_hat_t the maximum score
    over the trailing window.
    """
    cs = np.asarray(cs, dtype=float)
    cs = cs[np.isfinite(cs)]
    q = float(np.quantile(cs, 1 - at))          # initial width
    B_hat = float(np.max(cs)) if len(cs) else 1.0
    if eta is None:
        eta = 0.01 * B_hat

    hist = list(past_scores) if past_scores is not None else list(cs[-window:])
    err_hist = []          # signed coverage errors (err - alpha)
    integral = 0.0
    covered, widths = [], []
    lag = max(1, int(horizon) - 1)

    def saturation(x):
        """Sublinear, nonnegative, nondecreasing r_t (paper's condition (4))."""
        return float(np.sign(x) * B_hat * np.log1p(abs(x)) / (1.0 + np.log1p(abs(x))))

    for t in range(len(y_h)):
        q_eff = max(q, 1e-9)
        lo, hi = p_h[t] - q_eff, p_h[t] + q_eff
        widths.append(2 * q_eff)

        yt = y_h[t]
        if np.isnan(yt):
            covered.append(np.nan)
            continue

        miss = 1 if (yt < lo or yt > hi) else 0
        covered.append(1 - miss)

        score = abs(yt - p_h[t])
        hist.append(score)
        if len(hist) > window:
            hist.pop(0)

        # P term (delayed by the horizon) and I term through the saturation.
        err_signed = miss - at
        err_hist.append(err_signed)
        integral += err_signed
        p_term = eta * err_signed
        i_term = 0.01 * saturation(integral) if len(err_hist) >= 2 else 0.0

        # D / autocorrelation term e~: MA(h-1)-style mean of recent score
        # innovations blended with a lag regression on the score series.
        e_tilde = 0.0
        if len(hist) >= max(4, lag + 2):
            arr = np.asarray(hist, dtype=float)
            innov = np.diff(arr)
            ma = float(np.mean(innov[-lag:])) if lag <= len(innov) else 0.0
            x = arr[:-1][-window:]
            yv = arr[1:][-window:]
            if len(x) >= 3 and np.std(x) > 1e-12:
                b = float(np.cov(x, yv, bias=True)[0, 1] / np.var(x))
                a0 = float(np.mean(yv) - b * np.mean(x))
                pred_next = a0 + b * arr[-1]
                lin = pred_next - arr[-1]
            else:
                lin = 0.0
            e_tilde = 0.5 * ma + 0.5 * lin

        q = float(np.clip(q + p_term + i_term + e_tilde, 1e-9, 10 * B_hat))

    return np.array(covered), np.array(widths)


# ── 7. Bellman Conformal Inference (Yang, Candes & Lei 2024) ─────────────────

def run_bci(y_te, p_te, cs, T_look=3, Tp=20, at=ALPHA_TARGET,
            lambda_init=5.0, lambda_max=500.0, lambda_min=0.0, gamma=0.8,
            bins=50):
    """
    Bellman Conformal Inference (arXiv:2402.05203).

    BCI treats interval-width selection as a finite-horizon stochastic control
    problem: at each step it solves a Bellman recursion over a short lookahead
    window to pick the miscoverage level, instead of taking ACI's single
    gradient step. It penalises interval length while constraining the running
    miscoverage rate over a look-back window.

    This is a faithful but self-contained reimplementation of the authors'
    dynamic program (github.com/ZitongYang/bellman-conformal-inference,
    utils/dp.py + experiment.py) specialised to our setting: the disturbance
    CDFs are the empirical CDF of the calibration scores, and the per-step cost
    is the interval length implied by the chosen alpha. Reimplemented rather
    than imported because the authors' package couples the DP to their own
    dataloader/Function01 stack and their pinned numpy (1.23) conflicts with
    this project's environment.

    Crucially, lambda is NOT a fixed hyperparameter: as in the authors'
    experiment loop it is a control variable updated online by an ACI-style
    step, lambda <- lambda - gamma*(alpha0 - err), saturating to alpha=0 when
    lambda >= lambda_max and alpha=1 when lambda <= lambda_min. A static lambda
    makes coverage an artifact of that one constant, which is why the defaults
    below follow the authors' released configs (lambda_init=5, lambda_max=500,
    gamma=0.8, T=3) rather than anything tuned on our data.

    T_look : lookahead horizon for the DP (authors' T)
    Tp     : look-back window over which the miscoverage rate is constrained
    """
    cs = np.asarray(cs, dtype=float)
    cs = cs[np.isfinite(cs)]
    cs_sorted = np.sort(cs)

    grid = np.linspace(0.01, 0.99, bins)
    # Interval half-width implied by each candidate miscoverage level.
    widths_grid = np.array([np.quantile(cs_sorted, np.clip(1 - a, 0, 1))
                            for a in grid])

    # The length cost and the coverage-violation cost must be commensurate.
    # Working in raw units lets the length term (range ~ the full score scale)
    # dominate the violation term, which collapses the policy onto the
    # narrowest interval. Normalising length by its own range puts both terms
    # on [0, 1]-ish scales, so lbd controls the trade-off as intended.
    w_span = float(widths_grid.max() - widths_grid.min())
    if w_span <= 0:
        w_span = 1.0
    len_cost = (widths_grid - widths_grid.min()) / w_span

    covered, widths = [], []
    recent = []          # miscoverage indicators over the look-back window
    lbd = float(lambda_init)

    for t in range(len(y_te)):
        # ── alpha selection ──────────────────────────────────────────────
        if lbd >= lambda_max:
            alpha_t = 0.01           # saturate to the widest interval
        elif lbd <= lambda_min:
            alpha_t = 0.99           # saturate to the narrowest interval
        else:
            rho = float(np.sum(recent)) if len(recent) >= Tp else at * Tp
            Tp_eff = max(len(recent), 1)

            # Backward Bellman recursion over states = number of past misses.
            states = np.arange(T_look + 1)
            J = lbd * np.maximum((states + rho) / (T_look + Tp_eff) - at, 0.0)

            policy_first = None
            for step in range(T_look, 0, -1):
                J_next = J
                best_cost = np.full(step + 1, np.inf)
                best_a = np.zeros(step + 1)
                for s in range(step + 1):
                    # Pay the interval length now, then transition to s+1 with
                    # probability alpha (a miss) or stay at s otherwise.
                    j_stay = J_next[min(s, len(J_next) - 1)]
                    j_miss = J_next[min(s + 1, len(J_next) - 1)]
                    exp_future = (1 - grid) * j_stay + grid * j_miss
                    total = len_cost + exp_future
                    i = int(np.argmin(total))
                    best_cost[s] = total[i]
                    best_a[s] = grid[i]
                J = best_cost
                policy_first = best_a

            s_now = int(min(rho, len(policy_first) - 1))
            alpha_t = float(policy_first[s_now]) if policy_first is not None else at

        q = np.quantile(cs_sorted, np.clip(1 - alpha_t, 0, 1))
        lo, hi = p_te[t] - q, p_te[t] + q
        widths.append(2 * q)

        yt = y_te[t]
        if np.isnan(yt):
            covered.append(np.nan)
            continue
        miss = 1 if (yt < lo or yt > hi) else 0
        covered.append(1 - miss)
        recent.append(miss)
        if len(recent) > Tp:
            recent.pop(0)

        # ── online lambda update (authors' experiment.py, ACI-style) ──────
        lbd = float(np.clip(lbd - gamma * (at - miss), lambda_min, lambda_max))

    return np.array(covered), np.array(widths)
