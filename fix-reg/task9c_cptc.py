"""
TASK 9 PART C — CPTC with an HMM-based state input (NOT the original REDSDS).

*** PROTOCOL EXCEPTION, DISCLOSED UP FRONT ***
Every other strategy in this paper consumes fixed, pre-computed nonconformity
scores and fits nothing. CPTC cannot: its algorithm requires a per-timestep
state-probability matrix (z_prob) and per-state predictive means (z_mean) from a
switching-dynamics model. This script therefore fits ONE latent-state model per
domain — the only model fit anywhere in the Task 7/8/9 evaluation — on the
training partition only, purely to supply CPTC's state input. This is stated
here, in the write-up's first paragraph, and in the results table's label,
rather than blended in with the other seven strategies.

WHAT THIS IS NOT: the CPTC authors (Sun & Yu, NeurIPS 2025) use a REDSDS
(Recurrent Explicit Duration Switching Dynamical System). Their repository ships
only precomputed REDSDS *inference outputs* for their six datasets, not a
trainer, and training a REDSDS per domain is outside this task's scope. Per the
task specification's authorised fallback, the state model here is a Gaussian HMM
fit solely to produce z_prob in the shape CPTC expects. Results are therefore
labelled "CPTC (HMM state input)" throughout and must not be read as the
published method's performance — the same convention used for the 5-component
HOSPITAL variant in Task 8.

The conformal machinery below is a faithful port of the authors' released
algorithm (github.com/Rose-STL-Lab/CPTC, algos/cptc.py): per-state residual
pools, per-state alpha updates, the smallest-set union over states ordered by
state probability, and the sampled-state update rule.

Outputs:
  fix-reg/task9c_cptc.csv
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
    coverage_and_width, wilson_ci_from_indicator, block_bootstrap_ci,
)

OUT = "fix-reg"
N_FIX = 254
ALPHA = 0.10
GAMMA = 0.2          # authors' default in conformal_prediction_cptc
MIN_RESIDUALS = 25   # authors' default
SEED = 11


# ── Gaussian HMM (EM), self-contained: state model for the exception ────────
class GaussianHMM:
    """
    Minimal Gaussian HMM with EM (Baum-Welch). Written out rather than pulled
    from hmmlearn to keep the disclosed exception auditable in one place and
    avoid adding a dependency for a single baseline.
    """

    def __init__(self, n_states=3, n_iter=60, seed=SEED):
        self.K = n_states
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)

    def fit(self, x):
        x = np.asarray(x, dtype=float).ravel()
        T, K = len(x), self.K
        # init: quantile-spaced means, shared variance
        qs = np.linspace(10, 90, K)
        self.mu = np.percentile(x, qs)
        self.var = np.full(K, max(np.var(x), 1e-6))
        self.pi = np.full(K, 1.0 / K)
        self.A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(self.A, 0.9)

        for _ in range(self.n_iter):
            B = self._emission(x)                 # (T,K)
            alpha, c = self._forward(B)
            beta = self._backward(B, c)
            g = alpha * beta
            g /= np.clip(g.sum(1, keepdims=True), 1e-300, None)

            xi = np.zeros((K, K))
            for t in range(T - 1):
                m = (alpha[t][:, None] * self.A * B[t + 1][None, :] *
                     beta[t + 1][None, :])
                s = m.sum()
                if s > 0:
                    xi += m / s

            self.pi = np.clip(g[0], 1e-12, None); self.pi /= self.pi.sum()
            self.A = np.clip(xi, 1e-12, None)
            self.A /= self.A.sum(1, keepdims=True)
            w = g.sum(0)
            self.mu = (g * x[:, None]).sum(0) / np.clip(w, 1e-12, None)
            self.var = np.clip(
                (g * (x[:, None] - self.mu) ** 2).sum(0) / np.clip(w, 1e-12, None),
                1e-6, None)
        return self

    def _emission(self, x):
        d = x[:, None] - self.mu[None, :]
        return np.clip(np.exp(-0.5 * d ** 2 / self.var) /
                       np.sqrt(2 * np.pi * self.var), 1e-300, None)

    def _forward(self, B):
        T, K = B.shape
        alpha = np.zeros((T, K)); c = np.zeros(T)
        alpha[0] = self.pi * B[0]
        c[0] = alpha[0].sum(); alpha[0] /= max(c[0], 1e-300)
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            c[t] = alpha[t].sum(); alpha[t] /= max(c[t], 1e-300)
        return alpha, c

    def _backward(self, B, c):
        T, K = B.shape
        beta = np.zeros((T, K)); beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1])
            beta[t] /= max(c[t + 1], 1e-300)
        return beta

    def predict_proba(self, x):
        """Filtered state probabilities (causal: uses data up to t only)."""
        x = np.asarray(x, dtype=float).ravel()
        B = self._emission(x)
        alpha, _ = self._forward(B)
        return alpha / np.clip(alpha.sum(1, keepdims=True), 1e-300, None)


# ── CPTC, ported from the authors' algos/cptc.py ─────────────────────────────
def run_cptc(gt, preds, z_prob, z_mean, warm=100, alpha=ALPHA, gamma=GAMMA,
             max_width=None, min_residuals=MIN_RESIDUALS, seed=SEED):
    """
    Faithful port of conformal_prediction_cptc for the univariate case.
    gt, preds : (T,) ground truth and point predictions
    z_prob    : (T,K) per-timestep state probabilities
    z_mean    : (T,K) per-state predictive means
    warm      : number of leading points used only to seed the residual pools
    """
    rng = np.random.default_rng(seed)
    T = len(gt)
    K = z_prob.shape[1]
    states = list(range(K))
    if max_width is None:
        max_width = float(np.nanmax(np.abs(gt - preds))) if T else 1.0

    S_z = {z: [max_width] for z in states}
    alpha_z = {z: alpha for z in states}

    for t in range(min(warm, T)):
        for z, p in enumerate(z_prob[t]):
            if p > 0.3:
                S_z[z].append(abs(gt[t] - preds[t]))
    S_z["all"] = [abs(gt[t] - preds[t]) for t in range(min(warm, T))] + [max_width]

    covered, widths = [], []
    for t in range(min(warm, T), T):
        intervals = []
        for z in states:
            p_z = z_prob[t, z]
            if p_z == 0:
                continue
            res = S_z[z]
            if len(res) < min_residuals:
                res = S_z["all"]
            if len(res) == 0:
                res = np.concatenate([np.asarray(v) for v in S_z.values()])
            score = np.quantile(res, np.clip(1 - alpha_z[z], 0, 1))
            intervals.append((z_mean[t, z], score, p_z))

        # smallest set: take states in DECREASING probability until mass >= 1-alpha
        intervals.sort(key=lambda x: -x[2])
        chosen, run = [], 0.0
        for iv in intervals:
            if run < 1 - alpha:
                chosen.append(iv); run += iv[2]
            else:
                break
        if not chosen:
            chosen = intervals[:1]

        widths.append(max((2 * r for _, r, _ in chosen), default=0.0))

        y_t = gt[t]
        if not np.isfinite(y_t):
            covered.append(np.nan); continue
        hit = any((y_t >= m - r) and (y_t <= m + r) for m, r, _ in chosen)
        covered.append(1 if hit else 0)

        z_hat = rng.choice(states, p=z_prob[t] / z_prob[t].sum())
        alpha_z[z_hat] = float(np.clip(
            alpha_z[z_hat] + gamma * (alpha - int(not hit)), 0.01, 0.99))
        S_z[z_hat].append(abs(preds[t] - y_t))

    return np.array(covered, dtype=float), np.array(widths, dtype=float)


def summarize(domain, model, horizon, covered, widths, y_te, block):
    cov, w = coverage_and_width(covered, widths)
    wlo, whi = wilson_ci_from_indicator(covered)
    blo, bhi = block_bootstrap_ci(covered, block=block)
    yv = np.asarray(y_te, float); yv = yv[np.isfinite(yv)]
    spread = float(np.percentile(yv, 95) - np.percentile(yv, 5))
    return dict(domain=domain, model=model, horizon=horizon,
                strategy="cptc_hmm_state_input",
                n=int(np.isfinite(np.asarray(covered, float)).sum()),
                coverage=round(cov, 2), wilson_lo=round(wlo, 2),
                wilson_hi=round(whi, 2), boot_lo=round(blo, 2),
                boot_hi=round(bhi, 2), mean_width=round(w, 3),
                width_ratio=round(w / spread, 1) if spread > 0 else None)


def build_states(train_series, test_series, pred_test, n_states=3):
    """Fit the HMM on TRAINING data only; infer states causally on test."""
    hmm = GaussianHMM(n_states=n_states).fit(train_series)
    zp = hmm.predict_proba(test_series)
    order = np.argsort(hmm.mu)
    zp = zp[:, order]
    mu = hmm.mu[order]
    # Per-state predictive mean: the point forecast shifted toward each state's
    # level, which is the univariate analogue of REDSDS's per-state emission.
    base = np.asarray(pred_test, float)[:, None]
    z_mean = base + (mu[None, :] - mu.mean())
    return zp, z_mean


print("=" * 78)
print("TASK 9C — CPTC with an HMM state input (NOT the original REDSDS)")
print("=" * 78)
print("  PROTOCOL EXCEPTION: this fits one latent-state model per domain, on")
print("  training data only, solely to supply CPTC's required state input.")
print("  It is the only model fit anywhere in the Task 7/8/9 comparison.")

rows = []

# ── Recession ───────────────────────────────────────────────────────────────
print("\n" + "-" * 78)
print("RECESSION")
print("-" * 78)
import task_oof_and_probit as T  # noqa: E402
LABELS = ["Current", "1M", "3M", "6M"]
for h_idx, h in enumerate(LABELS):
    y_tr = T.y_train[:, h_idx]
    y_te = T.y_test[:, h_idx]
    p_te = T.preds_test[:, h_idx]
    tr_ok = y_tr[np.isfinite(y_tr)]
    zp, zm = build_states(tr_ok, np.nan_to_num(y_te, nan=float(np.nanmean(y_te))),
                          p_te)
    cov, wid = run_cptc(y_te, p_te, zp, zm, warm=10)
    r = summarize("Recession", "stacking-chain", h, cov, wid, y_te, block=12)
    rows.append(r)
    print(f"  {h:8s} n={r['n']:3d} cov={r['coverage']:6.2f}% "
          f"[{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}] w={r['mean_width']:.2f}")

# ── Healthcare ──────────────────────────────────────────────────────────────
print("\n" + "-" * 78)
print("HEALTHCARE")
print("-" * 78)
import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    y_te = g["Y_TEST"]; p_te = g["PREDS_TEST"][name]
    s = g["SCORES"][name]
    y_tr_proxy = s[np.isfinite(s)]
    zp, zm = build_states(y_tr_proxy, y_te, p_te)
    cov, wid = run_cptc(y_te, p_te, zp, zm, warm=20)
    r = summarize("Healthcare", name, "30-day", cov, wid, y_te, block=1)
    rows.append(r)
    print(f"  {name:10s} n={r['n']:3d} cov={r['coverage']:6.2f}% "
          f"[{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}] w={r['mean_width']:.2f}")

# ── Climate ────────────────────────────────────────────────────────────────
print("\n" + "-" * 78)
print("CLIMATE")
print("-" * 78)
gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    y_te = gc["Y_TEST"]; p_te = gc["PREDS_TEST"][name]
    s = gc["SCORES"][name]
    y_tr_proxy = s[np.isfinite(s)]
    zp, zm = build_states(y_tr_proxy, y_te, p_te)
    cov, wid = run_cptc(y_te, p_te, zp, zm, warm=30)
    r = summarize("Climate", name, "region-month", cov, wid, y_te, block=12)
    rows.append(r)
    print(f"  {name:10s} n={r['n']:3d} cov={r['coverage']:6.2f}% "
          f"[{r['wilson_lo']:.2f},{r['wilson_hi']:.2f}] w={r['mean_width']:.2f}")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/task9c_cptc.csv", index=False)

# ── Does it beat the diversity-optimal selector anywhere? ───────────────────
print("\n" + "=" * 78)
print("DOES CPTC (HMM variant) BEAT THE DIVERSITY-OPTIMAL SELECTOR?")
print("=" * 78)
comp = []
for f, dom in [("task7_baselines_recession.csv", "Recession"),
               ("task7_baselines_healthcare.csv", "Healthcare"),
               ("task7_baselines_climate.csv", "Climate")]:
    p = f"{OUT}/{f}"
    if not os.path.exists(p):
        continue
    d = pd.read_csv(p)
    if "scoring" in d and (d.scoring == "out-of-fold").any():
        d = d[d.scoring == "out-of-fold"]
    for _, r in d[d.strategy == "diversity_optimal"].iterrows():
        c = res[(res.domain == dom) & (res.model == r.model) &
                (res.horizon == r.horizon)]
        if c.empty:
            continue
        c = c.iloc[0]
        comp.append(dict(domain=dom, model=r.model, horizon=r.horizon,
                         divopt=r.coverage, cptc=c.coverage,
                         divopt_w=r.mean_width, cptc_w=c.mean_width,
                         cptc_beats=bool(c.wilson_lo > r.coverage)))
if comp:
    cdf = pd.DataFrame(comp)
    print(cdf.to_string(index=False))
    print(f"\n  Cases where CPTC separates above diversity-optimal: "
          f"{int(cdf['cptc_beats'].sum())} of {len(cdf)}")
print(f"\nSaved {OUT}/task9c_cptc.csv")
