"""
TASK 18, Item 8 — synthetic generator magnitude adjustment.

Task 17 Item 4's correlated latent-severity generator gets the SIGN right at all
four real domains but over-predicts the gap's MAGNITUDE 3-7x in healthcare and
climate. This tries two structurally motivated modifications to that same
generator — not a new generator, and explicitly NOT a parameter search.

BASELINE (Task 17 Item 4):
    Z_t = rho*Z_{t-1} + eta_t
    P(R_t=1 | Z_t) = sigmoid(a*Z_t + b)
    S_t = |N(0,1) + Delta_S * Z_t|              constant-variance noise

MOD A — heteroskedastic noise.
    S_t = |eps_t * sigma(Z_t) + Delta_S * Z_t|,  sigma(Z) = exp(0.5 * Z)
    Rationale (structural, stated before running): real forecast errors plausibly
    become noisier, not merely larger, in severe periods. exp(0.5*Z) gives a
    ~1.65x noise-scale ratio per unit of latent severity, which is a modest,
    non-tuned choice — the coefficient 0.5 is half the unit scale of Z, not a
    value selected to match any domain.

MOD B — state-dependent coupling.
    a_t varies with a slow-moving state rather than being fixed:
    a_t = a * (1 + 0.5*sin(2*pi*t/120)), a 10-year cycle over the 635-month pool.
    Rationale: the strength of the link between severity and event labelling is
    itself regime-dependent (recording standards, policy thresholds, and
    diagnostic criteria all drift). The 120-month period is the business-cycle
    scale, chosen structurally; the 0.5 amplitude keeps a_t positive throughout.

GUARDRAIL (spec): one motivated parameter choice per modification, run once,
reported honestly. If neither narrows the magnitude gap, that is a valid and
informative outcome — it would mean the gap reflects something the generator's
basic structure cannot capture. NO iterating until the numbers match.

Outputs:
  fix-reg/task18_8_generator_magnitude.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "fix-reg"
POOL, TEST, N_FIX, N_DRAWS = 635, 65, 254, 200
GAMMA, ALPHA = 0.005, 0.10
A_COUPLING, RHO_PERSIST = 2.0, 0.6

# Real domain coordinates and measured gaps (from Tasks 16C / 16A)
REAL = [("Recession 3M", 0.0844, 2.194, 0.402),
        ("Recession 6M", 0.0844, 1.257, 0.442),
        ("Healthcare", 0.1027, 2.409, 0.122),
        ("Climate", 0.1071, 2.975, 0.062)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def latent(n, rho, seed):
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    eta = rng.normal(0, np.sqrt(max(1 - rho ** 2, 1e-6)), n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + eta[t]
    return (z - z.mean()) / (z.std() + 1e-12), rng


def make(n, freq, delta, variant, seed=0):
    z, rng = latent(n, RHO_PERSIST, seed)

    if variant == "state_dependent_a":
        t = np.arange(n)
        a_t = A_COUPLING * (1 + 0.5 * np.sin(2 * np.pi * t / 120.0))
    else:
        a_t = np.full(n, A_COUPLING)

    def prev(b):
        return sigmoid(a_t * z + b).mean() - freq
    try:
        b = brentq(prev, -50, 50)
    except ValueError:
        b = float(np.log(freq / (1 - freq)))
    is_rare = rng.random(n) < sigmoid(a_t * z + b)

    eps = rng.normal(0, 1, n)
    if variant == "heteroskedastic":
        sigma_z = np.exp(0.5 * z)          # structural, not tuned
        scores = np.abs(eps * sigma_z + delta * z)
    else:
        scores = np.abs(eps + delta * z)
    return scores, is_rare


def aci_cov(test_scores, cal, gamma=GAMMA, at=ALPHA):
    covered, a = [], at
    for t in range(len(test_scores)):
        q = np.quantile(cal, np.clip(1 - a, 0, 1))
        miss = 1 if test_scores[t] > q else 0
        covered.append(1 - miss)
        a = float(np.clip(a + gamma * (at - miss), 0.01, 0.99))
    return float(np.mean(covered) * 100)


def support(s):
    return float(np.percentile(s, 95) - np.percentile(s, 5))


def gap_at(freq, delta, variant, seed=1):
    ts, _ = make(TEST, freq, delta, variant, seed=999)
    ps, pr = make(POOL, freq, delta, variant, seed=seed)
    rng = np.random.default_rng(2026 + seed)
    xs, ys, rs = [], [], []
    for _ in range(N_DRAWS):
        idx = rng.choice(POOL, size=N_FIX, replace=False)
        cs = ps[idx]
        ys.append(aci_cov(ts, cs))
        xs.append(support(cs))
        rs.append(int(pr[idx].sum()))
    xs, ys, rs = np.array(xs), np.array(ys), np.array(rs, float)
    if np.std(ys) < 1e-9 or np.std(rs) == 0:
        return np.nan, np.nan, np.nan
    rho_s = stats.spearmanr(xs, ys).correlation
    rho_r = stats.spearmanr(rs, ys).correlation
    return rho_s - rho_r, rho_s, rho_r


print("=" * 96)
print("TASK 18 Item 8 — can a structural modification close the magnitude gap?")
print("=" * 96)
print("  MOD A heteroskedastic : sigma(Z) = exp(0.5*Z)   (noisier in severe periods)")
print("  MOD B state-dependent : a_t = a*(1 + 0.5*sin(2*pi*t/120))  (10-yr cycle)")
print("  Both parameter choices are structural and fixed BEFORE running.")

VARIANTS = ["baseline", "heteroskedastic", "state_dependent_a"]
rows = []
for nm, f, d, actual in REAL:
    line = {}
    for v in VARIANTS:
        g, rs, rr = gap_at(f, d, v)
        rows.append(dict(domain=nm, variant=v, freq=f, delta_s=d,
                         predicted_gap=None if not np.isfinite(g) else round(float(g), 3),
                         rho_supp=None if not np.isfinite(rs) else round(float(rs), 3),
                         rho_rare=None if not np.isfinite(rr) else round(float(rr), 3),
                         actual_gap=actual,
                         sign_correct=bool(np.isfinite(g) and
                                           np.sign(g) == np.sign(actual)),
                         abs_error=None if not np.isfinite(g) else round(abs(g - actual), 3),
                         ratio_to_actual=None if not np.isfinite(g) or actual == 0
                         else round(g / actual, 2)))
        line[v] = g
    print(f"\n  {nm:16s} actual={actual:+.3f}")
    for v in VARIANTS:
        r = [x for x in rows if x["domain"] == nm and x["variant"] == v][0]
        print(f"    {v:20s} predicted={r['predicted_gap']:+.3f}  "
              f"|err|={r['abs_error']:.3f}  ratio={r['ratio_to_actual']}x")

R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/task18_8_generator_magnitude.csv", index=False)

print("\n" + "=" * 96)
print("VERDICT — does either modification narrow the magnitude gap?")
print("=" * 96)
for v in VARIANTS:
    sub = R[R["variant"] == v]
    print(f"  {v:20s} mean |error| = {sub['abs_error'].mean():.3f}  "
          f"median ratio = {sub['ratio_to_actual'].median():.2f}x  "
          f"signs correct = {int(sub['sign_correct'].sum())}/{len(sub)}")

base_err = R[R["variant"] == "baseline"]["abs_error"].mean()
improved = [v for v in VARIANTS[1:]
            if R[R["variant"] == v]["abs_error"].mean() < base_err]
print(f"\n  Modifications improving on the baseline: "
      f"{improved if improved else 'NONE'}")
print(f"\nSaved {OUT}/task18_8_generator_magnitude.csv")
