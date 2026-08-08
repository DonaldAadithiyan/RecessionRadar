"""
TASK 17, Item 4 — a correlated-generator synthetic construction.

Task 16 Item C found the synthetic model predicts the WRONG SIGN at every real
domain's coordinates, and diagnosed the likely cause: rare-event status and
score magnitude were drawn as two INDEPENDENT components, which makes rare-count
artificially informative (knowing how many rare draws you took nearly determines
the upper tail — untrue of real scores).

This tests that diagnosis with one corrected generator, specified from the
mechanism rather than tuned:

    Z_t = rho * Z_{t-1} + eta_t                  latent severity, persistent
    P(R_t = 1 | Z_t) = sigmoid(a*Z_t + b)        event probability driven by Z
    S_t = mu(Z_t) + eps_t                        score magnitude ALSO driven by Z

Both the event indicator and the score now depend on a shared latent severity,
so rare-count and score magnitude are correlated through Z rather than yoked
together by construction. `a` controls the coupling strength; `b` is solved per
cell to hit the target prevalence; `rho` adds temporal persistence.

*** THIS ITEM IS EXPLICITLY ALLOWED TO FAIL. *** Per the spec: one generator
design, derived from the diagnosed cause, run once, reported honestly either
way. It is NOT iterated until it matches the real domains. If it still predicts
the wrong sign, the honest conclusion is that the mismatch runs deeper than the
independence artefact and this class of hand-specified synthetic isolation may
not be achievable at all.

Outputs:
  fix-reg/task17_4_correlated_synthetic.csv
  figures/task17_correlated_phase_diagram.{pdf,png}
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "fix-reg"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

POOL, TEST, N_FIX, N_DRAWS = 635, 65, 254, 200
GAMMA, ALPHA = 0.005, 0.10

# Same plane as Task 16C so the two are directly comparable.
FREQS = [0.02, 0.05, 0.08, 0.12, 0.20, 0.30]
DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]

# Generator parameters, fixed in advance (NOT tuned to the real domains)
A_COUPLING = 2.0      # how strongly latent severity drives event probability
RHO_PERSIST = 0.6     # temporal persistence of the latent severity


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def make_correlated(n, target_freq, delta_s, a=A_COUPLING, rho=RHO_PERSIST,
                    seed=0):
    """
    Latent-severity generator. Both the rare-event indicator and the score
    magnitude are driven by the same Z_t, so they are correlated rather than
    independently assigned.
    """
    rng = np.random.default_rng(seed)

    # persistent latent severity, standardized so `a` has a stable meaning
    z = np.zeros(n)
    eta = rng.normal(0, np.sqrt(max(1 - rho ** 2, 1e-6)), n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + eta[t]
    z = (z - z.mean()) / (z.std() + 1e-12)

    # solve the intercept b so E[P(R=1)] matches the target prevalence
    def prevalence(b):
        return sigmoid(a * z + b).mean() - target_freq
    try:
        b = brentq(prevalence, -50, 50)
    except ValueError:
        b = float(np.log(target_freq / (1 - target_freq)))
    p_event = sigmoid(a * z + b)
    is_rare = rng.random(n) < p_event

    # score magnitude driven by the SAME latent variable: mu(Z) = delta_s * Z
    # (so a high-severity month has both a higher event probability and a
    # larger expected score, which is the correlation the fix introduces)
    scores = np.abs(rng.normal(0, 1, n) + delta_s * z)
    return scores, is_rare


def aci_coverage(test_scores, cal_scores, gamma=GAMMA, alpha_t=ALPHA):
    covered, a = [], alpha_t
    for t in range(len(test_scores)):
        q = np.quantile(cal_scores, np.clip(1 - a, 0, 1))
        miss = 1 if test_scores[t] > q else 0
        covered.append(1 - miss)
        a = float(np.clip(a + gamma * (alpha_t - miss), 0.01, 0.99))
    return float(np.mean(covered) * 100)


def support(s):
    return float(np.percentile(s, 95) - np.percentile(s, 5))


def run_cell(freq, delta, seed=1):
    test_scores, _ = make_correlated(TEST, freq, delta, seed=999)
    pool_scores, pool_rare = make_correlated(POOL, freq, delta, seed=seed)
    rng = np.random.default_rng(2026 + seed)

    xs, ys, rs = [], [], []
    for _ in range(N_DRAWS):
        idx = rng.choice(POOL, size=N_FIX, replace=False)
        cs = pool_scores[idx]
        ys.append(aci_coverage(test_scores, cs))
        xs.append(support(cs))
        rs.append(int(pool_rare[idx].sum()))
    xs, ys, rs = np.array(xs), np.array(ys), np.array(rs, float)

    if np.std(ys) < 1e-9:
        return dict(freq=freq, delta_s=delta, rho_supp=np.nan,
                    rho_rare=np.nan, gap=np.nan, degenerate=True,
                    cov_mean=round(float(np.mean(ys)), 2),
                    realized_freq=round(float(pool_rare.mean()), 4))
    rho_s = stats.spearmanr(xs, ys).correlation
    rho_r = stats.spearmanr(rs, ys).correlation if np.std(rs) > 0 else np.nan
    gap = rho_s - rho_r if np.isfinite(rho_r) else np.nan
    return dict(freq=freq, delta_s=delta,
                rho_supp=round(float(rho_s), 3),
                rho_rare=None if not np.isfinite(rho_r) else round(float(rho_r), 3),
                gap=None if not np.isfinite(gap) else round(float(gap), 3),
                degenerate=False, cov_mean=round(float(np.mean(ys)), 2),
                realized_freq=round(float(pool_rare.mean()), 4))


print("=" * 96)
print("TASK 17 Item 4 — correlated-latent-severity generator")
print("=" * 96)
print(f"  Z_t = {RHO_PERSIST}*Z_(t-1) + eta   |   P(R|Z) = sigmoid({A_COUPLING}*Z + b)")
print(f"  S_t = |N(0,1) + Delta_S * Z|       (score driven by the SAME Z)")
print("  Parameters fixed in advance; the generator is NOT tuned to match the")
print("  real domains (spec guardrail).")

rows = []
for f in FREQS:
    line = []
    for d in DELTAS:
        r = run_cell(f, d)
        rows.append(r)
        line.append("  n/a" if r["gap"] is None else f"{r['gap']:+.2f}")
    print(f"  freq={f:.2f}: " + "  ".join(line))

G = pd.DataFrame(rows)
G.to_csv(f"{OUT}/task17_4_correlated_synthetic.csv", index=False)

# ── Real domain coordinates (reuse Task 16C's measurements) ────────────────
coord_path = f"{OUT}/task16c_real_domain_coords.csv"
real = []
if os.path.exists(coord_path):
    rc = pd.read_csv(coord_path)
    real = [(r["domain"], float(r["frequency"]), float(r["delta_s"]))
            for _, r in rc.iterrows()]
    print("\n  Real domain coordinates (from Task 16C):")
    for nm, f, d in real:
        print(f"    {nm:16s} freq={f*100:5.1f}%  Delta_S={d:+.2f}")

# ── Does the corrected generator get the sign right? ───────────────────────
REAL_GAPS = {"Recession 3M": 0.402, "Recession 6M": 0.442,
             "Healthcare": 0.122, "Climate": 0.062}

print("\n" + "=" * 96)
print("THE TEST: does the corrected generator predict the RIGHT SIGN at the")
print("real domains' coordinates? (all real gaps are POSITIVE)")
print("=" * 96)

piv = G.pivot_table(index="freq", columns="delta_s", values="gap")
check = []
for nm, f, d in real:
    fi = min(FREQS, key=lambda x: abs(x - f))
    di = min(DELTAS, key=lambda x: abs(x - d))
    pred = piv.loc[fi, di] if (fi in piv.index and di in piv.columns) else np.nan
    actual = REAL_GAPS.get(nm, np.nan)
    ok = bool(np.isfinite(pred) and np.isfinite(actual) and
              np.sign(pred) == np.sign(actual))
    check.append(dict(domain=nm, freq=f, delta_s=d,
                      nearest_cell=f"({fi},{di})",
                      predicted_gap=None if not np.isfinite(pred) else round(float(pred), 3),
                      actual_gap=actual, sign_correct=ok))
    print(f"  {nm:16s} predicted={pred:+.3f}  actual={actual:+.3f}  "
          f"sign {'CORRECT' if ok else 'WRONG'}")

C = pd.DataFrame(check)
C.to_csv(f"{OUT}/task17_4_sign_check.csv", index=False)
n_ok = int(C["sign_correct"].sum()) if len(C) else 0
print(f"\n  Sign correct at {n_ok} of {len(C)} real domains "
      f"(Task 16's independent generator: 0 of 4)")

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6.2))
im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="RdYlBu_r",
               vmin=-0.4, vmax=max(0.4, np.nanmax(piv.values)),
               extent=[-0.5, len(DELTAS) - 0.5, -0.5, len(FREQS) - 0.5])
ax.set_xticks(range(len(DELTAS)))
ax.set_xticklabels([f"{d:g}" for d in DELTAS])
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels([f"{f*100:g}%" for f in FREQS])
ax.set_xlabel(r"score separation $\Delta_S$ (latent-severity coupling)",
              fontsize=10.5)
ax.set_ylabel("rare-event prevalence", fontsize=10.5)
for i in range(len(FREQS)):
    for j in range(len(DELTAS)):
        v = piv.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.4)
for nm, f, d in real:
    xi = float(np.interp(d, DELTAS, np.arange(len(DELTAS))))
    yi = float(np.interp(f, FREQS, np.arange(len(FREQS))))
    ax.plot(xi, yi, "k*", ms=17, mec="white", mew=1.4, zorder=5)
    ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=(11, 7),
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=.82,
                          ec="none"))
fig.colorbar(im, ax=ax, label=r"$\rho$(diversity) $-$ $\rho$(rare-count)")
ax.set_title("Correlated latent-severity generator\n"
             "(both event probability and score magnitude driven by the same "
             "$Z_t$; stars = real domains)", fontsize=12, pad=11)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/task17_correlated_phase_diagram.{ext}", dpi=170,
                bbox_inches="tight")

print(f"\nSaved {FIG}/task17_correlated_phase_diagram.pdf")
print(f"Saved {OUT}/task17_4_correlated_synthetic.csv, task17_4_sign_check.csv")
