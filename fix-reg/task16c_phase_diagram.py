"""
TASK 16, Item C — synthetic phase diagram over frequency and score separation.

Section 5.2 reports seven synthetic scenario points and observes that the
diversity advantage narrows as event frequency or score magnitude rises. That is
inferred from seven points, not mapped. This sweeps a grid so the boundary can
be seen rather than extrapolated.

PARAMETERISATION. The spec asks for Delta_S = the difference in MEAN SCORE
between rare and normal draws, expressed relative to the normal-score standard
deviation. Phase 4a instead parameterised by a magnitude multiplier on the
scale. Delta_S is used here because it is the quantity the spec defines and is
directly interpretable, and because it lets the real domains be located on the
same axes (their measured Delta_S is computed from actual scores below).

Everything downstream reuses Phase 4a's machinery: the same half-normal score
construction, the same ACI-on-magnitudes runner, the same fixed-size draw
diagnostic. Only the sweep is new.

EXPLORATORY, per the spec's guardrail: the grid is reported as measured. No
parametric boundary is fitted to it and presented as a discovered law.

Outputs:
  fix-reg/task16c_phase_diagram.csv
  figures/task16c_phase_diagram.{pdf,png}
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
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
CLUSTER = 6

FREQS = [0.02, 0.05, 0.08, 0.12, 0.20, 0.30]

# The spec's suggested grid {0,1,2,4,8,16} was run first and SATURATES: beyond
# Delta_S ~ 2, 100% of rare scores already exceed the pool's 90th percentile, so
# pushing them further changes nothing and the columns 4/8/16 returned
# numerically identical gaps (verified directly). Those cells carry no
# information about the mechanism.
#
# The grid is therefore refined where the variation and the real domains both
# live (measured Delta_S = 1.26-2.97), while retaining 4 and 8 to *show* the
# saturation plateau rather than hide it.
DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0]


def make_series(n, rare_frac, delta_s, cluster_len, base_sd=1.0, seed=0):
    """
    Normal scores ~ |N(0, base_sd)|; rare scores shifted so their MEAN exceeds
    the normal mean by delta_s * base_sd. Rare months fall in clusters, matching
    Phase 4a's temporal structure.
    """
    rng = np.random.default_rng(seed)
    is_rare = np.zeros(n, bool)
    n_rare = int(round(rare_frac * n))
    if n_rare > 0:
        n_clusters = max(1, int(round(n_rare / cluster_len)))
        hi = max(1, n - cluster_len)
        starts = rng.choice(np.arange(0, hi),
                            size=min(n_clusters, hi), replace=False)
        for s in starts:
            is_rare[s:s + cluster_len] = True

    scores = np.abs(rng.normal(0, base_sd, n))
    k = int(is_rare.sum())
    if k:
        # half-normal mean is base_sd*sqrt(2/pi); shift rare scores by delta_s SDs
        shift = delta_s * base_sd
        scores[is_rare] = np.abs(rng.normal(0, base_sd, k)) + shift
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
    test_scores, _ = make_series(TEST, freq, delta, CLUSTER, seed=999)
    pool_scores, pool_rare = make_series(POOL, freq, delta, CLUSTER, seed=seed)
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
        return dict(freq=freq, delta_s=delta, n_draws=N_DRAWS,
                    rho_supp=np.nan, rho_rare=np.nan, gap=np.nan,
                    cov_mean=round(float(np.mean(ys)), 2), degenerate=True)
    rho_s = stats.spearmanr(xs, ys).correlation
    rho_r = (stats.spearmanr(rs, ys).correlation if np.std(rs) > 0 else np.nan)
    gap = (rho_s - rho_r) if np.isfinite(rho_r) else np.nan
    return dict(freq=freq, delta_s=delta, n_draws=N_DRAWS,
                rho_supp=round(float(rho_s), 3),
                rho_rare=None if not np.isfinite(rho_r) else round(float(rho_r), 3),
                gap=None if not np.isfinite(gap) else round(float(gap), 3),
                cov_mean=round(float(np.mean(ys)), 2), degenerate=False)


print("=" * 92)
print("TASK 16C — phase diagram: diversity advantage over frequency x separation")
print("=" * 92)
print(f"  grid: {len(FREQS)} frequencies x {len(DELTAS)} separations = "
      f"{len(FREQS)*len(DELTAS)} cells, {N_DRAWS} draws each")

rows = []
for f in FREQS:
    line = []
    for dl in DELTAS:
        r = run_cell(f, dl)
        rows.append(r)
        line.append("  n/a" if r["gap"] is None else f"{r['gap']:+.2f}")
    print(f"  freq={f:.2f}: " + "  ".join(line))

G = pd.DataFrame(rows)
G.to_csv(f"{OUT}/task16c_phase_diagram.csv", index=False)

# ── Where do the real domains actually sit? ─────────────────────────────────
print("\n" + "-" * 92)
print("Locating the real domains on these axes (measured from actual scores)")
print("-" * 92)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

real = []
try:
    import task_oof_and_probit as T  # noqa: E402
    for h_idx, h in [(2, "3M"), (3, "6M")]:
        s = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
        rare = (T.train_df["recession_probability"].values >= 50)
        m = np.isfinite(s)
        s, rare = s[m], rare[m]
        if rare.sum() < 3:
            continue
        sd = np.std(s[~rare])
        d = (np.mean(s[rare]) - np.mean(s[~rare])) / sd if sd > 0 else np.nan
        real.append(("Recession " + h, float(rare.mean()), float(d)))
except Exception as e:
    print(f"  [recession skipped] {e}")

import runpy  # noqa: E402
for path, nm, key in [("fix-reg/domain_healthcare.py", "Healthcare", "gradboost"),
                      ("fix-reg/domain_climate.py", "Climate", "gradboost")]:
    try:
        gg = runpy.run_path(path, run_name="_x")
        s = np.asarray(gg["SCORES"][key], float)
        rare = np.asarray(gg["RARE_POOL"], bool)
        m = np.isfinite(s)
        s, rare = s[m], rare[m]
        sd = np.std(s[~rare])
        d = (np.mean(s[rare]) - np.mean(s[~rare])) / sd if sd > 0 else np.nan
        real.append((nm, float(rare.mean()), float(d)))
    except Exception as e:
        print(f"  [{nm} skipped] {e}")

for nm, f, d in real:
    print(f"  {nm:16s} frequency={f*100:5.1f}%   Delta_S={d:+.2f} SDs")

pd.DataFrame(real, columns=["domain", "frequency", "delta_s"]).to_csv(
    f"{OUT}/task16c_real_domain_coords.csv", index=False)

# ── Figure ──────────────────────────────────────────────────────────────────
piv = G.pivot_table(index="freq", columns="delta_s", values="gap")
fig, ax = plt.subplots(figsize=(13.5, 6.4))
im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="RdYlBu_r",
               vmin=-0.2, vmax=max(0.4, np.nanmax(piv.values)),
               extent=[-0.5, len(DELTAS) - 0.5, -0.5, len(FREQS) - 0.5])
ax.set_xticks(range(len(DELTAS)))
ax.set_xticklabels([f"{d:g}" for d in DELTAS])
ax.set_yticks(range(len(FREQS)))
ax.set_yticklabels([f"{f*100:g}%" for f in FREQS])
ax.set_xlabel(r"score separation $\Delta_S$ (SDs between rare and normal means)",
              fontsize=10.5)
ax.set_ylabel("rare-event frequency", fontsize=10.5)

for i in range(len(FREQS)):
    for j in range(len(DELTAS)):
        v = piv.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.2,
                    color="black")

# overlay the real domains
for nm, f, d in real:
    xi = float(np.interp(d, DELTAS, np.arange(len(DELTAS))))
    yi = float(np.interp(f, FREQS, np.arange(len(FREQS))))
    ax.plot(xi, yi, "k*", ms=17, mec="white", mew=1.4, zorder=5)
    ax.annotate(nm, (xi, yi), textcoords="offset points", xytext=(11, 7),
                fontsize=9.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=.82,
                          ec="none"))

fig.colorbar(im, ax=ax, label=r"$\rho$(diversity) $-$ $\rho$(rare-count)")
ax.set_title("Where diversity dominates rare-event count\n"
             "(positive = diversity is the better predictor; stars = this "
             "paper's real domains)", fontsize=12, pad=11)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{FIG}/task16c_phase_diagram.{ext}", dpi=170,
                bbox_inches="tight")
print(f"\nSaved {FIG}/task16c_phase_diagram.pdf / .png")

print("\n" + "=" * 92)
print("GRID AS MEASURED (no fitted boundary — exploratory, per the guardrail)")
print("=" * 92)
print(piv.round(3).to_string())
valid = G[G["gap"].notna()]
print(f"\n  cells with a positive gap: {int((valid['gap'] > 0).sum())} "
      f"of {len(valid)} non-degenerate")
print(f"  degenerate cells (coverage constant): {int(G['degenerate'].sum())}")
print(f"\nSaved {OUT}/task16c_phase_diagram.csv")
