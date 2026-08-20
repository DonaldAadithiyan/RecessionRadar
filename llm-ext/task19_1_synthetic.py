"""
TASK 19, Item 1 — synthetic mechanism test for LLM-style nonconformity scores.

Mirrors the paper's Section 3.5 (Task 16C / Task 17 Item 4) exactly, but with
score semantics restyled for an LLM setting. Two generators:

  (1) INDEPENDENT-DRAW  — the paper's original, later-falsified construction,
      functional form copied unchanged from task16c_phase_diagram.py. Hard-flag
      placement (clustered) and score magnitude are INDEPENDENT draws, linked
      only by a constant +Delta_S shift on flagged examples.

  (2) CORRELATED-SEVERITY — the paper's corrected construction. A persistent
      latent difficulty Z_t drives BOTH the hard-flag probability and the score
      magnitude:
          Z_t = rho*Z_{t-1} + eta_t
          P(H_t = 1 | Z_t) = sigmoid(a*Z_t + b)     b solved for target prevalence
          S_t = |N(0,1) + Delta_S * Z_t|

Both are run through the paper's own three-part diagnostic:
  (A) fixed-size ablation      — force a fixed hard-count into a fixed-N cal set
  (B) support-width vs count R^2 across many random calibration draws
  (C) within-support-tertile redundancy check on hard-count

*** THIS ITEM IS A GATE AND IS EXPLICITLY ALLOWED TO FAIL. ***
No parameter here is tuned against a target. Every value is fixed below from a
stated rationale before running. If the independent-draw generator is the one
that matches, or neither does, that is the finding and Items 2-4 do not run.

Outputs: llm-ext/task19_1_synthetic.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

OUT = "llm-ext"
os.makedirs(OUT, exist_ok=True)

# ── Parameters, FIXED IN ADVANCE (rationale stated; none tuned to a target) ──
# Sizes mirror the paper's synthetic plane (POOL/TEST/N_FIX) so the two are
# directly comparable; only the domain semantics change, not the geometry.
POOL, TEST, N_FIX, N_DRAWS = 635, 65, 254, 200
GAMMA, ALPHA = 0.005, 0.10          # identical ACI settings to Task 17 Item 4
CLUSTER = 6                         # carried over unchanged from Task 16C
A_COUPLING = 2.0                    # carried over unchanged from Task 17 Item 4
RHO_PERSIST = 0.6                   # carried over unchanged from Task 17 Item 4
SEED_TEST, SEED_POOL, SEED_DRAW = 999, 1, 2026   # same seed convention

# Sweep plane. Prevalence range brackets plausible LLM hard-example rates
# (2%-30% of an eval set being genuinely hard); Delta_S is the severity-coupling
# strength, swept over the same grid the paper used.
FREQS = [0.02, 0.05, 0.08, 0.12, 0.20, 0.30]
DELTAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def make_independent(n, target_freq, delta_s, seed=0, cluster_len=CLUSTER,
                     base_sd=1.0):
    """
    Generator 1 — the paper's ORIGINAL, falsified construction, functional form
    copied unchanged from task16c_phase_diagram.py's make_series().

    Hard-flag placement and score magnitude are INDEPENDENT draws: the flag is
    assigned by clustered position, and flagged examples get a deterministic
    +delta_s shift on an otherwise independent |N(0,base_sd)| draw. Nothing
    links *which* example is hard to *how large* its score is beyond that
    constant shift — the independence artefact the paper diagnosed.
    """
    rng = np.random.default_rng(seed)
    is_rare = np.zeros(n, bool)
    n_rare = int(round(target_freq * n))
    if n_rare > 0:
        n_clusters = max(1, int(round(n_rare / cluster_len)))
        hi = max(1, n - cluster_len)
        starts = rng.choice(np.arange(0, hi), size=min(n_clusters, hi),
                            replace=False)
        for s in starts:
            is_rare[s:s + cluster_len] = True

    scores = np.abs(rng.normal(0, base_sd, n))
    k = int(is_rare.sum())
    if k:
        scores[is_rare] = np.abs(rng.normal(0, base_sd, k)) + delta_s * base_sd
    return scores, is_rare


def make_correlated(n, target_freq, delta_s, a=A_COUPLING, rho=RHO_PERSIST,
                    seed=0):
    """
    Generator 2 — the paper's CORRECTED construction, functional form unchanged
    from task17_4_correlated_synthetic.py. Shared latent Z drives both channels.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros(n)
    eta = rng.normal(0, np.sqrt(max(1 - rho ** 2, 1e-6)), n)
    for t in range(1, n):
        z[t] = rho * z[t - 1] + eta[t]
    z = (z - z.mean()) / (z.std() + 1e-12)

    def prevalence(b):
        return sigmoid(a * z + b).mean() - target_freq
    try:
        b = brentq(prevalence, -50, 50)
    except ValueError:
        b = float(np.log(target_freq / (1 - target_freq)))
    is_rare = rng.random(n) < sigmoid(a * z + b)
    scores = np.abs(rng.normal(0, 1, n) + delta_s * z)
    return scores, is_rare


GENERATORS = {"independent": make_independent, "correlated": make_correlated}


def aci_coverage(test_scores, cal_scores, gamma=GAMMA, alpha_t=ALPHA):
    """Identical ACI loop to Task 17 Item 4."""
    covered, a = [], alpha_t
    for t in range(len(test_scores)):
        q = np.quantile(cal_scores, np.clip(1 - a, 0, 1))
        miss = 1 if test_scores[t] > q else 0
        covered.append(1 - miss)
        a = float(np.clip(a + gamma * (alpha_t - miss), 0.01, 0.99))
    return float(np.mean(covered) * 100)


def support(s):
    return float(np.percentile(s, 95) - np.percentile(s, 5))


def run_cell(gen_name, freq, delta):
    """Runs diagnostics (B) and (C) for one (generator, freq, delta) cell."""
    gen = GENERATORS[gen_name]
    test_scores, _ = gen(TEST, freq, delta, seed=SEED_TEST)
    pool_scores, pool_rare = gen(POOL, freq, delta, seed=SEED_POOL)
    rng = np.random.default_rng(SEED_DRAW + SEED_POOL)

    xs, ys, rs = [], [], []
    for _ in range(N_DRAWS):
        idx = rng.choice(POOL, size=N_FIX, replace=False)
        cs = pool_scores[idx]
        ys.append(aci_coverage(test_scores, cs))
        xs.append(support(cs))
        rs.append(int(pool_rare[idx].sum()))
    xs, ys, rs = np.array(xs), np.array(ys), np.array(rs, float)

    base = dict(generator=gen_name, freq=freq, delta_s=delta,
                cov_mean=round(float(np.mean(ys)), 2),
                cov_sd=round(float(np.std(ys)), 3),
                realized_freq=round(float(pool_rare.mean()), 4))

    if np.std(ys) < 1e-9 or np.std(xs) < 1e-9:
        base.update(rho_supp=None, rho_rare=None, gap=None,
                    r2_supp=None, r2_rare=None, r2_gap=None,
                    within_tert_rho=None, degenerate=True)
        return base

    rho_s = stats.spearmanr(xs, ys).correlation
    rho_r = stats.spearmanr(rs, ys).correlation if np.std(rs) > 0 else np.nan
    # (B) Pearson R^2, support width vs hard-count, exactly as phase1 reports it
    r2_s = stats.pearsonr(xs, ys).statistic ** 2
    r2_r = (stats.pearsonr(rs, ys).statistic ** 2) if np.std(rs) > 0 else np.nan

    # (C) within-support-tertile redundancy of hard-count
    terts = np.quantile(xs, [0, 1 / 3, 2 / 3, 1.0])
    within = []
    for ti in range(3):
        lo, hi = terts[ti], terts[ti + 1]
        m = (xs >= lo) & (xs <= hi if ti == 2 else xs < hi)
        if m.sum() < 6 or np.std(rs[m]) == 0 or np.std(ys[m]) == 0:
            continue
        within.append(stats.spearmanr(rs[m], ys[m]).correlation)
    within_mean = float(np.nanmean(within)) if within else np.nan

    def rnd(v):
        return None if not np.isfinite(v) else round(float(v), 3)

    base.update(rho_supp=rnd(rho_s), rho_rare=rnd(rho_r),
                gap=rnd(rho_s - rho_r) if np.isfinite(rho_r) else None,
                r2_supp=rnd(r2_s), r2_rare=rnd(r2_r),
                r2_gap=rnd(r2_s - r2_r) if np.isfinite(r2_r) else None,
                within_tert_rho=rnd(within_mean), degenerate=False)
    return base


def fixed_size_ablation(gen_name, freq, delta, rare_k_list=(0, 4, 8, 16, 32)):
    """
    (A) Fixed-size ablation: hold N_FIX constant, force the calibration set to
    contain exactly k hard examples. If hard-COUNT drives coverage, coverage
    should climb with k. If SUPPORT WIDTH drives it, coverage should track the
    resulting support width instead.
    """
    gen = GENERATORS[gen_name]
    test_scores, _ = gen(TEST, freq, delta, seed=SEED_TEST)
    pool_scores, pool_rare = gen(POOL, freq, delta, seed=SEED_POOL)
    rare_idx = np.where(pool_rare)[0]
    easy_idx = np.where(~pool_rare)[0]
    rng = np.random.default_rng(SEED_DRAW)
    rows = []
    for k in rare_k_list:
        if k > len(rare_idx) or (N_FIX - k) > len(easy_idx):
            continue
        covs, supps = [], []
        for _ in range(30):
            sel = np.concatenate([rng.choice(rare_idx, k, replace=False),
                                  rng.choice(easy_idx, N_FIX - k, replace=False)])
            cs = pool_scores[sel]
            covs.append(aci_coverage(test_scores, cs))
            supps.append(support(cs))
        rows.append(dict(generator=gen_name, freq=freq, delta_s=delta, rare_k=k,
                         cov_mean=round(float(np.mean(covs)), 2),
                         supp_mean=round(float(np.mean(supps)), 3)))
    return rows


print("=" * 96)
print("TASK 19 Item 1 — synthetic mechanism test, LLM-style nonconformity scores")
print("=" * 96)
print(f"  independent : H clustered (len {CLUSTER}); S=|N(0,1)|, flagged shifted +Delta_S  (INDEPENDENT draws)")
print(f"  correlated  : Z_t = {RHO_PERSIST}*Z_(t-1)+eta ; P(H|Z)=sigmoid({A_COUPLING}*Z+b) ;"
      f" S = |N(0,1)+Delta_S*Z|")
print("  All parameters fixed in advance. This item is a GATE and may fail.\n")

rows = []
for gname in GENERATORS:
    print(f"  --- {gname} generator: gap = rho(support,cov) - rho(hard_count,cov) ---")
    for f in FREQS:
        line = []
        for d in DELTAS:
            r = run_cell(gname, f, d)
            rows.append(r)
            line.append("  n/a" if r["gap"] is None else f"{r['gap']:+.2f}")
        print(f"    freq={f:.2f}: " + "  ".join(line))
    print()

G = pd.DataFrame(rows)
G.to_csv(f"{OUT}/task19_1_synthetic.csv", index=False)
print(f"Saved {OUT}/task19_1_synthetic.csv  ({len(G)} cells)")

# ── Fixed-size ablation at a mid-plane operating point, both generators ──────
# Operating point chosen BEFORE running: freq=0.12, Delta_S=1.0 is the centre of
# the swept plane, not selected for its result.
ABL_FREQ, ABL_DELTA = 0.12, 1.0
abl = []
for gname in GENERATORS:
    abl.extend(fixed_size_ablation(gname, ABL_FREQ, ABL_DELTA))
A = pd.DataFrame(abl)
A.to_csv(f"{OUT}/task19_1_ablation.csv", index=False)
print(f"\n(A) Fixed-size ablation at freq={ABL_FREQ}, Delta_S={ABL_DELTA} (N={N_FIX} held constant):")
for gname in GENERATORS:
    sub = A[A.generator == gname]
    print(f"  {gname:12s} " + "  ".join(
        f"k={int(r.rare_k)}:cov={r.cov_mean:.1f}/supp={r.supp_mean:.2f}"
        for r in sub.itertuples()))

# ── Verdict: does either generator reproduce "support beats count"? ─────────
print("\n" + "=" * 96)
print("VERDICT — fraction of non-degenerate cells where support width beats hard-count")
print("=" * 96)
summary = []
for gname in GENERATORS:
    sub = G[(G.generator == gname) & (~G.degenerate)]
    sub = sub[sub["gap"].notna()]
    n = len(sub)
    pos = int((sub["gap"] > 0).sum())
    r2pos = int((sub["r2_gap"] > 0).sum())
    med_gap = float(sub["gap"].median()) if n else float("nan")
    med_r2gap = float(sub["r2_gap"].median()) if n else float("nan")
    med_within = float(sub["within_tert_rho"].median()) if n else float("nan")
    print(f"  {gname:12s} n={n:3d}  gap>0: {pos}/{n} ({100*pos/max(n,1):.0f}%)  "
          f"median gap={med_gap:+.3f}  |  R2 gap>0: {r2pos}/{n}  "
          f"median R2 gap={med_r2gap:+.3f}  |  median within-tertile rho(count,cov)={med_within:+.3f}")
    summary.append(dict(generator=gname, n_cells=n, frac_gap_pos=round(pos / max(n, 1), 3),
                        median_gap=round(med_gap, 3), frac_r2gap_pos=round(r2pos / max(n, 1), 3),
                        median_r2_gap=round(med_r2gap, 3),
                        median_within_tertile_rho=round(med_within, 3)))
pd.DataFrame(summary).to_csv(f"{OUT}/task19_1_verdict.csv", index=False)
print(f"\nSaved {OUT}/task19_1_verdict.csv")
