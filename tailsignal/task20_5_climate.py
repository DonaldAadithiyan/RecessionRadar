"""
TASK 20, Item 5 (follow-up) — Test Signal A on the CLIMATE domain.

Motivation: Item 3 found Signal A uninformative on recession because ACI's alpha
never travels near the q=0.5 boundary where the margin can discriminate. That is
a property of the testbed, not necessarily of the signal. If climate's alpha
genuinely approaches 0.5, climate becomes a real test.

This script answers that empirically rather than by assumption:
  STEP 1 — measure climate's actual alpha trajectory and the margin range it
           implies. If the margin never approaches 1, the test is structurally
           impossible here too, and that is reported as the finding.
  STEP 2 — run the Item 3 quartile validation regardless, so a result exists
           either way.

Reuses domain_climate.py's exported SCORES / PREDS_TEST / Y_TEST unchanged.
No new model runs. Same no-leakage discipline as Item 3.

Outputs: tailsignal/task20_5_climate.csv, task20_5_climate_alpha.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fix-reg"))
sys.path.insert(0, HERE)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import GAMMA_DEFAULT, ALPHA_INIT, ALPHA_TARGET  # noqa: E402
import selector_lib as SEL  # noqa: E402
import task20_1_signalA as SA  # noqa: E402
import task20_2_signalB as SB  # noqa: E402
from task20_3_validation import trace_aci, quartile_test  # noqa: E402

import domain_climate as DC  # noqa: E402

OUT = "tailsignal"
K = SB.K_DEFAULT

print("=" * 100)
print("TASK 20 Item 5 — Signal A on CLIMATE: does alpha reach the q=0.5 boundary?")
print("=" * 100)

y_test = DC.Y_TEST
rows, arows, traces = [], [], []

for model in ["ridge", "gradboost"]:
    scores_all = DC.SCORES[model]
    pred_test = DC.PREDS_TEST[model]
    valid = np.where(np.isfinite(scores_all))[0]
    pool = scores_all[valid]
    N = min(254, int(0.6 * len(valid)))

    strategies = {
        "pooled_trailing": pool[-N:],
        "diversity_optimal": pool[SEL.support_width_selector(pool, N)],
    }
    for sname, cal in strategies.items():
        tr = trace_aci(y_test, pred_test, cal, pool, sname, N)
        tr["horizon"] = "region-month"
        tr["strategy"] = f"{model}/{sname}"
        traces.append(tr)

        q = tr["q_t"].values
        m = tr["marginA"].values
        arows.append(dict(domain="Climate", model=model, strategy=sname, N=N,
                          alpha_min=round(float(tr.alpha_t.min()), 4),
                          alpha_max=round(float(tr.alpha_t.max()), 4),
                          q_min=round(float(q.min()), 4),
                          q_max=round(float(q.max()), 4),
                          margin_min=round(float(m.min()), 3),
                          margin_max=round(float(m.max()), 3),
                          dist_to_boundary=round(float(q.min() - 0.5), 4),
                          reaches_boundary=bool(m.min() <= 2.0),
                          n_test=int(len(tr)),
                          coverage=round(100 * (1 - tr["miss"].mean()), 2)))
        print(f"  {model:10s} {sname:18s} N={N:4d} n_test={len(tr):5d}  "
              f"alpha in [{tr.alpha_t.min():.4f},{tr.alpha_t.max():.4f}]  "
              f"q_min={q.min():.4f}  margin in [{m.min():.3f},{m.max():.3f}]  "
              f"cov={100*(1-tr['miss'].mean()):.2f}%")

        for col, low_risky, lab in [("marginA", True, "A_margin"),
                                    ("driftB_max", False, "B_drift_max"),
                                    ("driftB_p90", False, "B_drift_p90")]:
            r = quartile_test(tr, col, low_risky, lab, "region-month",
                              f"{model}/{sname}")
            if r:
                rows.append(r)

A = pd.DataFrame(arows)
A.to_csv(f"{OUT}/task20_5_climate_alpha.csv", index=False)
V = pd.DataFrame(rows)

# BH across this domain's own family
if len(V):
    p = V["fisher_p"].values
    order = np.argsort(p); mfam = len(p); bh = np.empty(mfam); prev = 1.0
    for rank in range(mfam - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * mfam / (rank + 1))
        bh[i] = prev
    V["fisher_p_bh"] = np.round(bh, 4)
V.to_csv(f"{OUT}/task20_5_climate.csv", index=False)

print("\n" + "=" * 100)
print("STEP 1 — Does climate's alpha approach the q=0.5 boundary?")
print("=" * 100)
print("  margin=1.000 requires q=0.500 (alpha=0.500); margin<2 requires q<=0.750 (alpha>=0.250)")
for r in A.itertuples():
    print(f"  {r.model:10s} {r.strategy:18s} closest approach: q_min={r.q_min:.4f} "
          f"(distance to 0.5 boundary = {r.dist_to_boundary:+.4f}), "
          f"min margin={r.margin_min:.3f}  -> "
          f"{'REACHES strained regime' if r.reaches_boundary else 'never strained'}")

print("\n" + "=" * 100)
print("STEP 2 — Quartile validation on climate (run regardless)")
print("=" * 100)
print(f"{'model/strategy':30} {'signal':12} {'n':>5} {'base%':>7} {'flag%':>7} "
      f"{'lift_pp':>8} {'fisher_p':>9} {'BH_q':>7}")
for r in V.itertuples():
    print(f"{r.strategy:30} {r.signal:12} {r.n_scored:5d} {r.base_miss_pct:7.2f} "
          f"{r.flagged_miss_pct:7.2f} {r.lift_pp:+8.2f} {r.fisher_p:9.4f} {r.fisher_p_bh:7.4f}")

print(f"\nSaved {OUT}/task20_5_climate.csv, {OUT}/task20_5_climate_alpha.csv")


# ── STEP 3 — permutation null + does margin discriminate WITHIN its own range? ──
print("\n" + "=" * 100)
print("STEP 3 — circular-shift permutation null (preserves autocorrelation)")
print("=" * 100)
T = pd.concat(traces, ignore_index=True)
T.to_csv(f"{OUT}/task20_5_climate_traces.csv", index=False)
rng = np.random.default_rng(20)
prows = []
for r in V.itertuples():
    tr = T[T.strategy == r.strategy]
    col = {"A_margin": "marginA", "B_drift_max": "driftB_max",
           "B_drift_p90": "driftB_p90"}[r.signal]
    d = tr[tr[col].notna() & tr["miss"].notna()]
    if len(d) < 8 or d["miss"].nunique() < 2:
        prows.append(dict(strategy=r.strategy, signal=r.signal,
                          observed_lift=r.lift_pp, perm_p=None))
        continue
    sig, miss = d[col].values, d["miss"].values
    low = (r.signal == "A_margin")
    n = len(sig)

    def lift_for(s):
        th = np.quantile(s, 0.25 if low else 0.75)
        f = (s <= th) if low else (s >= th)
        if f.sum() < 3 or (~f).sum() < 3:
            return np.nan
        return 100 * miss[f].mean() - 100 * miss.mean()

    obs = lift_for(sig)
    null = np.array([lift_for(np.roll(sig, int(k)))
                     for k in rng.integers(1, n, size=2000)])
    null = null[np.isfinite(null)]
    prows.append(dict(strategy=r.strategy, signal=r.signal,
                      observed_lift=round(float(obs), 2),
                      perm_p=round(float((null >= obs).mean()), 4)))
P = pd.DataFrame(prows)
P.to_csv(f"{OUT}/task20_5_climate_permutation.csv", index=False)
for r in P.itertuples():
    pp = "  n/a" if r.perm_p is None else f"{r.perm_p:8.4f}"
    print(f"  {r.strategy:30} {r.signal:12} lift={r.observed_lift:+7.2f}  perm_p={pp}")

# CONFOUND CHECK: on the pooled baseline, supply is EMPIRICAL (counts points
# above the pool median) and alpha_t moves with recent misses. So a low margin
# may simply be a proxy for "ACI recently raised alpha because it was missing" —
# i.e. autocorrelation of misses, not tail-reach diagnosis. Test directly.
print("\n" + "=" * 100)
print("STEP 4 — CONFOUND: is low margin just a proxy for 'recently missed'?")
print("=" * 100)
for sname in T.strategy.unique():
    d = T[(T.strategy == sname) & T["miss"].notna()].reset_index(drop=True)
    if d["miss"].nunique() < 2:
        continue
    prev = d["miss"].shift(1)
    ok = prev.notna()
    rho_prev = stats.spearmanr(d.marginA[ok], prev[ok]).correlation
    # partial: does margin still separate misses AFTER conditioning on prev-miss?
    sub = d[ok]
    parts = []
    for pv in (0.0, 1.0):
        g = sub[prev[ok] == pv]
        if len(g) < 20 or g["miss"].nunique() < 2:
            continue
        th = g.marginA.quantile(0.25)
        f = g.marginA <= th
        if f.sum() < 3 or (~f).sum() < 3:
            continue
        parts.append((pv, len(g), 100 * g["miss"][f].mean() - 100 * g["miss"].mean()))
    ptxt = "  ".join(f"prev_miss={int(pv)}: n={n} lift={lf:+.2f}pp" for pv, n, lf in parts)
    print(f"  {sname:30} rho(margin, prev_miss)={rho_prev:+.3f}   {ptxt if parts else '(insufficient)'}")
