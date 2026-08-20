"""
TASK 20, Item 3 — Falsification: do either signal actually predict misses?

Runs on the primary recession testbed's EXISTING out-of-fold results. No new
model runs: oof_pred / preds_test / y_test are imported unchanged from
task_oof_and_probit.py, exactly as task7_baseline_horse_race.py does.

Procedure, per horizon and per calibration strategy:
  1. Re-run the ACI loop step-by-step, recording at EACH test month t the state
     that existed BEFORE the outcome at t was known: alpha_t, q_t = 1-alpha_t,
     Q_C(q_t), Signal A margin, Signal B drift ratio.
  2. ONLY AFTERWARD, record whether month t's interval actually missed.
  3. Compare miss rate in the риск-flagged quartile of each signal against the
     base miss rate over the whole test period, separately for each signal.

NO-LEAKAGE ENFORCEMENT (the task's single hard constraint):
  - Signal A at t uses N and alpha_t only. alpha_t is ACI's state entering step
    t, updated from outcomes at steps < t. Never touches y_test[t].
  - Signal B at t uses resolved scores at steps < t strictly. The realized score
    at t is appended only AFTER the signals for t are recorded.
  - Q_G is never computed or referenced anywhere.
  - An explicit assertion verifies the ordering: signals for step t are frozen
    before miss[t] is read.

Outputs: tailsignal/task20_3_validation.csv, task20_3_permutation.csv
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
from task_oof_and_probit import (  # noqa: E402
    y_train, y_test, oof_pred, preds_test, LABELS, N_FIX,
)
import task20_1_signalA as SA  # noqa: E402
import task20_2_signalB as SB  # noqa: E402

OUT = "tailsignal"
os.makedirs(OUT, exist_ok=True)
K = SB.K_DEFAULT


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = stats.binomtest(int(k), int(n)).proportion_ci(confidence_level=0.95,
                                                          method="wilson")
    return (100 * lo, 100 * hi)


def trace_aci(y_te, pred_te, cal_scores, pool_scores, strategy, N):
    """
    Step-by-step ACI with per-step signal capture. Mirrors
    domain_common.run_aci exactly (same gamma/alpha/clip), but records the
    pre-outcome state at each step.
    """
    alpha_t = ALPHA_INIT
    resolved = []          # realized |error| at steps < t
    recs = []
    for t in range(len(y_te)):
        # ---- state BEFORE the outcome at t is known ----
        q_t = float(np.clip(1 - alpha_t, 0.0, 1.0))
        Q_C = float(np.quantile(cal_scores, q_t))

        dem = SA.demand(N, q_t)
        if strategy == "diversity_optimal":
            sup = SA.supply_alternating(N)
        else:
            sup = SA.supply_empirical(cal_scores, pool_scores)
        marginA = SA.margin(sup, dem)

        driftB_max = SB.drift_ratio(resolved, t, Q_C, K=K, stat="max")
        driftB_p90 = SB.drift_ratio(resolved, t, Q_C, K=K, stat="p90")

        frozen = dict(t=t, alpha_t=alpha_t, q_t=q_t, Q_C=Q_C,
                      demand=dem, supply=sup, marginA=marginA,
                      driftB_max=driftB_max, driftB_p90=driftB_p90)

        # ---- ONLY NOW is the outcome consulted ----
        y_t = y_te[t]
        if np.isnan(y_t):
            frozen["miss"] = np.nan
        else:
            realized = abs(pred_te[t] - y_t)
            miss = 1 if realized > Q_C else 0
            frozen["miss"] = miss
            alpha_t = float(np.clip(alpha_t + GAMMA_DEFAULT * (ALPHA_TARGET - miss),
                                    0.01, 0.99))
            resolved.append(realized)
        recs.append(frozen)
    return pd.DataFrame(recs)


def quartile_test(df, col, low_is_risky, label, horizon, strategy):
    """
    Miss rate in the risk-flagged quartile vs. base rate, with Wilson CIs and
    a Fisher exact test. low_is_risky=True for Signal A (low margin = strained),
    False for Signal B (high drift = risky).
    """
    d = df[df[col].notna() & df["miss"].notna()]
    n_all = len(d)
    if n_all < 8:
        return None
    base_miss = d["miss"].mean() * 100
    thresh = d[col].quantile(0.25 if low_is_risky else 0.75)
    flag = (d[col] <= thresh) if low_is_risky else (d[col] >= thresh)
    sub, rest = d[flag], d[~flag]
    if len(sub) < 3 or len(rest) < 3:
        return None
    k_sub, n_sub = int(sub["miss"].sum()), len(sub)
    k_rest, n_rest = int(rest["miss"].sum()), len(rest)
    lo_s, hi_s = wilson(k_sub, n_sub)
    p_fisher = stats.fisher_exact([[k_sub, n_sub - k_sub],
                                   [k_rest, n_rest - k_rest]],
                                  alternative="greater").pvalue
    rho = stats.spearmanr(d[col], d["miss"]).correlation if d["miss"].nunique() > 1 else np.nan
    return dict(horizon=horizon, strategy=strategy, signal=label,
                n_scored=n_all, base_miss_pct=round(base_miss, 2),
                n_flagged=n_sub, flagged_miss_pct=round(100 * k_sub / n_sub, 2),
                flagged_wilson_lo=round(lo_s, 2), flagged_wilson_hi=round(hi_s, 2),
                n_rest=n_rest, rest_miss_pct=round(100 * k_rest / n_rest, 2),
                lift_pp=round(100 * k_sub / n_sub - base_miss, 2),
                fisher_p=round(float(p_fisher), 4),
                spearman_rho=None if not np.isfinite(rho) else round(float(rho), 3))


print("=" * 100)
print("TASK 20 Item 3 — falsification: do Signal A / Signal B predict misses?")
print("=" * 100)
print(f"  Data: existing OOF results (task_oof_and_probit.py), N={N_FIX}, K={K}")
print("  Signals frozen BEFORE each month's outcome is read. Q_G never used.\n")

n_pool = oof_pred.shape[0]
rows, traces = [], []
for h_idx, h in enumerate(LABELS):
    s_oof = np.abs(oof_pred[:, h_idx] - y_train[:, h_idx])
    valid = np.where(np.isfinite(s_oof))[0]
    pool = s_oof[valid]

    strategies = {
        "pooled_trailing": s_oof[valid[-N_FIX:]],
        "diversity_optimal": s_oof[valid[SEL.support_width_selector(pool, N_FIX)]],
    }
    for sname, cal in strategies.items():
        tr = trace_aci(y_test[:, h_idx], preds_test[:, h_idx], cal, pool,
                       sname, N_FIX)
        tr["horizon"] = h
        tr["strategy"] = sname
        traces.append(tr)

        cov = 100 * (1 - tr["miss"].mean())
        print(f"  {h:8s} {sname:18s} coverage={cov:6.2f}%  "
              f"marginA range=[{tr.marginA.min():.2f},{tr.marginA.max():.2f}]  "
              f"driftB(max) median={tr.driftB_max.median():.3f}")

        for col, low_risky, lab in [("marginA", True, "A_margin"),
                                    ("driftB_max", False, "B_drift_max"),
                                    ("driftB_p90", False, "B_drift_p90")]:
            r = quartile_test(tr, col, low_risky, lab, h, sname)
            if r:
                rows.append(r)

T = pd.concat(traces, ignore_index=True)
T.to_csv(f"{OUT}/task20_3_traces.csv", index=False)
V = pd.DataFrame(rows)
V.to_csv(f"{OUT}/task20_3_validation.csv", index=False)

print("\n" + "=" * 100)
print("QUARTILE TEST — miss rate in risk-flagged quartile vs. base rate")
print("=" * 100)
print(f"{'horizon':8} {'strategy':18} {'signal':12} {'n':>4} {'base%':>7} "
      f"{'flag%':>7} {'lift_pp':>8} {'fisher_p':>9}")
for r in V.itertuples():
    print(f"{r.horizon:8} {r.strategy:18} {r.signal:12} {r.n_scored:4d} "
          f"{r.base_miss_pct:7.2f} {r.flagged_miss_pct:7.2f} "
          f"{r.lift_pp:+8.2f} {r.fisher_p:9.4f}")

print(f"\nSaved {OUT}/task20_3_validation.csv, {OUT}/task20_3_traces.csv")


# ── Multiplicity + a leakage-proof permutation null ────────────────────────
# 24 quartile tests were run. A single p=0.04 among 24 is what chance produces,
# so the nominal p-values above cannot be read individually. Two corrections:
#   (1) Benjamini-Hochberg FDR across the whole family.
#   (2) A CIRCULAR-SHIFT permutation null that preserves each signal's temporal
#       autocorrelation (a plain shuffle would destroy it and give an
#       optimistically narrow null for an autocorrelated signal).
print("\n" + "=" * 100)
print("MULTIPLICITY CORRECTION + CIRCULAR-SHIFT PERMUTATION NULL")
print("=" * 100)

_p = V["fisher_p"].values
_order = np.argsort(_p)
_m = len(_p)
_bh = np.empty(_m)
_prev = 1.0
for _rank in range(_m - 1, -1, -1):
    _i = _order[_rank]
    _prev = min(_prev, _p[_i] * _m / (_rank + 1))
    _bh[_i] = _prev
V["fisher_p_bh"] = np.round(_bh, 4)

rng = np.random.default_rng(20)
perm_rows = []
for r in V.itertuples():
    tr = T[(T.horizon == r.horizon) & (T.strategy == r.strategy)]
    col = {"A_margin": "marginA", "B_drift_max": "driftB_max",
           "B_drift_p90": "driftB_p90"}[r.signal]
    d = tr[tr[col].notna() & tr["miss"].notna()]
    if len(d) < 8 or d["miss"].nunique() < 2:
        perm_rows.append(dict(horizon=r.horizon, strategy=r.strategy,
                              signal=r.signal, observed_lift=r.lift_pp,
                              perm_p=None, note="degenerate (no miss variation)"))
        continue
    sig = d[col].values
    miss = d["miss"].values
    low_risky = (r.signal == "A_margin")
    n = len(sig)

    def lift_for(s):
        th = np.quantile(s, 0.25 if low_risky else 0.75)
        f = (s <= th) if low_risky else (s >= th)
        if f.sum() < 3 or (~f).sum() < 3:
            return np.nan
        return 100 * miss[f].mean() - 100 * miss.mean()

    obs = lift_for(sig)
    null = np.array([lift_for(np.roll(sig, int(k)))
                     for k in rng.integers(1, n, size=2000)])
    null = null[np.isfinite(null)]
    pp = float((null >= obs).mean()) if len(null) else np.nan
    perm_rows.append(dict(horizon=r.horizon, strategy=r.strategy,
                          signal=r.signal, observed_lift=round(float(obs), 2),
                          perm_p=round(pp, 4), note=""))

P = pd.DataFrame(perm_rows)
P.to_csv(f"{OUT}/task20_3_permutation.csv", index=False)
V.to_csv(f"{OUT}/task20_3_validation.csv", index=False)

print(f"{'horizon':8} {'strategy':18} {'signal':12} {'lift_pp':>8} "
      f"{'fisher_p':>9} {'BH_q':>7} {'perm_p':>8}")
for r, pr in zip(V.itertuples(), P.itertuples()):
    pp = "  n/a" if pr.perm_p is None else f"{pr.perm_p:8.4f}"
    print(f"{r.horizon:8} {r.strategy:18} {r.signal:12} {r.lift_pp:+8.2f} "
          f"{r.fisher_p:9.4f} {r.fisher_p_bh:7.4f} {pp}")

print(f"\n  Family size m={_m}. Minimum BH q-value = {V.fisher_p_bh.min():.4f}")
print(f"  Cells surviving BH at q<0.10: {int((V.fisher_p_bh < 0.10).sum())}")
print(f"\nSaved {OUT}/task20_3_permutation.csv")
