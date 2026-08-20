"""
TASK 22, Item 4 — re-run Task 21 Item 3's comparison at 6M with Items 1 and 2
applied (rolling recentering + fit-derived ceiling).

Same baselines, same scoring rule, same statistical discipline as Task 21:
BH correction across the family, circular-shift and sign-flip permutation nulls
respecting temporal autocorrelation, and the WIN-RATE reported alongside the
mean (the check that caught Task 21's 40.7% confound).

Ablation arms included so the contribution of each fix is visible separately:
  task21_original     frozen CDF   + HI=1.25   (Task 21's method, reproduced)
  fix1_only           rolling CDF  + HI=1.25
  fix2_only           frozen CDF   + HI=2.79
  task22_both         rolling CDF  + HI=2.79   (the method under test)

Outputs: errordir/task22_4_comparison.csv, task22_4_significance.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg"))
sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from domain_common import GAMMA_DEFAULT, ALPHA_TARGET  # noqa: E402
import selector_lib as SEL  # noqa: E402
import baselines_lib as B  # noqa: E402
import task21_2_scaling as SC21  # noqa: E402
import task22_1_recentering as R  # noqa: E402
import task22_2_ceiling as CEIL  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_test, preds_test, test_df,
)

OUT = "errordir"
H_IDX, H = 3, "6M"
LO = 0.75

scaler, beta, fit_ok, cal_ok, abs_err = R.fit_beta()
X_train, X_test = X_train_df.values, X_test_df.values
proj_cal = scaler.transform(X_train[cal_ok]) @ beta
proj_test = scaler.transform(X_test) @ beta
cal_scores = abs_err[cal_ok]

u_static = R.static_ranks(proj_cal, proj_test)
u_roll = R.rolling_ranks(proj_cal, proj_test)

y_h, p_h = y_test[:, H_IDX], preds_test[:, H_IDX]


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def run(cs, mult=None):
    a = ALPHA_TARGET
    los, his, cov = [], [], []
    for t in range(len(y_h)):
        q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
        if mult is not None:
            q *= float(mult[t])
        lo, hi = p_h[t] - q, p_h[t] + q
        los.append(lo); his.append(hi)
        yt = y_h[t]
        if np.isnan(yt):
            cov.append(np.nan); continue
        miss = 1 if (yt < lo or yt > hi) else 0
        cov.append(1 - miss)
        a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), 0.01, 0.99))
    return np.array(los), np.array(his), np.array(cov)


def summ(name, los, his, cov, mult=None):
    ok = ~np.isnan(cov)
    wk = np.array([winkler(y_h[t], los[t], his[t]) for t in range(len(y_h)) if ok[t]])
    return dict(method=name, horizon=H, n=int(ok.sum()),
                coverage=round(100 * float(np.nanmean(cov[ok])), 2),
                mean_width=round(float(np.mean(his - los)), 3),
                winkler_mean=round(float(np.mean(wk)), 3),
                winkler_median=round(float(np.median(wk)), 3),
                mean_multiplier=None if mult is None else round(float(mult.mean()), 4)), wk


rows, wk_store = [], {}

sel_idx = SEL.support_width_selector(cal_scores, min(254, len(cal_scores)))
cal_sel = cal_scores[sel_idx]

l, hh, c = run(cal_sel); r, wk = summ("diversity_optimal", l, hh, c); rows.append(r); wk_store[r["method"]] = wk
cov_d, wid_d = B.run_dtaci(y_h, p_h, cal_scores)
r, wk = summ("dtaci", p_h - wid_d / 2, p_h + wid_d / 2, cov_d); rows.append(r); wk_store[r["method"]] = wk

ARMS = [
    ("task21_original", u_static, 1.25),
    ("fix1_only",       u_roll,   1.25),
    ("fix2_only",       u_static, CEIL.HI_NEW),
    ("task22_both",     u_roll,   CEIL.HI_NEW),
]
mults = {}
for nm, u, hi in ARMS:
    m = LO + (hi - LO) * np.clip(u, 0, 1)
    mults[nm] = m
    l, hh, c = run(cal_scores, mult=m)
    r, wk = summ(nm, l, hh, c, m); rows.append(r); wk_store[nm] = wk

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task22_4_comparison.csv", index=False)

print("=" * 104)
print(f"TASK 22 Item 4 — comparison at {H} with fixes applied")
print("=" * 104)
print(f"  ceiling HI: 1.25 (old) -> {CEIL.HI_NEW:.4f} (fit-derived)")
print(f"  mean multiplier — Task21: {mults['task21_original'].mean():.4f}  "
      f"Task22: {mults['task22_both'].mean():.4f}\n")
print(D.to_string(index=False))

print("\n" + "=" * 104)
print("SIGNIFICANCE — paired Winkler differences (negative = method better)")
print("=" * 104)
rng = np.random.default_rng(22)
srows = []
for method in ["task22_both", "fix1_only", "fix2_only"]:
    for base in ["diversity_optimal", "dtaci"]:
        a, b = wk_store[method], wk_store[base]
        n = min(len(a), len(b))
        d = a[:n] - b[:n]
        obs = float(np.mean(d))
        winrate = 100 * float((d < 0).mean())
        null = np.array([float(np.mean(np.roll(d, int(k))))
                         for k in rng.integers(1, n, size=2000)])
        blocks = max(1, n // 12)
        nullb = np.array([float(np.mean(d * np.repeat(
            rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
        w_p = stats.wilcoxon(a[:n], b[:n]).pvalue if n > 10 else np.nan
        srows.append(dict(method=method, baseline=base, n=n,
                          mean_winkler_diff=round(obs, 3),
                          win_rate_pct=round(winrate, 1),
                          wilcoxon_p=round(float(w_p), 4),
                          perm_shift_p=round(float((null <= obs).mean()), 4),
                          perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
S = pd.DataFrame(srows)
p = S["wilcoxon_p"].values
order = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]; prev = min(prev, p[i] * m / (rank + 1)); bh[i] = prev
S["wilcoxon_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task22_4_significance.csv", index=False)
print(S.to_string(index=False))
print(f"\nSaved {OUT}/task22_4_comparison.csv, {OUT}/task22_4_significance.csv")
