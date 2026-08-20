"""
TASK 23, Item 4 — re-run the comparison for whichever horizons validated in
Item 2, with the FULL anti-gaming discipline Task 22 showed was necessary:

  - WIN RATE reported alongside the mean, as a first-class result
  - WIDTH/SPREAD ratio against this project's ~100 vacuity threshold
  - PERMUTATION tests respecting temporal autocorrelation (NOT Wilcoxon, which
    assumes exchangeable pairs on what is a time series with large outliers)
  - BH correction across every cell tested

Task 22's trap, restated so it can be recognised if it recurs: a
BH-significant mean Winkler gain, produced by a low win rate and a width ratio
past ~100, is NOT a validated result.

Scaling uses Task 21's ORIGINAL band [0.75, 1.25] with the mean-multiplier-1.0
property, and Task 22 Item 1's rolling recentering (which was verified leak-free
and is the correct default). Task 22's raised ceiling is NOT used — it is a
known dead end.

Outputs: errordir/task23_4_comparison.csv, task23_4_significance.csv
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

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
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from task23_1_disagreement_check import disagreement  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_train as y_pool, y_test,
    oof_pred as preds_pool_oof, preds_test, n_pool, LABELS,
)

OUT = "errordir"
LO, HI = 0.75, 1.25          # Task 21's band; Task 22's ceiling is a dead end
FIT_FRAC = 0.60
TARGET_SPREAD = {"Current": 0.804, "1M": 0.808, "3M": 0.732, "6M": 0.596}

n_fit = int(round(FIT_FRAC * n_pool))
FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)
D_train, _ = disagreement(X_train_df)
D_test, _ = disagreement(X_test_df)

V = pd.read_csv(f"{OUT}/task23_2_validation.csv")
VALID = [r.horizon for r in V.itertuples() if r.beta_validated]
print("=" * 104)
print("TASK 23 Item 4 — comparison for validated horizons:", VALID or "(none)")
print("=" * 104)
if not VALID:
    print("\n  No horizon validated in Item 2 — nothing to compare. Stopping.")
    pd.DataFrame().to_csv(f"{OUT}/task23_4_comparison.csv", index=False)
    sys.exit(0)


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def run(y_h, p_h, cs, mult=None):
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


rows, wk_store = [], {}
for h in VALID:
    h_idx = LABELS.index(h)
    y_h, p_h = y_test[:, h_idx], preds_test[:, h_idx]
    spread = TARGET_SPREAD[h]

    dcols = [f"disag_std_{h}", f"disag_range_{h}", f"disag_maxpair_{h}"]
    Xtr = np.column_stack([X_train_df.values, D_train[dcols].values])
    Xte = np.column_stack([X_test_df.values, D_test[dcols].values])
    ae = np.abs(preds_pool_oof[:, h_idx] - y_pool[:, h_idx])
    fo = FIT_IDX[np.isfinite(ae[FIT_IDX])]
    co = CAL_IDX[np.isfinite(ae[CAL_IDX])]
    sc = StandardScaler().fit(Xtr[fo])
    rg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(sc.transform(Xtr[fo]), ae[fo])
    beta = rg.coef_.astype(float)
    proj_cal = sc.transform(Xtr[co]) @ beta
    proj_test = sc.transform(Xte) @ beta
    cal_scores = ae[co]

    # rolling recentering (Task 22 Item 1, verified leak-free)
    u = np.empty(len(proj_test))
    for t in range(len(proj_test)):
        ref = np.sort(np.concatenate([proj_cal, proj_test[:t]]))
        u[t] = np.searchsorted(ref, proj_test[t], side="right") / max(len(ref), 1)
    mult = LO + (HI - LO) * np.clip(u, 0, 1)

    def summ(name, los, his, cov, mu=None):
        ok = ~np.isnan(cov)
        wk = np.array([winkler(y_h[t], los[t], his[t])
                       for t in range(len(y_h)) if ok[t]])
        w = float(np.mean(his - los))
        return dict(horizon=h, method=name, n=int(ok.sum()),
                    coverage=round(100 * float(np.nanmean(cov[ok])), 2),
                    mean_width=round(w, 3),
                    width_spread_ratio=round(w / spread, 1),
                    vacuous=bool(w / spread > 100),
                    winkler_mean=round(float(np.mean(wk)), 3),
                    winkler_median=round(float(np.median(wk)), 3),
                    mean_multiplier=None if mu is None else round(float(mu.mean()), 4)), wk

    sel = SEL.support_width_selector(cal_scores, min(254, len(cal_scores)))
    l, hh, c = run(y_h, p_h, cal_scores[sel])
    r, wk = summ("diversity_optimal", l, hh, c); rows.append(r); wk_store[(h, r["method"])] = wk
    cov_d, wid_d = B.run_dtaci(y_h, p_h, cal_scores)
    r, wk = summ("dtaci", p_h - wid_d / 2, p_h + wid_d / 2, cov_d); rows.append(r); wk_store[(h, r["method"])] = wk
    l, hh, c = run(y_h, p_h, cal_scores, mult=mult)
    r, wk = summ("errordir_disag", l, hh, c, mult); rows.append(r); wk_store[(h, r["method"])] = wk

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task23_4_comparison.csv", index=False)
print(D.to_string(index=False))

print("\n" + "=" * 104)
print("SIGNIFICANCE — permutation only (no Wilcoxon: pairs are not exchangeable)")
print("=" * 104)
rng = np.random.default_rng(23)
srows = []
for h in VALID:
    for base in ["diversity_optimal", "dtaci"]:
        a, b = wk_store[(h, "errordir_disag")], wk_store[(h, base)]
        n = min(len(a), len(b)); d = a[:n] - b[:n]
        obs = float(np.mean(d))
        winrate = 100 * float((d < 0).mean())
        null = np.array([float(np.mean(np.roll(d, int(k))))
                         for k in rng.integers(1, n, size=2000)])
        blocks = max(1, n // 12)
        nullb = np.array([float(np.mean(d * np.repeat(
            rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
        srows.append(dict(horizon=h, baseline=base, n=n,
                          mean_winkler_diff=round(obs, 3),
                          win_rate_pct=round(winrate, 1),
                          perm_shift_p=round(float((null <= obs).mean()), 4),
                          perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
S = pd.DataFrame(srows)
p = S["perm_signflip_p"].values
order = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]; prev = min(prev, p[i] * m / (rank + 1)); bh[i] = prev
S["perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task23_4_significance.csv", index=False)
print(S.to_string(index=False))
print(f"\nSaved {OUT}/task23_4_comparison.csv, {OUT}/task23_4_significance.csv")
