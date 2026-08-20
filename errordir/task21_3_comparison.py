"""
TASK 21, Item 3 — the test: does the error-direction method improve on the
existing frontier, or move along it?

RUNS ON 6M ONLY. Item 1's gate validated beta at 6M alone (1/4 horizons);
Current/1M/3M failed the perturbation check and are excluded per the guardrail
("if either check fails, stop here — do not proceed with an unvalidated beta").

Compared against the two methods the spec names, not a strawman:
  - diversity_optimal selector (the paper's method)
  - DtACI (Gibbs & Candes 2024) — the current Winkler leader
on coverage, mean width, and Winkler score (alpha=0.10, the repo's own rule).

PRE-REGISTERED OUTCOMES (stated before results were seen):
  1. Better coverage AND narrower width than BOTH -> frontier-escaping claim,
     to be met with maximum skepticism.
  2. Matched coverage, narrower width (or vice versa) -> a real, reportable
     improvement: a better position on the frontier.
  3. No improvement / worse on both -> valid negative result.

STATISTICAL DISCIPLINE (same as Task 20):
  - BH multiple-comparison correction across every cell tested
  - circular-shift permutation test preserving temporal autocorrelation
  - held-out TIME split, not just cross-validation

Outputs: errordir/task21_3_comparison.csv, task21_3_significance.csv
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
import task21_2_scaling as SC  # noqa: E402
from task_oof_and_probit import (  # noqa: E402
    X_train as X_train_df, X_test as X_test_df, y_train as y_pool,
    y_test, oof_pred as preds_pool_oof, preds_test, n_pool,
)
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = "errordir"
H_IDX, H = 3, "6M"           # 6M only — the one validated horizon
FIT_FRAC, SEED = 0.60, 5
N_FIX = 254

X_train, X_test = X_train_df.values, X_test_df.values
n_fit = int(round(FIT_FRAC * n_pool))
FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)

abs_err = np.abs(preds_pool_oof[:, H_IDX] - y_pool[:, H_IDX])
fit_ok = FIT_IDX[np.isfinite(abs_err[FIT_IDX])]
cal_ok = CAL_IDX[np.isfinite(abs_err[CAL_IDX])]

# refit beta identically to Item 1 (FIT split only)
scaler = StandardScaler().fit(X_train[fit_ok])
reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scaler.transform(X_train[fit_ok]),
                                                 abs_err[fit_ok])
beta = reg.coef_.astype(float)

proj_cal = scaler.transform(X_train[cal_ok]) @ beta
proj_test = scaler.transform(X_test) @ beta
to_rank = SC.fit_cdf(proj_cal)               # CDF frozen on CAL only
u_test = to_rank(proj_test)
mult = SC.multiplier(u_test)

y_h = y_test[:, H_IDX]
p_h = preds_test[:, H_IDX]
cal_scores = abs_err[cal_ok]                 # CAL split = the calibration set


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def run_aci_intervals(y_h, p_h, cs, gamma=GAMMA_DEFAULT, at=ALPHA_TARGET,
                      mult=None):
    """ACI returning per-step bounds, optionally width-scaled by mult[t]."""
    a = at
    los, his, cov = [], [], []
    for t in range(len(y_h)):
        q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
        if mult is not None:
            q = q * float(mult[t])
        lo, hi = p_h[t] - q, p_h[t] + q
        los.append(lo); his.append(hi)
        yt = y_h[t]
        if np.isnan(yt):
            cov.append(np.nan); continue
        miss = 1 if (yt < lo or yt > hi) else 0
        cov.append(1 - miss)
        a = float(np.clip(a + gamma * (at - miss), 0.01, 0.99))
    return np.array(los), np.array(his), np.array(cov)


def summarize(name, los, his, cov):
    ok = ~np.isnan(cov)
    wk = np.array([winkler(y_h[t], los[t], his[t])
                   for t in range(len(y_h)) if ok[t]])
    return dict(method=name, horizon=H, n=int(ok.sum()),
                coverage=round(100 * float(np.nanmean(cov[ok])), 2),
                mean_width=round(float(np.mean(his - los)), 3),
                winkler_mean=round(float(np.mean(wk)), 3),
                winkler_median=round(float(np.median(wk)), 3)), wk, cov[ok]


rows, wk_store, cov_store = [], {}, {}

# ── Baseline 1: diversity-optimal selector, on the SAME CAL split ───────────
sel_idx = SEL.support_width_selector(cal_scores, min(N_FIX, len(cal_scores)))
cal_sel = cal_scores[sel_idx]
l, hh, c = run_aci_intervals(y_h, p_h, cal_sel)
r, wk, cv = summarize("diversity_optimal", l, hh, c); rows.append(r)
wk_store["diversity_optimal"], cov_store["diversity_optimal"] = wk, cv

# ── Baseline 2: DtACI (widths symmetric about p_h, so bounds reconstruct) ───
cov_d, wid_d = B.run_dtaci(y_h, p_h, cal_scores)
l_d, h_d = p_h - wid_d / 2, p_h + wid_d / 2
r, wk, cv = summarize("dtaci", l_d, h_d, cov_d); rows.append(r)
wk_store["dtaci"], cov_store["dtaci"] = wk, cv

# ── Baseline 3: pooled/trailing ACI, for reference ─────────────────────────
l, hh, c = run_aci_intervals(y_h, p_h, cal_scores)
r, wk, cv = summarize("pooled_trailing", l, hh, c); rows.append(r)
wk_store["pooled_trailing"], cov_store["pooled_trailing"] = wk, cv

# ── THE METHOD: error-direction scaled, on both calibration bases ──────────
l, hh, c = run_aci_intervals(y_h, p_h, cal_scores, mult=mult)
r, wk, cv = summarize("errordir_pooled", l, hh, c); rows.append(r)
wk_store["errordir_pooled"], cov_store["errordir_pooled"] = wk, cv

# NOTE: an "errordir_selector" arm was also run and produced byte-identical
# results to errordir_pooled — at 6M the selector's calibration subset yields
# the same operative quantile as the full CAL set, so the two arms are the same
# method. It is omitted rather than reported twice as if it were independent
# evidence.

D = pd.DataFrame(rows)
D.to_csv(f"{OUT}/task21_3_comparison.csv", index=False)

print("=" * 100)
print(f"TASK 21 Item 3 — comparison at {H} (the one horizon where beta validated)")
print("=" * 100)
print(f"  CAL split = {len(cal_scores)} months (beta never saw these)")
print(f"  multiplier: mean={mult.mean():.4f}  range=[{mult.min():.3f},{mult.max():.3f}]")
frac_hi = float((u_test >= 1.0).mean()); frac_lo = float((u_test <= 0.0).mean())
print(f"  *** COVARIATE DRIFT: test-projection mean rank = {u_test.mean():.3f}, "
      f"not ~0.5.")
print(f"      {100*frac_hi:.1f}% of test months project ABOVE every CAL month; "
      f"{100*frac_lo:.1f}% below all.")
print(f"      The design pinned the mean multiplier to 1.0 under a uniform rank "
      f"distribution;")
print(f"      the realized mean is {mult.mean():.4f}, so part of any coverage gain "
      f"is uniform widening,")
print(f"      NOT differential allocation. This is reported, not corrected away.\n")
drift = dict(test_mean_rank=round(float(u_test.mean()), 4),
             frac_rank_saturated_high=round(frac_hi, 4),
             frac_rank_saturated_low=round(frac_lo, 4),
             realized_mean_multiplier=round(float(mult.mean()), 4),
             designed_mean_multiplier=1.0)
pd.DataFrame([drift]).to_csv(f"{OUT}/task21_3_drift.csv", index=False)
print(D.to_string(index=False))

# ── Significance: paired, temporal-autocorrelation-aware ───────────────────
print("\n" + "=" * 100)
print("SIGNIFICANCE — paired Winkler differences vs. each baseline")
print("=" * 100)
rng = np.random.default_rng(21)
srows = []
for method in ["errordir_pooled"]:
    for base in ["diversity_optimal", "dtaci", "pooled_trailing"]:
        a, b = wk_store[method], wk_store[base]
        n = min(len(a), len(b))
        d = a[:n] - b[:n]                       # negative = method is better
        obs = float(np.mean(d))
        # circular-shift null preserving autocorrelation of the difference
        null = np.array([float(np.mean(np.roll(d, int(k))))
                         for k in rng.integers(1, n, size=2000)])
        # sign-flip block permutation as a second null
        blocks = max(1, n // 12)
        nullb = []
        for _ in range(2000):
            sg = np.repeat(rng.choice([-1, 1], size=blocks + 1), 12)[:n]
            nullb.append(float(np.mean(d * sg)))
        nullb = np.array(nullb)
        p_shift = float((null <= obs).mean())
        p_flip = float((nullb <= obs).mean())
        w_p = stats.wilcoxon(a[:n], b[:n]).pvalue if n > 10 else np.nan
        srows.append(dict(method=method, baseline=base, n=n,
                          mean_winkler_diff=round(obs, 3),
                          pct_better=round(100 * float((d < 0).mean()), 1),
                          wilcoxon_p=round(float(w_p), 4),
                          perm_shift_p=round(p_shift, 4),
                          perm_signflip_p=round(p_flip, 4)))
S = pd.DataFrame(srows)
# BH across the whole family
p = S["wilcoxon_p"].values
order = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]; prev = min(prev, p[i] * m / (rank + 1)); bh[i] = prev
S["wilcoxon_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task21_3_significance.csv", index=False)
print(S.to_string(index=False))
print(f"\nSaved {OUT}/task21_3_comparison.csv, {OUT}/task21_3_significance.csv")
