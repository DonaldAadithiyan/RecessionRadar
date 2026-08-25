"""
TASK 31, Items 2-3 — full comparison suite on epidemic, and the test of the
pre-registered prediction.

PRE-REGISTERED PREDICTION (stated in the task, restated here before results):
  Methods anchored to an ONLINE-UPDATING estimate (errordir, DtACI, and other
  ACI-family methods) will outperform methods that FREEZE a calibration-period
  estimate (RLCP, CQR) on this domain's concept-drift-affected test period,
  mirroring energy's pattern.

NOTE ON DIRECTION (from Item 1): epidemic's drift runs OPPOSITE to energy's --
errors SHRINK in the test period (ratio 0.443) rather than grow (1.315). So the
frozen-estimate methods should fail here by being TOO WIDE / over-covering,
whereas on energy they failed by being too narrow / under-covering. Same
mechanism, opposite symptom. This makes it a stricter test.

Full suite: errordir (beta_mean + fixed multiplier), corrected RLCP (Task 29),
CQR, pooled_trailing, mondrian, pid_conformal, evt_tail, dtaci, acmcp,
bellman_ci, diversity_optimal. Same FIT/CAL/TEST discipline as Task 21 onward.

Anti-gaming (Task 30's pre-registered handling reused, not rebuilt):
  win rate, unbounded-rate column, Winkler median alongside mean, block
  sign-flip permutation, BH across every cell.

Outputs: task31_2_baselines.csv, task31_3_comparison.csv,
         task31_3_significance.csv
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg")); sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs; ensemble_stubs.install()
from domain_common import GAMMA_DEFAULT, ALPHA_TARGET, rolling_origin_folds
import selector_lib as SEL
import baselines_lib as B
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from task27_1_cqr import cqr_intervals
from task29_1_rlcp import rlcp_intervals, median_heuristic_gamma

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC = 0.60
N_TEST, POOL_CAP = 400, 4000

# ONLINE-ANCHORED vs FROZEN-ESTIMATE, classified BEFORE results are seen
ONLINE = {"errordir", "pooled_trailing", "diversity_optimal", "dtaci",
          "pid_conformal", "acmcp", "bellman_ci", "evt_tail", "mondrian"}
FROZEN = {"rlcp", "cqr"}

d = pd.read_csv("data/domains/task31/epidemic.csv").sort_values("date")
d = d.drop(columns=["date"]).reset_index(drop=True)
y = d["target"].values.astype(float)
X = d.drop(columns=["target"]).values.astype(float)
keep = np.arange(max(0, len(y) - (POOL_CAP + N_TEST)), len(y))
X, y = X[keep], y[keep]
Xp, yp, Xt, yt = X[:-N_TEST], y[:-N_TEST], X[-N_TEST:], y[-N_TEST:]
spread = float(np.percentile(yt, 95) - np.percentile(yt, 5))
n_pool = len(yp); n_fit = int(round(FIT_FRAC * n_pool))
FIT_I, CAL_I = np.arange(0, n_fit), np.arange(n_fit, n_pool)

sc0 = StandardScaler().fit(Xp)
okp = np.isfinite(yp)
base_m = Ridge(alpha=1.0).fit(sc0.transform(Xp[okp]), yp[okp])
oof = np.full(n_pool, np.nan)
for tr, te in rolling_origin_folds(n_pool, n_folds=5):
    o = tr[np.isfinite(yp[tr])]
    if len(o) < 20: continue
    oof[te] = Ridge(alpha=1.0).fit(sc0.transform(Xp[o]), yp[o]).predict(sc0.transform(Xp[te]))
scores = np.abs(oof - yp); ok = np.isfinite(scores)
fit_ok, cal_ok = FIT_I[ok[FIT_I]], CAL_I[ok[CAL_I]]
pred_test = base_m.predict(sc0.transform(Xt))
cal_scores = scores[cal_ok]

scb = StandardScaler().fit(Xp[fit_ok])
bm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(scb.transform(Xp[fit_ok]),
                                                scores[fit_ok]).coef_
bu = bm / (np.linalg.norm(bm) + 1e-12)
z_fit = scb.transform(Xp[fit_ok]) @ bu
z_cal = scb.transform(Xp[cal_ok]) @ bu
z_test = scb.transform(Xt) @ bu
gamma = median_heuristic_gamma(z_fit)

print(f"epidemic: pool={n_pool} fit={len(fit_ok)} cal={len(cal_ok)} "
      f"test={len(yt)} feats={Xp.shape[1]} spread={spread:.3f}")


def winkler(yy, lo, hi, alpha=ALPHA_TARGET):
    if not np.isfinite(hi - lo): return np.inf
    w = hi - lo
    if yy < lo: return w + (2.0 / alpha) * (lo - yy)
    if yy > hi: return w + (2.0 / alpha) * (yy - hi)
    return w


def aci(cs, mult=None):
    a = ALPHA_TARGET; los, his = [], []
    for t in range(len(yt)):
        q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
        if mult is not None: q *= float(mult[t])
        los.append(pred_test[t] - q); his.append(pred_test[t] + q)
        miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
        a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
    return np.array(los), np.array(his)


N = min(254, len(cal_scores)); trailing = cal_scores[-N:]
arms = {"pooled_trailing": aci(trailing),
        "diversity_optimal": aci(cal_scores[SEL.support_width_selector(cal_scores, N)])}
rare = cal_scores >= np.percentile(cal_scores, 90)
cbr = {True: cal_scores[rare], False: cal_scores[~rare]}
if min(len(cbr[True]), len(cbr[False])) >= 2:
    treg = [bool(v) for v in (yt >= np.percentile(yt, 90))]
    c, w = B.run_mondrian(yt, pred_test, cbr, treg)
    arms["mondrian"] = (pred_test - w / 2, pred_test + w / 2)
for nm, fn, arg in [("pid_conformal", B.run_pid, trailing),
                    ("evt_tail", B.run_evt, cal_scores),
                    ("dtaci", B.run_dtaci, cal_scores),
                    ("acmcp", B.run_acmcp, cal_scores),
                    ("bellman_ci", B.run_bci, cal_scores)]:
    c, w = fn(yt, pred_test, arg)
    arms[nm] = (pred_test - w / 2, pred_test + w / 2)
arms["cqr"] = cqr_intervals(Xp[fit_ok], yp[fit_ok], Xp[cal_ok], yp[cal_ok], Xt)
lo_r, hi_r, unb = rlcp_intervals(z_cal, cal_scores, z_test, pred_test, gamma)
arms["rlcp"] = (lo_r, hi_r)

u = np.empty(len(z_test))
for t in range(len(z_test)):
    ref = np.sort(np.concatenate([z_cal, z_test[:t]]))
    u[t] = np.searchsorted(ref, z_test[t], side="right") / max(len(ref), 1)
arms["errordir"] = aci(cal_scores, mult=LO + (HI - LO) * np.clip(u, 0, 1))

rows, wk_store = [], {}
for nm, (los, his) in arms.items():
    fin = np.isfinite(his - los)
    wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
    cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
    wid = float(np.mean((his - los)[fin])) if fin.any() else np.nan
    unbr = float(1 - fin.mean())
    rows.append(dict(domain="epidemic", method=nm,
                     anchor=("online" if nm in ONLINE else "frozen"),
                     n=len(yt), coverage=round(cov, 2),
                     mean_width=round(wid, 4),
                     width_spread_ratio=round(wid / spread, 2),
                     vacuous=bool(wid / spread > 100),
                     unbounded_rate=round(unbr, 4),
                     winkler_mean=(None if not np.all(np.isfinite(wk))
                                   else round(float(np.mean(wk)), 4)),
                     winkler_median=(None if unbr >= 0.5
                                     else round(float(np.median(wk)), 4))))
    wk_store[nm] = wk

C = pd.DataFrame(rows)
C.to_csv(f"{OUT}/task31_2_baselines.csv", index=False)
C.to_csv(f"{OUT}/task31_3_comparison.csv", index=False)
print("\nEPIDEMIC — sorted by Winkler mean:")
print(C.sort_values("winkler_mean", na_position="last")[
    ["method", "anchor", "coverage", "mean_width", "width_spread_ratio",
     "unbounded_rate", "winkler_mean", "winkler_median"]].to_string(index=False))

rng = np.random.default_rng(31)
srows = []
a = wk_store["errordir"]
for nm in [k for k in arms if k != "errordir"]:
    b = wk_store[nm]
    n = min(len(a), len(b)); dd = a[:n] - b[:n]
    if not np.all(np.isfinite(dd)):
        srows.append(dict(baseline=nm, anchor=("online" if nm in ONLINE else "frozen"),
                          n=n, mean_winkler_diff=None, win_rate_pct=None,
                          perm_signflip_p=None, note="unbounded intervals"))
        continue
    obs = float(np.mean(dd)); wr = 100 * float((dd < 0).mean())
    blocks = max(1, n // 12)
    nullb = np.array([float(np.mean(dd * np.repeat(
        rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
    assert nullb.std() > 1e-12
    srows.append(dict(baseline=nm, anchor=("online" if nm in ONLINE else "frozen"),
                      n=n, mean_winkler_diff=round(obs, 4),
                      win_rate_pct=round(wr, 1),
                      perm_signflip_p=round(float((nullb <= obs).mean()), 4), note=""))
S = pd.DataFrame(srows)
fin = S["perm_signflip_p"].notna()
p = S.loc[fin, "perm_signflip_p"].values
o = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rk in range(m - 1, -1, -1):
    i = o[rk]; prev = min(prev, p[i] * m / (rk + 1)); bh[i] = prev
S.loc[fin, "perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task31_3_significance.csv", index=False)
print("\nerrordir vs each baseline (negative = errordir better):")
print(S.to_string(index=False))

print("\n" + "=" * 92)
print("THE PRE-REGISTERED TEST: online-anchored vs frozen-estimate methods")
print("=" * 92)
on = C[(C.anchor == "online") & C.winkler_mean.notna()]
fr = C[(C.anchor == "frozen")]
print(f"  ONLINE  (n={len(on)}): median Winkler={on.winkler_mean.median():.4f}  "
      f"best={on.winkler_mean.min():.4f} ({on.loc[on.winkler_mean.idxmin(),'method']})")
if fr.winkler_mean.notna().any():
    frv = fr[fr.winkler_mean.notna()]
    print(f"  FROZEN  (n={len(frv)}): median Winkler={frv.winkler_mean.median():.4f}  "
          f"best={frv.winkler_mean.min():.4f} ({frv.loc[frv.winkler_mean.idxmin(),'method']})")
for r in fr.itertuples():
    print(f"    {r.method:8s} cov={r.coverage:6.2f}  width={r.mean_width:8.4f}  "
          f"ratio={r.width_spread_ratio:5.2f}  unbounded={r.unbounded_rate:.3f}  "
          f"winkler={r.winkler_mean}")
print(f"\nSaved task31_2_baselines.csv, task31_3_comparison.csv, task31_3_significance.csv")
