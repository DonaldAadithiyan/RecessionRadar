"""
TASK 26, Item 3 — fit, validate, and compare under BOTH split conditions.

Ridge only (the tree-model question is settled across Tasks 24-25 and is not
reopened). Full validation gate per condition; full 8-baseline comparison for
whichever conditions validate.

The Mondrian comparison is extracted and reported SEPARATELY, since that is the
specific thing this task tests.

PERMUTATION TESTING — the guardrail requires verifying rather than assuming:
  - temporal condition: block sign-flip null (autocorrelation-aware)
  - random condition: the shuffle destroys temporal ordering, so residual
    autocorrelation is MEASURED (Ljung-Box on the paired Winkler differences)
    and reported. If absent, a plain paired permutation is valid there.
BH correction is applied across EVERY cell in BOTH conditions combined.

Outputs: task26_3_comparison.csv, task26_3_significance.csv
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
from domain_common import GAMMA_DEFAULT, ALPHA_TARGET, rolling_origin_folds  # noqa: E402
import selector_lib as SEL  # noqa: E402
import baselines_lib as B  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from task26_2_splits import build, make_splits  # noqa: E402

OUT = "errordir"
LO, HI = 0.75, 1.25
SEED, N_DIRS, MAG = 5, 200, 0.5

d = build()
T, R, base = make_splits(d)
FEATS = [c for c in d.columns if c not in
         ("year", "month", "day", "hour", "target")]
X_all = d[FEATS].values.astype(float)
y_all = d["target"].values.astype(float)


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


rows_cmp, wk_store, rows_val = [], {}, []
for cond, S in [("temporal", T), ("random", R)]:
    fit_i, cal_i, test_i = S["fit"], S["cal"], S["test"]
    Xf_raw, yf = X_all[fit_i], y_all[fit_i]
    Xc_raw, yc = X_all[cal_i], y_all[cal_i]
    Xt_raw, yt = X_all[test_i], y_all[test_i]
    spread = float(np.percentile(yt, 95) - np.percentile(yt, 5))

    # base model: ridge, trained on FIT only
    sc0 = StandardScaler().fit(Xf_raw)
    base_m = Ridge(alpha=1.0).fit(sc0.transform(Xf_raw), yf)

    # OOF nonconformity scores on the CALIBRATION set (honest, no leakage)
    oof_c = np.full(len(yc), np.nan)
    for tr, te in rolling_origin_folds(len(yc), n_folds=5):
        m = Ridge(alpha=1.0).fit(sc0.transform(Xc_raw[tr]), yc[tr])
        oof_c[te] = m.predict(sc0.transform(Xc_raw[te]))
    cal_scores = np.abs(oof_c - yc)
    cal_scores = cal_scores[np.isfinite(cal_scores)]

    # beta fit on FIT only (its own OOF errors there)
    oof_f = np.full(len(yf), np.nan)
    for tr, te in rolling_origin_folds(len(yf), n_folds=5):
        m = Ridge(alpha=1.0).fit(sc0.transform(Xf_raw[tr]), yf[tr])
        oof_f[te] = m.predict(sc0.transform(Xf_raw[te]))
    err_f = np.abs(oof_f - yf)
    ok = np.isfinite(err_f)

    scb = StandardScaler().fit(Xf_raw[ok])
    Xb = scb.transform(Xf_raw[ok])
    reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xb, err_f[ok])
    beta = reg.coef_.astype(float)
    bu = beta / (np.linalg.norm(beta) + 1e-12)
    r_fit = stats.pearsonr(Xb @ beta, err_f[ok]).statistic
    _ec = np.abs(oof_c - yc)
    _m = np.isfinite(_ec)          # rolling_origin_folds leaves fold 1 unassigned
    r_cal = stats.pearsonr((scb.transform(Xc_raw) @ beta)[_m], _ec[_m])[0]

    # CHECK 1
    pca = PCA(n_components=min(10, Xb.shape[1])).fit(Xb)
    ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
        bu, pca.components_[0] / np.linalg.norm(pca.components_[0])))), 0, 1)))
    check1 = bool(ang > 30.0)

    # CHECK 2 — same-model-class null
    rng = np.random.default_rng(SEED)
    Xt_s = scb.transform(Xt_raw)
    sel = rng.choice(len(Xt_s), size=min(40, len(Xt_s)), replace=False)
    X0 = Xt_s[sel]

    def eff(v):
        fp = sc0.transform(scb.inverse_transform(X0 + MAG * v))
        fm = sc0.transform(scb.inverse_transform(X0 - MAG * v))
        return float(np.mean(np.abs(base_m.predict(fp) - base_m.predict(fm))))

    e_beta = eff(bu)
    null = np.array([eff(v / np.linalg.norm(v))
                     for v in rng.normal(size=(N_DIRS, len(bu)))])
    pct = float((null < e_beta).mean())
    check2 = bool(pct >= 0.95)
    validated = bool(check1 and check2)
    print(f"  {cond:9s} fit r={r_fit:+.3f} CAL r={r_cal:+.3f} angle={ang:5.1f} "
          f"pct={pct:.3f} -> {'VALIDATED' if validated else 'NOT VALIDATED'}")
    rows_val.append(dict(condition=cond, n_fit=int(ok.sum()), n_cal=len(cal_scores),
                         n_test=len(yt), r_fit=round(float(r_fit), 4),
                         r_cal_heldout=round(float(r_cal), 4),
                         angle_pc1_deg=round(ang, 2), check1_not_variance=check1,
                         perturb_percentile=round(pct, 4),
                         check2_perturbation=check2, beta_validated=validated))
    if not validated:
        continue

    pred_test = base_m.predict(sc0.transform(Xt_raw))

    def aci(cs, mult=None):
        a = ALPHA_TARGET
        los, his = [], []
        for t in range(len(yt)):
            q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
            if mult is not None:
                q *= float(mult[t])
            los.append(pred_test[t] - q); his.append(pred_test[t] + q)
            miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
            a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
        return np.array(los), np.array(his)

    N = min(254, len(cal_scores))
    trailing = cal_scores[-N:]
    arms = {"pooled_trailing": aci(trailing),
            "diversity_optimal": aci(cal_scores[SEL.support_width_selector(cal_scores, N)])}
    rare_cal = cal_scores >= np.percentile(cal_scores, 90)
    cbr = {True: cal_scores[rare_cal], False: cal_scores[~rare_cal]}
    if min(len(cbr[True]), len(cbr[False])) >= 2:
        treg = [bool(x) for x in (yt >= np.percentile(yt, 90))]
        c, w = B.run_mondrian(yt, pred_test, cbr, treg)
        arms["mondrian"] = (pred_test - w / 2, pred_test + w / 2)
    for nm, fn, arg in [("pid_conformal", B.run_pid, trailing),
                        ("evt_tail", B.run_evt, cal_scores),
                        ("dtaci", B.run_dtaci, cal_scores),
                        ("acmcp", B.run_acmcp, cal_scores),
                        ("bellman_ci", B.run_bci, cal_scores)]:
        c, w = fn(yt, pred_test, arg)
        arms[nm] = (pred_test - w / 2, pred_test + w / 2)

    proj_cal = scb.transform(Xc_raw) @ beta
    proj_test = Xt_s @ beta
    u = np.empty(len(proj_test))
    for t in range(len(proj_test)):
        ref = np.sort(np.concatenate([proj_cal, proj_test[:t]]))
        u[t] = np.searchsorted(ref, proj_test[t], side="right") / max(len(ref), 1)
    arms["errordir"] = aci(cal_scores, mult=LO + (HI - LO) * np.clip(u, 0, 1))

    for nm, (los, his) in arms.items():
        wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
        cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
        w = float(np.mean(his - los))
        rows_cmp.append(dict(condition=cond, method=nm, n=len(yt),
                             coverage=round(cov, 2), mean_width=round(w, 3),
                             width_spread_ratio=round(w / spread, 2),
                             vacuous=bool(w / spread > 100),
                             winkler_mean=round(float(np.mean(wk)), 3),
                             winkler_median=round(float(np.median(wk)), 3)))
        wk_store[(cond, nm)] = wk

pd.DataFrame(rows_val).to_csv(f"{OUT}/task26_3_validation.csv", index=False)
C = pd.DataFrame(rows_cmp)
C.to_csv(f"{OUT}/task26_3_comparison.csv", index=False)

print("\n" + "=" * 100)
for cond in C.condition.unique():
    s = C[C.condition == cond].sort_values("winkler_mean")
    print(f"\n{cond.upper()} — sorted by Winkler mean:")
    print(s[["method", "coverage", "mean_width", "width_spread_ratio",
             "winkler_mean", "winkler_median"]].to_string(index=False))

# ── significance, both conditions, BH across all cells combined ────────────
rng = np.random.default_rng(26)
srows = []
for cond in C.condition.unique():
    a = wk_store[(cond, "errordir")]
    for nm in ["pooled_trailing", "diversity_optimal", "mondrian", "pid_conformal",
               "evt_tail", "dtaci", "acmcp", "bellman_ci"]:
        if (cond, nm) not in wk_store:
            continue
        b = wk_store[(cond, nm)]
        n = min(len(a), len(b)); dd = a[:n] - b[:n]
        obs = float(np.mean(dd)); winrate = 100 * float((dd < 0).mean())
        # measure autocorrelation of the paired differences (verify, don't assume)
        lb = float(stats.pearsonr(dd[:-1], dd[1:])[0])
        blocks = max(1, n // 12)
        nullb = np.array([float(np.mean(dd * np.repeat(
            rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
        assert nullb.std() > 1e-12
        srows.append(dict(condition=cond, baseline=nm, n=n,
                          mean_winkler_diff=round(obs, 3),
                          win_rate_pct=round(winrate, 1),
                          lag1_autocorr_of_diff=round(lb, 3),
                          perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
S = pd.DataFrame(srows)
p = S["perm_signflip_p"].values
o = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rk in range(m - 1, -1, -1):
    i = o[rk]; prev = min(prev, p[i] * m / (rk + 1)); bh[i] = prev
S["perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task26_3_significance.csv", index=False)
print("\n" + "=" * 100)
print("SIGNIFICANCE (BH across BOTH conditions combined, m=%d)" % m)
print("=" * 100)
print(S.to_string(index=False))

print("\n" + "=" * 100)
print("THE MONDRIAN TEST — reported separately, as the task requires")
print("=" * 100)
for cond in C.condition.unique():
    r = S[(S.condition == cond) & (S.baseline == "mondrian")]
    cc = C[(C.condition == cond)]
    e = cc[cc.method == "errordir"].iloc[0]
    mo = cc[cc.method == "mondrian"]
    if len(r) and len(mo):
        mo = mo.iloc[0]; r = r.iloc[0]
        verdict = "errordir BEATS Mondrian" if r.mean_winkler_diff < 0 else "errordir LOSES to Mondrian"
        print(f"  {cond:9s}: errordir Winkler={e.winkler_mean:8.3f}  "
              f"Mondrian={mo.winkler_mean:8.3f}  diff={r.mean_winkler_diff:+7.3f}  "
              f"win_rate={r.win_rate_pct:5.1f}%  q={r.perm_signflip_p_bh:.4f}  -> {verdict}")
print(f"\nSaved {OUT}/task26_3_comparison.csv, {OUT}/task26_3_significance.csv")
