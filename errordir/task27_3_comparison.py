"""
TASK 27, Items 1-3 — CQR baseline, beta_tail fit + unmodified gate, head-to-head.

Runs on climate/ridge and energy/ridge: the two domains with an existing real,
non-null errordir result to compare against.

beta_mean  : RidgeCV regression of |error| on features        (Tasks 21-26)
beta_tail  : LOGISTIC regression of 1{s_i > q_alpha} on features   (NEW)

TARGET CHOICE: the binary indicator d_i = 1{s_i > q_alpha(FIT scores)} is used
rather than a continuous tail-proximity score, because it is exactly the event
coverage is defined on -- ACI consumes the (1-alpha) quantile, so "is this point
beyond it" is the quantity, not "how far along the severity scale". q_alpha is
computed on the FIT split ONLY (never CAL/TEST), so the label definition itself
carries no calibration or test information.

VALIDATION GATE: Task 21's two-part gate, UNMODIFIED. No new tail-relevance
gate is added (per the task: it would be circular). beta_tail must clear the
same bar built for beta_mean.

ANTI-GAMING: win rate, width/spread vacuity, permutation test chosen by whether
the domain is temporal (block sign-flip) -- both these domains are temporal per
Task 24/25 -- and BH correction across every cell tested.

Outputs: task27_2_validation.csv, task27_3_comparison.csv, task27_3_significance.csv
"""
import os
import sys
import io
import contextlib
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
from sklearn.linear_model import RidgeCV, LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from task27_1_cqr import cqr_intervals  # noqa: E402

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG = 200, 0.5

_b = io.StringIO()
with contextlib.redirect_stdout(_b):
    import domain_climate as CL

# energy comes from Task 25's cached build
EN = pd.read_csv("data/domains/task25/energy.csv")


def get_domain(name):
    if name == "climate":
        Xp, yp = CL.X_pool, CL.y_pool
        Xt, yt = CL.X_test, np.asarray(CL.Y_TEST, float)
        return Xp, yp, Xt, yt
    y = EN["target"].values.astype(float)
    X = EN.drop(columns=["target"]).values.astype(float)
    n_test = 400
    keep = np.arange(max(0, len(y) - (4000 + n_test)), len(y))
    X, y = X[keep], y[keep]
    return X[:-n_test], y[:-n_test], X[-n_test:], y[-n_test:]


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo:
        return w + (2.0 / alpha) * (lo - y)
    if y > hi:
        return w + (2.0 / alpha) * (y - hi)
    return w


def gate(bu, Xf_scaled, base_m, sc_b, sc_0, Xt_raw, kind_ridge=True, seed=SEED):
    """Task 21's two-part gate, unmodified."""
    pca = PCA(n_components=min(10, Xf_scaled.shape[1])).fit(Xf_scaled)
    ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
        bu, pca.components_[0] / np.linalg.norm(pca.components_[0])))), 0, 1)))
    check1 = bool(ang > 30.0)
    rng = np.random.default_rng(seed)
    Xt_s = sc_b.transform(Xt_raw)
    sel = rng.choice(len(Xt_s), size=min(40, len(Xt_s)), replace=False)
    X0 = Xt_s[sel]

    def eff(v):
        fp = sc_0.transform(sc_b.inverse_transform(X0 + MAG * v))
        fm = sc_0.transform(sc_b.inverse_transform(X0 - MAG * v))
        return float(np.mean(np.abs(base_m.predict(fp) - base_m.predict(fm))))

    e = eff(bu)
    null = np.array([eff(v / np.linalg.norm(v))
                     for v in rng.normal(size=(N_DIRS, len(bu)))])
    pct = float((null < e).mean())
    return ang, check1, e, pct, bool(pct >= 0.95)


rows_val, rows_cmp, wk_store = [], [], {}
for dom in ["climate", "energy"]:
    Xp, yp, Xt, yt = get_domain(dom)
    n_pool = len(yp)
    n_fit = int(round(FIT_FRAC * n_pool))
    FIT_I, CAL_I = np.arange(0, n_fit), np.arange(n_fit, n_pool)
    spread = float(np.percentile(yt, 95) - np.percentile(yt, 5))

    sc0 = StandardScaler().fit(Xp)
    okp = np.isfinite(yp)
    base_m = Ridge(alpha=1.0).fit(sc0.transform(Xp[okp]), yp[okp])

    oof = np.full(n_pool, np.nan)
    for tr, te in rolling_origin_folds(n_pool, n_folds=5):
        o = tr[np.isfinite(yp[tr])]
        if len(o) < 20:
            continue
        m = Ridge(alpha=1.0).fit(sc0.transform(Xp[o]), yp[o])
        oof[te] = m.predict(sc0.transform(Xp[te]))
    scores = np.abs(oof - yp)
    ok = np.isfinite(scores)
    fit_ok, cal_ok = FIT_I[ok[FIT_I]], CAL_I[ok[CAL_I]]
    pred_test = base_m.predict(sc0.transform(Xt))
    cal_scores = scores[cal_ok]

    scb = StandardScaler().fit(Xp[fit_ok])
    Xfs = scb.transform(Xp[fit_ok])

    # ── beta_mean (existing) ────────────────────────────────────────────────
    bm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xfs, scores[fit_ok]).coef_
    bm_u = bm / (np.linalg.norm(bm) + 1e-12)

    # ── beta_tail (NEW): logistic on 1{s > q_alpha(FIT)} ───────────────────
    q_a = float(np.quantile(scores[fit_ok], 1 - ALPHA_TARGET))   # FIT ONLY
    d_lab = (scores[fit_ok] > q_a).astype(int)
    lg = LogisticRegression(max_iter=2000, C=1.0).fit(Xfs, d_lab)
    bt = lg.coef_.ravel().astype(float)
    bt_u = bt / (np.linalg.norm(bt) + 1e-12)

    for tag, bu in [("beta_mean", bm_u), ("beta_tail", bt_u)]:
        ang, c1, e, pct, c2 = gate(bu, Xfs, base_m, scb, sc0, Xt)
        # held-out quality on CAL
        proj_c = scb.transform(Xp[cal_ok]) @ bu
        r_cal = float(stats.pearsonr(proj_c, cal_scores)[0])
        auc_cal = float(stats.rankdata(proj_c)[cal_scores > q_a].mean()
                        / len(proj_c)) if (cal_scores > q_a).any() else np.nan
        print(f"  {dom:8s} {tag:10s} angle={ang:5.1f} pct={pct:.3f} "
              f"r_cal={r_cal:+.3f} -> {'VALIDATED' if (c1 and c2) else 'NOT VALIDATED'}")
        rows_val.append(dict(domain=dom, beta=tag, n_fit=len(fit_ok),
                             n_cal=len(cal_ok), n_test=len(yt),
                             tail_label_rate=round(float(d_lab.mean()), 4),
                             r_cal_heldout=round(r_cal, 4),
                             angle_pc1_deg=round(ang, 2), check1_not_variance=c1,
                             perturb_effect=round(e, 6),
                             perturb_percentile=round(pct, 4),
                             check2_perturbation=c2,
                             beta_validated=bool(c1 and c2)))

    # ── interval arms ──────────────────────────────────────────────────────
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

    # CQR — fit on FIT, conformalize on CAL (Item 1)
    arms["cqr"] = cqr_intervals(Xp[fit_ok], yp[fit_ok], Xp[cal_ok], yp[cal_ok], Xt)

    # errordir arms
    for tag, bu in [("errordir_mean", bm_u), ("errordir_tail", bt_u)]:
        pc = scb.transform(Xp[cal_ok]) @ bu
        pt = scb.transform(Xt) @ bu
        u = np.empty(len(pt))
        for t in range(len(pt)):
            ref = np.sort(np.concatenate([pc, pt[:t]]))
            u[t] = np.searchsorted(ref, pt[t], side="right") / max(len(ref), 1)
        arms[tag] = aci(cal_scores, mult=LO + (HI - LO) * np.clip(u, 0, 1))

    for nm, (los, his) in arms.items():
        wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
        cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t]) for t in range(len(yt))]))
        w = float(np.mean(his - los))
        rows_cmp.append(dict(domain=dom, method=nm, n=len(yt),
                             coverage=round(cov, 2), mean_width=round(w, 3),
                             width_spread_ratio=round(w / spread, 2),
                             vacuous=bool(w / spread > 100),
                             winkler_mean=round(float(np.mean(wk)), 3),
                             winkler_median=round(float(np.median(wk)), 3)))
        wk_store[(dom, nm)] = wk

pd.DataFrame(rows_val).to_csv(f"{OUT}/task27_2_validation.csv", index=False)
C = pd.DataFrame(rows_cmp)
C.to_csv(f"{OUT}/task27_3_comparison.csv", index=False)
for dom in C.domain.unique():
    s = C[C.domain == dom].sort_values("winkler_mean")
    print(f"\n{dom.upper()} — sorted by Winkler:")
    print(s[["method", "coverage", "mean_width", "width_spread_ratio",
             "winkler_mean", "winkler_median"]].to_string(index=False))

rng = np.random.default_rng(27)
srows = []
for dom in C.domain.unique():
    for tag in ["errordir_mean", "errordir_tail"]:
        a = wk_store[(dom, tag)]
        for nm in ["pooled_trailing", "diversity_optimal", "mondrian",
                   "pid_conformal", "evt_tail", "dtaci", "acmcp",
                   "bellman_ci", "cqr"]:
            if (dom, nm) not in wk_store:
                continue
            b = wk_store[(dom, nm)]
            n = min(len(a), len(b)); dd = a[:n] - b[:n]
            obs = float(np.mean(dd)); wr = 100 * float((dd < 0).mean())
            blocks = max(1, n // 12)
            nullb = np.array([float(np.mean(dd * np.repeat(
                rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
            assert nullb.std() > 1e-12
            srows.append(dict(domain=dom, beta=tag, baseline=nm, n=n,
                              mean_winkler_diff=round(obs, 3),
                              win_rate_pct=round(wr, 1),
                              perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
S = pd.DataFrame(srows)
p = S["perm_signflip_p"].values
o = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
for rk in range(m - 1, -1, -1):
    i = o[rk]; prev = min(prev, p[i] * m / (rk + 1)); bh[i] = prev
S["perm_signflip_p_bh"] = np.round(bh, 4)
S.to_csv(f"{OUT}/task27_3_significance.csv", index=False)

print("\n" + "=" * 100)
print("PRE-REGISTERED CRITERION: does beta_tail narrow the coverage gap vs")
print("diversity_optimal WITHOUT giving back the Winkler advantage over DtACI?")
print("=" * 100)
for dom in C.domain.unique():
    cc = C[C.domain == dom].set_index("method")
    do_cov = cc.loc["diversity_optimal", "coverage"]
    dt_wk = cc.loc["dtaci", "winkler_mean"]
    for tag in ["errordir_mean", "errordir_tail"]:
        gap = do_cov - cc.loc[tag, "coverage"]
        adv = dt_wk - cc.loc[tag, "winkler_mean"]
        print(f"  {dom:8s} {tag:14s} coverage gap vs divopt = {gap:+6.2f}pp   "
              f"Winkler advantage over DtACI = {adv:+7.3f}")
print(f"\nSaved task27_2_validation.csv, task27_3_comparison.csv, task27_3_significance.csv")
