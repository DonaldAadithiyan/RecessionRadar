"""
TASK 25, Items 4-6 — baselines, beta fits + validation gate, and the full
anti-gaming comparison, on the two new datasets.

MODELS: XGBoost and LightGBM (the task's two practical models), plus a RIDGE
reference arm. Ridge is included NOT as a third practical model but as a
control: Item 1 established that tree-based beta fails Check 2 on its merits.
If ridge validates on these new datasets while the tree models do not, that
isolates the failure to model class rather than to the new domains — a
distinction neither Item 1 nor Task 24 could make on two domains alone.

DISCIPLINE (unchanged from Tasks 21-24):
  - three-way split: FIT 60% / CAL 40% of pool, TEST held out
  - beta = continuous RidgeCV regression of |error| on features, FIT only
  - Check 1: angle(beta, PC1) > 30 deg
  - Check 2: same-model-class null (Item 1 construction A), >= 95th pct
  - scaling: Task 21 band [0.75,1.25] + Task 22 rolling recentering
  - comparison: full 8-baseline suite, win rate, vacuity, permutation, BH

PERMUTATION VALIDITY: Task 24 found the circular-shift null is VACUOUS for a
mean statistic (rolling a vector does not change its mean). Only the block
sign-flip null is used here, and its non-degeneracy is asserted at runtime.

Outputs: task25_4_baselines.csv, task25_5_validation.csv,
         task25_6_comparison.csv, task25_6_significance.csv
"""
import os, sys, warnings, numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "fix-reg")); sys.path.insert(0, HERE)
os.chdir(ROOT)

import ensemble_stubs; ensemble_stubs.install()
from domain_common import GAMMA_DEFAULT, ALPHA_TARGET, rolling_origin_folds
import selector_lib as SEL
import baselines_lib as B
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb

OUT = "errordir"
LO, HI = 0.75, 1.25
FIT_FRAC, SEED = 0.60, 5
N_DIRS, MAG = 200, 0.5
N_TEST = 400          # held-out evaluation stream per dataset
POOL_CAP = 4000       # keep the pool tractable; sampled BEFORE any fitting

DATA = {"insurance": "data/domains/task25/insurance.csv",
        "energy": "data/domains/task25/energy.csv"}
TEMPORAL = {"insurance": False, "energy": True}


def make_model(kind):
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "xgboost":
        return xgb.XGBRegressor(n_estimators=200, learning_rate=0.06, max_depth=4,
                                subsample=0.9, verbosity=0, random_state=SEED)
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, max_depth=4,
                             verbose=-1, random_state=SEED)


def winkler(y, lo, hi, alpha=ALPHA_TARGET):
    w = hi - lo
    if y < lo: return w + (2.0 / alpha) * (lo - y)
    if y > hi: return w + (2.0 / alpha) * (y - hi)
    return w


def load(dname):
    d = pd.read_csv(DATA[dname])
    y = d["target"].values.astype(float)
    X = d.drop(columns=["target"]).values.astype(float)
    rng = np.random.default_rng(SEED)
    if TEMPORAL[dname]:
        n = len(y)                                  # keep time order
        keep = np.arange(max(0, n - (POOL_CAP + N_TEST)), n)
    else:
        keep = rng.choice(len(y), size=min(POOL_CAP + N_TEST, len(y)), replace=False)
        keep = np.sort(keep)
    X, y = X[keep], y[keep]
    test_idx = np.arange(len(y) - N_TEST, len(y))
    pool_idx = np.arange(0, len(y) - N_TEST)
    return X[pool_idx], y[pool_idx], X[test_idx], y[test_idx]


rows_val, rows_cmp, wk_store, rows_base = [], [], {}, []
for dname in DATA:
    Xp, yp, Xt, yt = load(dname)
    spread = float(np.percentile(yt, 95) - np.percentile(yt, 5))
    n_pool = len(yp)
    n_fit = int(round(FIT_FRAC * n_pool))
    FIT_IDX, CAL_IDX = np.arange(0, n_fit), np.arange(n_fit, n_pool)
    print(f"\n{'='*100}\n{dname}: pool={n_pool} fit={len(FIT_IDX)} cal={len(CAL_IDX)} "
          f"test={len(yt)} feats={Xp.shape[1]} spread={spread:.3f}\n{'='*100}")

    for kind in ["xgboost", "lightgbm", "ridge"]:
        sc0 = StandardScaler().fit(Xp)
        Xp_s, Xt_s0 = sc0.transform(Xp), sc0.transform(Xt)
        A = Xp_s if kind == "ridge" else Xp
        Bx = Xt_s0 if kind == "ridge" else Xt

        # OOF scores on the pool (this domain's nonconformity scores)
        oof = np.full(n_pool, np.nan)
        for tr, te in rolling_origin_folds(n_pool, n_folds=5):
            m = make_model(kind); m.fit(A[tr], yp[tr]); oof[te] = m.predict(A[te])
        mfull = make_model(kind); mfull.fit(A, yp)
        pred_test = mfull.predict(Bx)
        scores = np.abs(oof - yp)

        ok = np.isfinite(scores)
        fit_ok, cal_ok = FIT_IDX[ok[FIT_IDX]], CAL_IDX[ok[CAL_IDX]]

        sc = StandardScaler().fit(Xp[fit_ok])
        Xf = sc.transform(Xp[fit_ok])
        reg = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf, scores[fit_ok])
        beta = reg.coef_.astype(float); bu = beta / (np.linalg.norm(beta) + 1e-12)
        r_fit = stats.pearsonr(Xf @ beta, scores[fit_ok]).statistic
        Xc = sc.transform(Xp[cal_ok])
        r_cal = stats.pearsonr(Xc @ beta, scores[cal_ok]).statistic

        # CHECK 1
        pca = PCA(n_components=min(10, Xf.shape[1])).fit(Xf)
        ang = np.degrees(np.arccos(np.clip(abs(float(np.dot(
            bu, pca.components_[0] / np.linalg.norm(pca.components_[0])))), 0, 1)))
        check1 = bool(ang > 30.0)

        # CHECK 2 — same-model-class null (Item 1 construction A)
        rng = np.random.default_rng(SEED)
        Xt_s = sc.transform(Xt)
        sel = rng.choice(len(Xt_s), size=min(40, len(Xt_s)), replace=False)
        X0 = Xt_s[sel]

        def eff(v):
            fp = sc.inverse_transform(X0 + MAG * v)
            fm = sc.inverse_transform(X0 - MAG * v)
            if kind == "ridge":
                fp, fm = sc0.transform(fp), sc0.transform(fm)
            return float(np.mean(np.abs(mfull.predict(fp) - mfull.predict(fm))))

        e_beta = eff(bu)
        null = np.array([eff(v / np.linalg.norm(v))
                         for v in rng.normal(size=(N_DIRS, len(bu)))])
        pct = float((null < e_beta).mean())
        check2 = bool(pct >= 0.95)
        validated = bool(check1 and check2)

        print(f"  {kind:10s} fit r={r_fit:+.3f} CAL r={r_cal:+.3f} angle={ang:5.1f} "
              f"pct={pct:.3f} -> {'VALIDATED' if validated else 'NOT VALIDATED'}")
        rows_val.append(dict(dataset=dname, model=kind, n_fit=len(fit_ok),
                             n_cal=len(cal_ok), n_test=len(yt),
                             r_fit=round(float(r_fit), 4),
                             r_cal_heldout=round(float(r_cal), 4),
                             angle_pc1_deg=round(ang, 2), check1_not_variance=check1,
                             perturb_effect=round(e_beta, 6),
                             perturb_null_p95=round(float(np.percentile(null, 95)), 6),
                             perturb_percentile=round(pct, 4),
                             check2_perturbation=check2, beta_validated=validated))

        # ── Item 4: full baseline suite (built for EVERY fit, not just valid) ──
        cal_scores = scores[cal_ok]
        N = min(254, len(cal_scores))
        trailing = cal_scores[-N:]
        selidx = SEL.support_width_selector(cal_scores, N)

        def aci(cs, mult=None):
            a = ALPHA_TARGET; los, his = [], []
            for t in range(len(yt)):
                q = float(np.quantile(cs, np.clip(1 - a, 0, 1)))
                if mult is not None: q *= float(mult[t])
                los.append(pred_test[t] - q); his.append(pred_test[t] + q)
                miss = 1 if (yt[t] < los[-1] or yt[t] > his[-1]) else 0
                a = float(np.clip(a + GAMMA_DEFAULT * (ALPHA_TARGET - miss), .01, .99))
            return np.array(los), np.array(his)

        arms = {"pooled_trailing": aci(trailing),
                "diversity_optimal": aci(cal_scores[selidx])}
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

        # errordir arm (only meaningful for validated fits, but computed always)
        proj_cal = Xc @ beta; proj_test = sc.transform(Xt) @ beta
        u = np.empty(len(proj_test))
        for t in range(len(proj_test)):
            ref = np.sort(np.concatenate([proj_cal, proj_test[:t]]))
            u[t] = np.searchsorted(ref, proj_test[t], side="right") / max(len(ref), 1)
        mult = LO + (HI - LO) * np.clip(u, 0, 1)
        arms["errordir"] = aci(cal_scores, mult=mult)

        for nm, (los, his) in arms.items():
            wk = np.array([winkler(yt[t], los[t], his[t]) for t in range(len(yt))])
            cov = 100 * float(np.mean([(los[t] <= yt[t] <= his[t])
                                       for t in range(len(yt))]))
            w = float(np.mean(his - los))
            rec = dict(dataset=dname, model=kind, method=nm, n=len(yt),
                       coverage=round(cov, 2), mean_width=round(w, 3),
                       width_spread_ratio=round(w / spread, 2),
                       vacuous=bool(w / spread > 100),
                       winkler_mean=round(float(np.mean(wk)), 3),
                       winkler_median=round(float(np.median(wk)), 3))
            rows_base.append(rec)
            if validated: rows_cmp.append(rec)
            wk_store[(dname, kind, nm)] = wk

pd.DataFrame(rows_val).to_csv(f"{OUT}/task25_5_validation.csv", index=False)
pd.DataFrame(rows_base).to_csv(f"{OUT}/task25_4_baselines.csv", index=False)
pd.DataFrame(rows_cmp).to_csv(f"{OUT}/task25_6_comparison.csv", index=False)

V = pd.DataFrame(rows_val)
print("\n" + "=" * 100); print("ITEM 5 — full validation table (all fits)"); print("=" * 100)
print(V[["dataset", "model", "r_cal_heldout", "angle_pc1_deg", "perturb_percentile",
         "check1_not_variance", "check2_perturbation", "beta_validated"]].to_string(index=False))

# ── Item 6: significance for validated fits ────────────────────────────────
srows = []
rng = np.random.default_rng(25)
for r in V[V.beta_validated].itertuples():
    a = wk_store[(r.dataset, r.model, "errordir")]
    for nm in ["pooled_trailing", "diversity_optimal", "mondrian", "pid_conformal",
               "evt_tail", "dtaci", "acmcp", "bellman_ci"]:
        if (r.dataset, r.model, nm) not in wk_store: continue
        b = wk_store[(r.dataset, r.model, nm)]
        n = min(len(a), len(b)); d = a[:n] - b[:n]
        obs = float(np.mean(d)); winrate = 100 * float((d < 0).mean())
        blocks = max(1, n // 12)
        nullb = np.array([float(np.mean(d * np.repeat(
            rng.choice([-1, 1], size=blocks + 1), 12)[:n])) for _ in range(2000)])
        assert nullb.std() > 1e-12, "sign-flip null is degenerate"
        srows.append(dict(dataset=r.dataset, model=r.model, baseline=nm, n=n,
                          mean_winkler_diff=round(obs, 3),
                          win_rate_pct=round(winrate, 1),
                          perm_signflip_p=round(float((nullb <= obs).mean()), 4)))
if srows:
    S = pd.DataFrame(srows)
    p = S["perm_signflip_p"].values
    o = np.argsort(p); m = len(p); bh = np.empty(m); prev = 1.0
    for rk in range(m - 1, -1, -1):
        i = o[rk]; prev = min(prev, p[i] * m / (rk + 1)); bh[i] = prev
    S["perm_signflip_p_bh"] = np.round(bh, 4)
    S.to_csv(f"{OUT}/task25_6_significance.csv", index=False)
    print("\n" + "=" * 100); print("ITEM 6 — significance (validated fits only)"); print("=" * 100)
    print(S.to_string(index=False))
else:
    pd.DataFrame().to_csv(f"{OUT}/task25_6_significance.csv", index=False)
    print("\n  No validated fits — no comparison run (Item 6 guardrail).")
print(f"\nSaved task25_4/5/6 outputs.")
