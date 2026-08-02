"""
TASK 13 — Temporal generalization of the calibration-diversity finding.

Implements the spec in task13_temporal_generalization_spec.md.

Every recession result in the paper rests on one fixed split (pre-2020 /
post-2020). This re-runs the core diagnostic at three historical cutoffs, each
with a genuine recession in its test window, and asks whether support-width
diversity still outpredicts rare-event count.

CUTOFFS (fixed after a feasibility check — see the spec):
  1989-01  tests the 1990-91 recession   (263 train months,  5 rare in test)
  2006-01  tests the 2007-09 crisis      (467 train months, 16 rare in test)
  2020-01  tests COVID + tightening      (635 train months,  2 rare in test)
The 1999 cutoff was dropped: the 2001 recession peaks at 31.3% in this smoothed
series, so its test window contains ZERO months above the paper's >=50 rare
threshold and cannot discriminate.

LEAKAGE DISCIPLINE (the decision that makes or breaks this task):
  The saved ensemble was fit on data through 2019-12. Using it to score a 1990
  test window would leak 30 years of future information. EVERY cutoff therefore
  uses a surrogate RegressorChain refit on that cutoff's training rows only —
  including 2020, so the comparison across cutoffs is like-for-like. The
  consequence is that the 2020 arm will not exactly match the published Table 8.

  Features are also recomputed per cutoff: STL, rolling statistics and anomaly
  thresholds all use training rows only.

REPRODUCTION GATE:
  The 2020 cutoff runs first. If the refit surrogate does not recover the
  published qualitative diagnostic (rho(supp) > rho(rare) at 6M), that is a
  harness bug and the run aborts rather than reporting novel cutoffs built on a
  broken base — the same discipline as Task 7's Phase-3 gate.

Outputs:
  fix-reg/task13_temporal_cutoffs.csv
  fix-reg/task13_selector_by_cutoff.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import RegressorChain

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()
from ensemble_stubs import recession_targets, safe_logit, safe_inv_logit  # noqa: E402

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, support_width, wilson_ci_from_indicator,
    random_draw_sweep, sweep_diagnostics, rolling_origin_folds, GAMMA_DEFAULT,
)
import selector_lib as SEL  # noqa: E402

DATA = "data/fix/feature_selected_reg_full.csv"
OUT = "fix-reg"
CUTOFFS = ["2020-01-01", "2006-01-01", "1989-01-01"]   # gate first, then novel
TEST_LEN = 65
N_DRAWS = 200
CAL_FRAC = 0.40
SEED = 5
LABELS = ["Current", "1M", "3M", "6M"]
RARE_THRESH = 50.0

INDICATORS = ["CPI", "INDPRO", "unemployment_rate", "share_price", "PPI",
              "OECD_CLI_index", "CSI_index", "gdp_per_capita",
              "10_year_rate", "1_year_rate", "3_months_rate", "6_months_rate"]

base = pd.read_csv(DATA)
base["date"] = pd.to_datetime(base["date"])
base = base.sort_values("date").reset_index(drop=True)
FEATURES = [c for c in base.columns if c not in recession_targets + ["date"]]


def build_features_for_cutoff(df, cutoff):
    """
    Recompute STL / rolling / anomaly features using TRAINING ROWS ONLY.

    STL is fit on the training slice and its trend/residual extended over the
    full frame by re-fitting on the training portion and reindexing; rolling
    statistics use .shift(1) so no same-month leakage; anomaly thresholds come
    from training residuals only.
    """
    fe = df.copy()
    tr_mask = fe["date"] < cutoff
    n_tr = int(tr_mask.sum())

    for col in INDICATORS:
        if col not in fe.columns:
            continue
        s_full = fe[col].ffill().bfill()
        s_tr = s_full[:n_tr]
        if s_tr.notna().sum() >= 24:
            # STL is a global smoother, so a single fit over the whole frame
            # would let post-cutoff observations shape pre-cutoff features AND
            # let future months inform the test rows. Both are leaks.
            #
            # Instead: fit on training rows for the training features, then
            # EXPAND one test month at a time — each test row's decomposition
            # uses only data available up to that month. Slower, but it is the
            # only construction that is honest for a 1990 cutoff.
            stl_tr = STL(s_tr, seasonal=13, period=12).fit()
            trend = np.full(len(fe), np.nan)
            resid = np.full(len(fe), np.nan)
            trend[:n_tr] = stl_tr.trend
            resid[:n_tr] = stl_tr.resid
            for t in range(n_tr, len(fe)):
                seg = s_full[:t + 1]
                if seg.notna().sum() < 24:
                    continue
                st = STL(seg, seasonal=13, period=12).fit()
                trend[t] = np.asarray(st.trend)[-1]
                resid[t] = np.asarray(st.resid)[-1]
            fe[f"{col}_trend"] = trend
            fe[f"{col}_residual"] = resid

        fe[f"{col}_diff1"] = fe[col].diff(1)
        fe[f"{col}_diff3"] = fe[col].diff(3)
        fe[f"{col}_pct_change1"] = fe[col].pct_change(1)
        for w in (3, 6, 12):
            sh = fe[col].shift(1)
            fe[f"{col}_rollstd{w}"] = sh.rolling(w).std()
            fe[f"{col}_rollmax{w}"] = sh.rolling(w).max()
            fe[f"{col}_rollmin{w}"] = sh.rolling(w).min()
        fe[f"{col}_lag6"] = fe[col].shift(6)

    if "PPI" in fe.columns and "CPI" in fe.columns:
        fe["PPI_CPI_diff"] = fe["PPI"] - fe["CPI"]

    for col in ["CPI", "unemployment_rate", "share_price",
                "3_months_rate", "6_months_rate", "10_year_rate"]:
        rcol = f"{col}_residual"
        if rcol not in fe.columns:
            continue
        mu = fe.loc[tr_mask, rcol].mean()
        sd = fe.loc[tr_mask, rcol].std()
        fe[f"{col}_anomaly"] = (((fe[rcol] < mu - 3 * sd) |
                                 (fe[rcol] > mu + 3 * sd)).astype(int))

    for c in FEATURES:
        if c not in fe.columns:
            fe[c] = np.nan
    return fe


def clean(d):
    return d.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)


def surrogate():
    return RegressorChain(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      max_depth=4, random_state=SEED),
        order=[0, 1, 2, 3])


def run_cutoff(cutoff):
    print("\n" + "=" * 90)
    print(f"CUTOFF {cutoff[:7]}")
    print("=" * 90)

    fe = build_features_for_cutoff(base, cutoff)
    tr = fe[fe["date"] < cutoff].reset_index(drop=True)
    te = fe[fe["date"] >= cutoff].head(TEST_LEN).reset_index(drop=True)

    X_tr = clean(tr[FEATURES]).values.astype(float)
    X_te = clean(te[FEATURES]).values.astype(float)
    y_tr = tr[recession_targets].values.astype(float)
    y_te = te[recession_targets].values.astype(float)

    rec_tr = tr["recession_probability"].values
    is_rare = rec_tr >= RARE_THRESH
    rare_te = int((te["recession_probability"].values >= RARE_THRESH).sum())
    n_cal = int(CAL_FRAC * len(tr))

    print(f"  train {len(tr)} months ({int(is_rare.sum())} rare) | "
          f"test {len(te)} months ({rare_te} rare) | N_cal={n_cal}")

    # ── Refit the surrogate on THIS cutoff's training rows, out-of-fold ─────
    ytr_logit = safe_logit(y_tr)
    oof = np.full_like(y_tr, np.nan, dtype=float)
    for k, (a, b) in enumerate(rolling_origin_folds(len(tr), n_folds=6), 1):
        a_ok = a[np.isfinite(ytr_logit[a]).all(axis=1)]
        if len(a_ok) < 24:
            continue
        m = surrogate()
        m.fit(X_tr[a_ok], ytr_logit[a_ok])
        oof[b] = safe_inv_logit(m.predict(X_tr[b]))

    full_ok = np.isfinite(ytr_logit).all(axis=1)
    m_full = surrogate()
    m_full.fit(X_tr[full_ok], ytr_logit[full_ok])
    preds_te = safe_inv_logit(m_full.predict(X_te))
    print(f"  OOF-scored training months: {int(np.isfinite(oof[:,0]).sum())}"
          f" / {len(tr)}")

    diag_rows, sel_rows = [], []
    for h_idx, h in enumerate(LABELS):
        scores = np.abs(oof[:, h_idx] - y_tr[:, h_idx])
        valid = np.isfinite(scores)
        if valid.sum() < n_cal + 10:
            print(f"    {h}: insufficient scored months, skipped")
            continue

        sweep = random_draw_sweep(scores, is_rare, y_te[:, h_idx],
                                  preds_te[:, h_idx], N=n_cal,
                                  n_draws=N_DRAWS, seed=SEED)
        d = sweep_diagnostics(sweep, domain=f"cutoff_{cutoff[:7]}",
                              extra=dict(cutoff=cutoff[:7], horizon=h,
                                         n_train=len(tr), N_cal=n_cal,
                                         n_test=len(te),
                                         rare_in_train=int(is_rare.sum()),
                                         rare_in_test=rare_te))
        d["gap"] = round(float(d["rho_supp"] - d["rho_rare"]), 3)
        diag_rows.append(d)
        print(f"    {h:8s} rho(supp)={d['rho_supp']:+.3f}  "
              f"rho(rare)={d['rho_rare']:+.3f}  gap={d['gap']:+.3f}  "
              f"within-tertile={d['within_tertile_rho_rare']}")

        # selector vs trailing baseline at this cutoff
        v = np.where(valid)[0]
        trailing = scores[v[-n_cal:]]
        sel = scores[SEL.support_width_selector(scores, n_cal)]
        cb, _, wb = run_aci(y_te[:, h_idx], preds_te[:, h_idx], trailing,
                            gamma=GAMMA_DEFAULT)
        cs, _, ws = run_aci(y_te[:, h_idx], preds_te[:, h_idx],
                            sel[np.isfinite(sel)], gamma=GAMMA_DEFAULT)
        cov_b, w_b = coverage_and_width(cb, wb)
        cov_s, w_s = coverage_and_width(cs, ws)
        blo, bhi = wilson_ci_from_indicator(cb)
        slo, shi = wilson_ci_from_indicator(cs)
        sel_rows.append(dict(cutoff=cutoff[:7], horizon=h, N_cal=n_cal,
                             n_test_scored=int(np.isfinite(y_te[:, h_idx]).sum()),
                             rare_in_test=rare_te,
                             trailing_cov=round(cov_b, 2),
                             trailing_lo=round(blo, 2), trailing_hi=round(bhi, 2),
                             trailing_w=round(w_b, 3),
                             selector_cov=round(cov_s, 2),
                             selector_lo=round(slo, 2), selector_hi=round(shi, 2),
                             selector_w=round(w_s, 3),
                             gain_pp=round(cov_s - cov_b, 2),
                             width_mult=round(w_s / w_b, 2) if w_b else np.nan))
    return diag_rows, sel_rows


print("=" * 90)
print("TASK 13 — temporal generalization across historical cutoffs")
print("=" * 90)
print("  Every cutoff uses a surrogate refit on ITS OWN training rows.")
print("  The saved ensemble is never used (it would leak future data).")

all_diag, all_sel = [], []

# ── Gate: 2020 first ───────────────────────────────────────────────────────
d20, s20 = run_cutoff("2020-01-01")
all_diag += d20
all_sel += s20

g6 = [r for r in d20 if r["horizon"] == "6M"]
print("\n" + "-" * 90)
print("REPRODUCTION GATE (2020 cutoff, refit surrogate)")
print("-" * 90)
if not g6:
    print("  6M row missing — aborting.")
    sys.exit(1)
g6 = g6[0]
gate_ok = g6["rho_supp"] > g6["rho_rare"]
print(f"  6M: rho(supp)={g6['rho_supp']:+.3f} vs rho(rare)={g6['rho_rare']:+.3f}"
      f"  -> {'PASS' if gate_ok else 'FAIL'}")
if not gate_ok:
    print("\n  Gate FAILED: the refit surrogate does not recover the published")
    print("  qualitative diagnostic. Fix the harness before trusting novel")
    print("  cutoffs. Aborting.")
    pd.DataFrame(all_diag).to_csv(f"{OUT}/task13_temporal_cutoffs_DEBUG.csv",
                                  index=False)
    sys.exit(1)
print("  Gate passed. Proceeding to the novel cutoffs.")

for c in CUTOFFS[1:]:
    d, s = run_cutoff(c)
    all_diag += d
    all_sel += s

D = pd.DataFrame(all_diag)
S = pd.DataFrame(all_sel)
cols = ["cutoff", "horizon", "n_train", "N_cal", "n_test", "rare_in_train",
        "rare_in_test", "rho_supp", "R2_supp", "rho_rare", "R2_rare", "gap",
        "within_tertile_rho_rare", "cov_mean", "cov_std"]
D = D[[c for c in cols if c in D.columns]]
D.to_csv(f"{OUT}/task13_temporal_cutoffs.csv", index=False)
S.to_csv(f"{OUT}/task13_selector_by_cutoff.csv", index=False)

print("\n" + "=" * 90)
print("DIAGNOSTIC BY CUTOFF — rho(support) vs rho(rare-count)")
print("=" * 90)
print(D.to_string(index=False))

print("\n" + "=" * 90)
print("VERDICT against the pre-registered criteria")
print("=" * 90)
piv = D.pivot_table(index="horizon", columns="cutoff", values="gap")
print("\n  gap = rho(supp) - rho(rare); positive means diversity dominates")
print(piv.round(3).to_string())

for cut in D["cutoff"].unique():
    sub = D[D["cutoff"] == cut]
    n_pos = int((sub["gap"] > 0).sum())
    print(f"\n  {cut}: diversity dominates at {n_pos}/{len(sub)} horizons "
          f"(6M gap = {sub[sub.horizon=='6M']['gap'].iloc[0]:+.3f})"
          if (sub.horizon == "6M").any() else "")

all_pos = int((D["gap"] > 0).sum())
print(f"\n  Overall: diversity dominates in {all_pos} of {len(D)} "
      f"(cutoff x horizon) cells")

print("\n" + "=" * 90)
print("SELECTOR BY CUTOFF")
print("=" * 90)
print(S[["cutoff", "horizon", "rare_in_test", "trailing_cov", "selector_cov",
         "gain_pp", "width_mult"]].to_string(index=False))
print(f"\nSaved {OUT}/task13_temporal_cutoffs.csv, task13_selector_by_cutoff.csv")
