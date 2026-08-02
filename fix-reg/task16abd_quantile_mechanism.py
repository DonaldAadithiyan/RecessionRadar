"""
TASK 16, Items A / B / D — direct tests of the quantile-reach mechanism.

The paper's theory section argues a three-link chain:
    support width  ->  Q_C(1-alpha)  ->  coverage
Every result so far tests links 1 and 3 jointly (support width vs coverage).
The middle link has been argued analytically and checked at a single point, but
never measured across the 200-draw diagnostic. These three items close that.

PIPELINE CHANGE MADE FIRST (per the spec's guardrail about approximating from
aggregates): `domain_common.random_draw_sweep` previously computed each draw's
calibration set and then discarded everything except support width, IQR,
entropy, coverage and width. It now also records **Q_C**, the set's own
(1-alpha) quantile — the quantity ACI actually consumes — and accepts
`alpha_target` so Item D can vary the target coverage level. Both are additive;
every existing caller is unaffected (verified).

ITEM A — is support width a good proxy for Q_C, or only for coverage?
    Three-way comparison per cell: support width, rare count, Q_C, each against
    coverage; plus rho(support width, Q_C) to test link 1 on its own.
    PRE-REGISTERED EXPECTATION (spec): Q_C should predict coverage at least as
    well as support width, and support width should be a moderate-to-strong
    predictor of Q_C. Reported honestly either way.

ITEM B — multivariate decomposition, N fixed.
    coverage ~ b0 + b1*SupportWidth + b2*RareCount, standardized coefficients,
    partial R^2, 95% CIs. N is fixed at each cell's published value (the 200
    draws already hold it fixed, so this is a direct extension).

ITEM D — does the finding hold at other target coverage levels?
    Re-run the diagnostic at alpha in {0.05, 0.10, 0.15, 0.20, 0.25} on the
    primary testbed. The paper currently infers alpha-robustness from ACI's
    observed operating range [0.073, 0.132]; this tests it directly.

Outputs:
  fix-reg/task16a_qc_direct_test.csv
  fix-reg/task16b_multivariate.csv
  fix-reg/task16d_alpha_sweep.csv
  fix-reg/task16_perdraw_recession.csv   (raw per-draw data, for audit)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ensemble_stubs  # noqa: E402
ensemble_stubs.install()

from domain_common import random_draw_sweep  # noqa: E402

OUT = "fix-reg"
N_DRAWS = 200
SEED = 7
N_FIX = 254
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25]
LABELS = ["Current", "1M", "3M", "6M"]


def rho(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(st.spearmanr(x, y).correlation)


def r2(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(st.pearsonr(x, y).statistic ** 2)


print("=" * 100)
print("TASK 16 A/B/D — direct tests of the support-width -> Q_C -> coverage chain")
print("=" * 100)

# ── Assemble the cells ──────────────────────────────────────────────────────
CELLS = []
import task_oof_and_probit as T  # noqa: E402
for h_idx, h in enumerate(LABELS):
    CELLS.append(("Recession", "stacking-chain", h,
                  np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx]),
                  (T.rec_prob_tr if hasattr(T, "rec_prob_tr")
                   else T.train_df["recession_probability"].values) >= 50,
                  T.y_test[:, h_idx], T.preds_test[:, h_idx], N_FIX))

import runpy  # noqa: E402
g = runpy.run_path("fix-reg/domain_healthcare.py", run_name="_hc")
for name in ["ridge", "gradboost"]:
    CELLS.append(("Healthcare", name, "30-day", g["SCORES"][name],
                  g["RARE_POOL"], g["Y_TEST"], g["PREDS_TEST"][name], 254))

gc = runpy.run_path("fix-reg/domain_climate.py", run_name="_cl")
for name in ["ridge", "gradboost"]:
    CELLS.append(("Climate", name, "region-month", gc["SCORES"][name],
                  gc["RARE_POOL"], gc["Y_TEST"], gc["PREDS_TEST"][name], 254))

# ── ITEM A ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 100)
print("ITEM A — three-way predictor comparison, plus link 1 (support -> Q_C)")
print("-" * 100)

a_rows, perdraw_store = [], []
for dom, mod, hor, scores, is_rare, y_te, p_te, N in CELLS:
    s = np.asarray(scores, float)
    valid = np.isfinite(s)
    if valid.sum() < N + 10:
        print(f"  [skip] {dom}/{mod}/{hor}")
        continue
    sw = random_draw_sweep(s, np.asarray(is_rare, bool), y_te, p_te, N=N,
                           n_draws=N_DRAWS, seed=SEED)
    sw.insert(0, "cell", f"{dom}/{mod}/{hor}")
    if dom == "Recession":
        perdraw_store.append(sw)

    cov = sw["cov"].values
    supp = sw["supp"].values
    rare = sw["rare_count"].values.astype(float)
    qc = sw["Q_C"].values

    row = dict(domain=dom, model=mod, horizon=hor, n_draws=len(sw), N=N,
               # link 1: does support width predict Q_C at all?
               rho_supp_qc=round(rho(supp, qc), 3),
               R2_supp_qc=round(r2(supp, qc), 3),
               # link 3 / three-way against coverage
               rho_supp_cov=round(rho(supp, cov), 3),
               R2_supp_cov=round(r2(supp, cov), 3),
               rho_rare_cov=round(rho(rare, cov), 3),
               R2_rare_cov=round(r2(rare, cov), 3),
               rho_qc_cov=round(rho(qc, cov), 3),
               R2_qc_cov=round(r2(qc, cov), 3))
    row["qc_beats_supp"] = bool(row["rho_qc_cov"] >= row["rho_supp_cov"])
    a_rows.append(row)
    print(f"  {dom:11s} {mod:14s} {hor:12s} | "
          f"supp->Q_C rho={row['rho_supp_qc']:+.3f} | "
          f"cov: supp={row['rho_supp_cov']:+.3f} rare={row['rho_rare_cov']:+.3f} "
          f"Q_C={row['rho_qc_cov']:+.3f}"
          f"{'  (Q_C >= supp)' if row['qc_beats_supp'] else '  (supp > Q_C)'}")

A = pd.DataFrame(a_rows)
A.to_csv(f"{OUT}/task16a_qc_direct_test.csv", index=False)
if perdraw_store:
    pd.concat(perdraw_store).to_csv(f"{OUT}/task16_perdraw_recession.csv",
                                    index=False)

print("\n  PRE-REGISTERED CHECK (stated in the spec before running):")
print(f"    (i)  Q_C predicts coverage at least as well as support width: "
      f"{int(A['qc_beats_supp'].sum())} of {len(A)} cells")
print(f"    (ii) support width is a moderate-to-strong predictor of Q_C: "
      f"median rho = {A['rho_supp_qc'].median():.3f}, "
      f"min = {A['rho_supp_qc'].min():.3f}")

# ── ITEM B ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 100)
print("ITEM B — coverage ~ SupportWidth + RareCount (standardized, N fixed)")
print("-" * 100)

b_rows = []


def standardize(v):
    v = np.asarray(v, float)
    sd = np.std(v)
    return (v - v.mean()) / sd if sd > 1e-12 else v * 0.0


for dom, mod, hor, scores, is_rare, y_te, p_te, N in CELLS:
    s = np.asarray(scores, float)
    if np.isfinite(s).sum() < N + 10:
        continue
    sw = random_draw_sweep(s, np.asarray(is_rare, bool), y_te, p_te, N=N,
                           n_draws=N_DRAWS, seed=SEED)
    y = sw["cov"].values
    # Near-constant coverage (the saturated Current horizon) leaves no variance
    # to decompose; standardising it would divide by ~0. Skip rather than emit
    # a meaningless regression.
    if np.std(y) < 1e-9:
        print(f"  [skip] {dom}/{mod}/{hor}: coverage constant across draws "
              f"(sd={np.std(y):.2e}) — nothing to decompose")
        continue
    X = np.column_stack([standardize(sw["supp"].values),
                         standardize(sw["rare_count"].values.astype(float))])
    yz = standardize(y)
    Xd = np.column_stack([np.ones(len(yz)), X])

    beta, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
    resid = yz - Xd @ beta
    n, k = len(yz), Xd.shape[1]
    dof = n - k
    sigma2 = float(resid @ resid / dof)
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)
    tcrit = st.t.ppf(0.975, dof)

    # partial R^2 for each predictor: drop it and compare residual sums
    def partial_r2(j):
        keep = [0] + [c for c in (1, 2) if c != j]
        Xr = Xd[:, keep]
        br, *_ = np.linalg.lstsq(Xr, yz, rcond=None)
        rr = yz - Xr @ br
        ssr_full = float(resid @ resid)
        ssr_red = float(rr @ rr)
        return (ssr_red - ssr_full) / ssr_red if ssr_red > 0 else 0.0

    sst = float(((yz - yz.mean()) ** 2).sum())
    r2_full = 1 - float(resid @ resid) / sst if sst > 1e-12 else np.nan
    b_rows.append(dict(
        domain=dom, model=mod, horizon=hor, n_draws=n, N=N,
        beta_supp=round(float(beta[1]), 4),
        beta_supp_lo=round(float(beta[1] - tcrit * se[1]), 4),
        beta_supp_hi=round(float(beta[1] + tcrit * se[1]), 4),
        partial_R2_supp=round(partial_r2(1), 4),
        beta_rare=round(float(beta[2]), 4),
        beta_rare_lo=round(float(beta[2] - tcrit * se[2]), 4),
        beta_rare_hi=round(float(beta[2] + tcrit * se[2]), 4),
        partial_R2_rare=round(partial_r2(2), 4),
        model_R2=round(r2_full, 4),
        rare_ci_excludes_zero=bool(
            (beta[2] - tcrit * se[2]) * (beta[2] + tcrit * se[2]) > 0)))
    r = b_rows[-1]
    print(f"  {dom:11s} {mod:14s} {hor:12s} "
          f"b_supp={r['beta_supp']:+.3f} [{r['beta_supp_lo']:+.3f},"
          f"{r['beta_supp_hi']:+.3f}] pR2={r['partial_R2_supp']:.3f} | "
          f"b_rare={r['beta_rare']:+.3f} [{r['beta_rare_lo']:+.3f},"
          f"{r['beta_rare_hi']:+.3f}] pR2={r['partial_R2_rare']:.3f}")

B = pd.DataFrame(b_rows)
B.to_csv(f"{OUT}/task16b_multivariate.csv", index=False)

# ── ITEM D ──────────────────────────────────────────────────────────────────
print("\n" + "-" * 100)
print("ITEM D — does the finding hold at other target coverage levels?")
print("-" * 100)
print("  primary testbed (recession), alpha in", ALPHAS)

d_rows = []
for h_idx, h in enumerate(LABELS):
    scores = np.abs(T.oof_pred[:, h_idx] - T.y_train[:, h_idx])
    is_rare = (T.train_df["recession_probability"].values >= 50)
    for a in ALPHAS:
        sw = random_draw_sweep(scores, is_rare, T.y_test[:, h_idx],
                               T.preds_test[:, h_idx], N=N_FIX,
                               n_draws=N_DRAWS, seed=SEED, alpha_target=a)
        cov = sw["cov"].values
        supp = sw["supp"].values
        rare = sw["rare_count"].values.astype(float)
        qc = sw["Q_C"].values
        d_rows.append(dict(horizon=h, alpha=a, nominal=round(100 * (1 - a), 1),
                           n_draws=len(sw),
                           rho_supp_qc=round(rho(supp, qc), 3),
                           rho_rare_qc=round(rho(rare, qc), 3),
                           rho_supp_cov=round(rho(supp, cov), 3),
                           rho_rare_cov=round(rho(rare, cov), 3),
                           gap=round(rho(supp, cov) - rho(rare, cov), 3),
                           cov_mean=round(float(np.mean(cov)), 2),
                           cov_std=round(float(np.std(cov)), 3)))
    sub = [r for r in d_rows if r["horizon"] == h]
    print(f"  {h:8s} gap by alpha: " +
          "  ".join(f"{r['alpha']:.2f}:{r['gap']:+.3f}" for r in sub))

D = pd.DataFrame(d_rows)
D.to_csv(f"{OUT}/task16d_alpha_sweep.csv", index=False)

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("\nItem A — three-way predictor comparison:")
print(A[["domain", "model", "horizon", "rho_supp_qc", "rho_supp_cov",
         "rho_rare_cov", "rho_qc_cov", "qc_beats_supp"]].to_string(index=False))
print("\nItem B — rare-count partial R^2 (confirmatory if ~0):")
print(B[["domain", "model", "horizon", "beta_supp", "partial_R2_supp",
         "beta_rare", "partial_R2_rare",
         "rare_ci_excludes_zero"]].to_string(index=False))
print("\nItem D — gap by alpha (positive = diversity dominates):")
print(D.pivot_table(index="horizon", columns="alpha",
                    values="gap").round(3).to_string())
print(f"\n  Cells where the gap stays positive: "
      f"{int((D['gap'] > 0).sum())} of {len(D)}")
print(f"\nSaved {OUT}/task16a_qc_direct_test.csv, task16b_multivariate.csv, "
      f"task16d_alpha_sweep.csv")
