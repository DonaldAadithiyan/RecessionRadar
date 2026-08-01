"""
DOMAIN 2 — Healthcare: 30-day hospital readmission (rare adverse event).

Dataset: UCI "Diabetes 130-US Hospitals for Years 1999-2008" (id 296),
101,766 encounters, freely downloadable without credentialing.
Rare event: readmission within 30 days ("<30"), ~11.2% base rate.

Per the revision spec this domain exists to test the *calibration mechanism*,
not to build an elaborate clinical forecasting system, so the prediction models
are deliberately simple. Two models are run from the start (Task 2
model-agnosticism): ridge regression and gradient boosting.

Nonconformity scores are out-of-fold from the start (Task 1 / Task 3): each
encounter is scored by a model that never saw it in training.

DESIGN NOTE (why the unit of analysis is a patient cohort, not an encounter).
An earlier version of this script regressed the raw 0/1 readmission indicator
scaled to 0-100. That construction is degenerate for this paper's question: the
nonconformity score of a rare unit is then |p - 100| ~ 100 by arithmetic, so
"more rare units in the calibration set" and "wider score support" become the
same variable (observed Spearman rho = 0.68-0.72 between them), and the two
competing predictors cannot be told apart. The recession testbed does not have
this problem because its target is a continuous probability whose residuals
vary within rare months.

To restore the comparison, encounters are aggregated into cohorts (grouped by
admission characteristics) and the target is the cohort's readmission RATE in
percent - a continuous, bounded quantity directly analogous to the recession
probability series. Rare-event cohorts are those with a high readmission rate.
Diversity and rare-count are then free to vary independently.

Outputs:
  fix-reg/domain_healthcare_ablation_{model}.csv     fixed-size ablation
  fix-reg/domain_healthcare_sweep_{model}.csv        200 random draws
  fix-reg/domain_healthcare_summary.csv              cross-model summary rows
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, support_width, fixed_size_ablation,
    random_draw_sweep, sweep_diagnostics, wilson_ci_from_indicator,
    block_bootstrap_ci, GAMMA_DEFAULT,
)

DATA = os.environ.get("HEALTHCARE_CSV", "data/domains/uci_diabetes_130.csv")
OUT = "fix-reg"
SEED = 17
N_CAL = 254          # same calibration size as the recession testbed
N_TEST = 400         # held-out evaluation stream
N_DRAWS = 200
POOL_CAP = 12000     # subsample the 100k encounters for a tractable pool

os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)

print("=" * 74)
print("DOMAIN 2 — HEALTHCARE (30-day readmission), UCI Diabetes 130-US Hospitals")
print("=" * 74)

df = pd.read_csv(DATA, low_memory=False)
df = df.replace("?", np.nan)
print(f"  Encounters loaded: {len(df):,}")

# ── Target: rare adverse event = readmission within 30 days ───────────────────
y_rare = (df["readmitted"] == "<30").astype(int).values
print(f"  Rare-event base rate: {100 * y_rare.mean():.2f}%  ({y_rare.sum():,} events)")

# ── Features: a compact, clinically sensible, low-effort feature set ──────────
num_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses"]
cat_cols = ["race", "gender", "age", "admission_type_id",
            "discharge_disposition_id", "admission_source_id",
            "A1Cresult", "max_glu_serum", "insulin", "change", "diabetesMed"]

df["_rare"] = y_rare

# ── Aggregate encounters into cohorts → continuous readmission-rate target ────
# Cohorts are defined by coarse admission/patient characteristics; each cohort's
# target is its 30-day readmission rate in percent. Cohorts need a minimum
# number of encounters so the rate is not pure noise.
COHORT_KEYS = ["age", "admission_type_id", "discharge_disposition_id",
               "admission_source_id", "medical_specialty"]
MIN_ENCOUNTERS = 25

df["_cohort"] = df[COHORT_KEYS].fillna("NA").astype(str).agg("|".join, axis=1)
grp = df.groupby("_cohort")
cohort = pd.DataFrame({
    "n_enc": grp.size(),
    "readmit_rate": grp["_rare"].mean() * 100.0,
})
for c in num_cols:
    cohort[c] = grp[c].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
# A few coarse case-mix shares as additional predictors.
for c in ["change", "diabetesMed", "insulin", "A1Cresult"]:
    d = pd.get_dummies(df[c].astype(str), prefix=c, dtype=float)
    d["_cohort"] = df["_cohort"].values
    cohort = cohort.join(d.groupby("_cohort").mean())

cohort = cohort[cohort["n_enc"] >= MIN_ENCOUNTERS].copy()
cohort = cohort.fillna(0.0)
print(f"  Cohorts (>= {MIN_ENCOUNTERS} encounters): {len(cohort):,}")

y_rate = cohort["readmit_rate"].values.astype(float)
X_all = cohort.drop(columns=["readmit_rate"]).values.astype(float)

# Rare-event cohort = high readmission rate (top decile of the cohort target),
# the analog of a recession month in the macro testbed.
RARE_THRESH = float(np.percentile(y_rate, 90))
is_rare_all = y_rate >= RARE_THRESH
print(f"  Readmission rate: mean={y_rate.mean():.2f}%  p90={RARE_THRESH:.2f}%  "
      f"max={y_rate.max():.2f}%")
print(f"  Rare-event cohorts: {is_rare_all.sum():,} ({100*is_rare_all.mean():.1f}%)")

# ── Split pool / test stream ──────────────────────────────────────────────────
order = rng.permutation(len(y_rate))
X_all, y_rate, is_rare_all = X_all[order], y_rate[order], is_rare_all[order]

N_TEST_EFF = min(N_TEST, max(60, len(y_rate) // 4))
test_idx = np.arange(len(y_rate) - N_TEST_EFF, len(y_rate))
pool_idx = np.arange(0, len(y_rate) - N_TEST_EFF)
X_pool, y_pool_pct = X_all[pool_idx], y_rate[pool_idx]
X_test, y_test_pct = X_all[test_idx], y_rate[test_idx]
rare_pool = is_rare_all[pool_idx]
N_CAL_EFF = min(N_CAL, int(0.6 * len(pool_idx)))
print(f"  Calibration pool: {len(y_pool_pct):,} cohorts ({rare_pool.sum()} rare) | "
      f"Test stream: {len(y_test_pct):,} | N_cal={N_CAL_EFF}")

# Task 2 (model-agnosticism): a simple linear baseline and a nonlinear model.
MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "gradboost": lambda: HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.06, max_depth=4, random_state=SEED),
}

scaler = StandardScaler().fit(X_pool)
Xp_s, Xt_s = scaler.transform(X_pool), scaler.transform(X_test)

# Exported for downstream reuse (Task 7 baseline horse race) so that the
# baseline comparison runs on exactly these scores rather than a re-derivation.
SCORES = {}
PREDS_TEST = {}
Y_TEST = y_test_pct
RARE_POOL = rare_pool

summary_rows = []
for name, factory in MODELS.items():
    print("\n" + "-" * 74)
    print(f"MODEL: {name}")
    print("-" * 74)
    Xp = Xp_s if name == "ridge" else X_pool
    Xt = Xt_s if name == "ridge" else X_test

    # ── Out-of-fold scores over the calibration pool (no in-sample leakage) ───
    oof = np.full(len(y_pool_pct), np.nan)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in kf.split(Xp):
        m = factory()
        m.fit(Xp[tr], y_pool_pct[tr])
        oof[te] = m.predict(Xp[te])

    # Test predictions come from a model fit on the whole pool (never on test).
    m_full = factory()
    m_full.fit(Xp, y_pool_pct)
    pred_test = m_full.predict(Xt)

    mae = float(np.nanmean(np.abs(oof - y_pool_pct)))
    print(f"  OOF MAE on readmission rate: {mae:.2f} pts")

    scores_all = np.abs(oof - y_pool_pct)          # out-of-fold nonconformity
    is_rare = rare_pool
    SCORES[name] = scores_all
    PREDS_TEST[name] = pred_test
    print(f"  OOF nonconformity: mean={np.nanmean(scores_all):.2f} "
          f"support(p95-p5)={support_width(scores_all):.2f}")

    # ── Fixed-size ablation: vary ONLY rare-event count (mirrors Table 1) ─────
    rare_counts = [0, 1, 4, 8, 16, 32, 64]
    abl = fixed_size_ablation(scores_all, is_rare, y_test_pct, pred_test,
                              rare_counts, N=N_CAL_EFF, seed=SEED)
    abl.insert(0, "domain", "Healthcare")
    abl.insert(1, "model", name)
    abl.to_csv(f"{OUT}/domain_healthcare_ablation_{name}.csv", index=False)
    print(f"\n  Fixed-size ablation (N={N_CAL_EFF}, varying rare-event count only):")
    print(abl[["Rare_Units", "Composition_pct", "Coverage",
               "Wilson_lo", "Wilson_hi", "Support_width"]].to_string(index=False))

    # ── 200 random draws: diversity vs rare-count ─────────────────────────────
    sweep = random_draw_sweep(scores_all, is_rare, y_test_pct, pred_test,
                              N=N_CAL_EFF, n_draws=N_DRAWS, seed=SEED)
    sweep.insert(0, "model", name)
    sweep.to_csv(f"{OUT}/domain_healthcare_sweep_{name}.csv", index=False)

    row = sweep_diagnostics(sweep, domain="Healthcare",
                            extra=dict(model=name, pool=len(y_pool_pct),
                                       N=N_CAL_EFF, test=len(y_test_pct),
                                       base_rate_pct=round(float(np.mean(y_rate)), 2)))
    summary_rows.append(row)
    print(f"\n  rho(support,cov)={row['rho_supp']}  R2={row['R2_supp']}  |  "
          f"rho(rare,cov)={row['rho_rare']}  R2={row['R2_rare']}")
    print(f"  within-tertile rho(rare,cov|support fixed) = "
          f"{row['within_tertile_rho_rare']}")

    # ── Headline coverage + CIs (Task 4) on the trailing-N calibration set ────
    tail = np.arange(len(y_pool_pct) - N_CAL_EFF, len(y_pool_pct))
    s_tail = scores_all[tail]
    s_tail = s_tail[~np.isnan(s_tail)]
    cov_arr, _, w_arr = run_aci(y_test_pct, pred_test, s_tail, gamma=GAMMA_DEFAULT)
    cov, w = coverage_and_width(cov_arr, w_arr)
    wlo, whi = wilson_ci_from_indicator(cov_arr)
    blo, bhi = block_bootstrap_ci(cov_arr, block=1)  # encounters are exchangeable
    print(f"  trailing-N coverage={cov:.2f}%  Wilson[{wlo:.2f},{whi:.2f}]  "
          f"bootstrap[{blo:.2f},{bhi:.2f}]  width={w:.2f}")
    summary_rows[-1].update(dict(
        trailing_cov=round(cov, 2), trailing_wilson_lo=round(wlo, 2),
        trailing_wilson_hi=round(whi, 2), trailing_boot_lo=round(blo, 2),
        trailing_boot_hi=round(bhi, 2), trailing_width=round(w, 2)))

pd.DataFrame(summary_rows).to_csv(f"{OUT}/domain_healthcare_summary.csv", index=False)
print(f"\nSaved {OUT}/domain_healthcare_summary.csv")
