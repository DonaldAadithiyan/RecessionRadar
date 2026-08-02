"""
TASK 15 — Healthcare statistical power (Study A) and headroom (Study C).

Implements task15_healthcare_power_spec.md. All decision thresholds were
pre-registered in that spec BEFORE this ran:

  "underpowered"  = at min_encounters=15 (n~214), the +3.4pp gradboost-selector
                    effect from Task 8 (90.41 -> 93.84) fails to separate.
  "low-headroom"  = under Study C's hardest admissible target, the trailing
                    baseline stays >= 88%.

STUDY A — threshold sweep. Rebuild the healthcare domain at
min_encounters in {25, 20, 15, 12, 10}. More cohorts means a longer test stream,
but noisier per-cohort rates: the binomial noise floor rises as the threshold
falls, and at min=10 it EXCEEDS the signal. Each threshold is therefore reported
with its noise/signal ratio, and the pre-registered rule is that separation
appearing only at ratio >= 1.0 is an artifact of target degradation, not a
finding.

STUDY C — headroom. Healthcare's baseline already sits at 88-90%, so even
infinite n leaves ~10pp of room, most of which would be overcoverage. Construct
harder targets and see whether the baseline can be pushed down at all.

  A2 GUARD (mandatory): a harder target must not recreate the degeneracy that
  made the original binary target unusable. For every candidate target this
  reports rho(rare-count, support-width) across the draws; > 0.5 means the two
  predictors are collinear and the target is discarded rather than reported.

Outputs:
  fix-reg/task15_healthcare_threshold_sweep.csv
  fix-reg/task15_headroom_targets.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain_common import (  # noqa: E402
    run_aci, coverage_and_width, wilson_ci_from_indicator,
    random_draw_sweep, sweep_diagnostics, GAMMA_DEFAULT,
)
import selector_lib as SEL  # noqa: E402
import augment_lib as AUG  # noqa: E402

DATA = "data/domains/uci_diabetes_130.csv"
OUT = "fix-reg"
SEED = 17
N_DRAWS = 200
CAL_FRAC = 0.60
BASE_RATE = 0.112          # observed 30-day readmission rate
THRESHOLDS = [25, 20, 15, 12, 10]

# Pre-registered (see spec)
PREREG_EFFECT_PP = 3.4     # gradboost-selector effect from Task 8
PREREG_BASE = 90.41
PREREG_HEADROOM_FLOOR = 88.0
A2_RHO_LIMIT = 0.5

NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses"]
COHORT_KEYS = ["age", "admission_type_id", "discharge_disposition_id",
               "admission_source_id", "medical_specialty"]

print("=" * 96)
print("TASK 15 — healthcare power (A) and headroom (C)")
print("=" * 96)
print(f"  Pre-registered: underpowered = +{PREREG_EFFECT_PP}pp effect fails to "
      f"separate at min=15")
print(f"                  low-headroom = baseline stays >= {PREREG_HEADROOM_FLOOR}%")
print(f"                  A2 guard     = rho(rare,support) > {A2_RHO_LIMIT} -> discard")

raw = pd.read_csv(DATA, low_memory=False).replace("?", np.nan)
raw["_rare"] = (raw["readmitted"] == "<30").astype(int)
raw["_cohort"] = raw[COHORT_KEYS].fillna("NA").astype(str).agg("|".join, axis=1)


def build_cohorts(min_enc, extra_filter=None):
    """Aggregate encounters to cohorts with a given minimum size."""
    df = raw if extra_filter is None else raw[extra_filter(raw)]
    g = df.groupby("_cohort")
    co = pd.DataFrame({"n_enc": g.size(),
                       "readmit_rate": g["_rare"].mean() * 100.0})
    for c in NUM_COLS:
        co[c] = g[c].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
    for c in ["change", "diabetesMed", "insulin", "A1Cresult"]:
        d = pd.get_dummies(df[c].astype(str), prefix=c, dtype=float)
        d["_cohort"] = df["_cohort"].values
        co = co.join(d.groupby("_cohort").mean())
    co = co[co["n_enc"] >= min_enc].fillna(0.0)
    return co


def fit_and_score(co, model="gradboost", seed=SEED):
    """Out-of-fold cohort-level scores + a held-out test stream."""
    rng = np.random.default_rng(seed)
    y = co["readmit_rate"].values.astype(float)
    X = co.drop(columns=["readmit_rate"]).values.astype(float)
    rare = y >= np.percentile(y, 90)

    order = rng.permutation(len(y))
    X, y, rare = X[order], y[order], rare[order]

    n_test = max(60, len(y) // 4)
    X_pool, y_pool, rare_pool = X[:-n_test], y[:-n_test], rare[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]
    n_cal = min(254, int(CAL_FRAC * len(y_pool)))

    if model == "ridge":
        sc = StandardScaler().fit(X_pool)
        X_pool, X_test = sc.transform(X_pool), sc.transform(X_test)
        mk = lambda: Ridge(alpha=1.0)          # noqa: E731
    else:
        mk = lambda: HistGradientBoostingRegressor(  # noqa: E731
            max_iter=200, learning_rate=0.06, max_depth=4, random_state=seed)

    oof = np.full(len(y_pool), np.nan)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X_pool):
        m = mk(); m.fit(X_pool[tr], y_pool[tr])
        oof[te] = m.predict(X_pool[te])
    m_full = mk(); m_full.fit(X_pool, y_pool)
    pred_test = m_full.predict(X_test)

    return np.abs(oof - y_pool), rare_pool, y_test, pred_test, n_cal


def evaluate(scores, rare_pool, y_test, pred_test, n_cal):
    """Baseline / selector / augmentation coverage with Wilson intervals."""
    v = np.where(np.isfinite(scores))[0]
    trailing = scores[v[-n_cal:]]
    out = {}
    for name, cal in [
        ("trailing", trailing),
        ("selector", scores[SEL.support_width_selector(scores, n_cal)]),
        ("augmented", AUG.augment_pool(trailing, synth_frac=0.20, seed=3)[0]),
    ]:
        c = np.asarray(cal, float); c = c[np.isfinite(c)]
        cov_arr, _, w_arr = run_aci(y_test, pred_test, c, gamma=GAMMA_DEFAULT)
        cov, w = coverage_and_width(cov_arr, w_arr)
        lo, hi = wilson_ci_from_indicator(cov_arr)
        out[name] = dict(cov=round(cov, 2), lo=round(lo, 2), hi=round(hi, 2),
                         w=round(w, 3))
    return out


# ── STUDY A ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 96)
print("STUDY A — cohort-threshold sweep")
print("-" * 96)

a_rows = []
for mn in THRESHOLDS:
    co = build_cohorts(mn)
    rate_sd = float(co["readmit_rate"].std())
    noise = float(np.sqrt(BASE_RATE * (1 - BASE_RATE) / mn) * 100)
    ns = noise / rate_sd if rate_sd else np.inf

    for model in ["ridge", "gradboost"]:
        scores, rare_pool, y_te, p_te, n_cal = fit_and_score(co, model)
        res = evaluate(scores, rare_pool, y_te, p_te, n_cal)
        sweep = random_draw_sweep(scores, rare_pool, y_te, p_te, N=n_cal,
                                  n_draws=N_DRAWS, seed=SEED)
        diag = sweep_diagnostics(sweep, domain=f"hc_min{mn}")
        rho_collin = float(st.spearmanr(sweep["rare_count"],
                                        sweep["supp"]).correlation)

        b, s, a = res["trailing"], res["selector"], res["augmented"]
        a_rows.append(dict(
            min_enc=mn, model=model, n_cohorts=len(co), n_test=len(y_te),
            N_cal=n_cal, rate_sd=round(rate_sd, 3), noise_floor=round(noise, 3),
            noise_signal=round(ns, 3),
            rho_supp=diag["rho_supp"], rho_rare=diag["rho_rare"],
            gap=round(diag["rho_supp"] - diag["rho_rare"], 3),
            rho_collinearity=round(rho_collin, 3),
            base_cov=b["cov"], base_lo=b["lo"], base_hi=b["hi"],
            sel_cov=s["cov"], sel_lo=s["lo"], sel_hi=s["hi"],
            sel_sep=bool(s["lo"] > b["cov"]),
            aug_cov=a["cov"], aug_lo=a["lo"], aug_hi=a["hi"],
            aug_sep=bool(a["lo"] > b["cov"]),
            sel_w_mult=round(s["w"] / b["w"], 2) if b["w"] else np.nan))
        r = a_rows[-1]
        print(f"  min={mn:3d} {model:10s} n={r['n_test']:4d} "
              f"noise/signal={r['noise_signal']:.2f} | "
              f"base={r['base_cov']:6.2f} sel={r['sel_cov']:6.2f}"
              f"{'*' if r['sel_sep'] else ' '} "
              f"aug={r['aug_cov']:6.2f}{'*' if r['aug_sep'] else ' '} | "
              f"gap={r['gap']:+.3f}")

A = pd.DataFrame(a_rows)
A.to_csv(f"{OUT}/task15_healthcare_threshold_sweep.csv", index=False)

# ── STUDY C ─────────────────────────────────────────────────────────────────
print("\n" + "-" * 96)
print("STUDY C — headroom: can the baseline be pushed below 88%?")
print("-" * 96)
print("  Each candidate target is checked against the A2 collinearity guard")
print("  BEFORE its headroom is reported.")

# Admissible targets filter on PRE-SPECIFIED COVARIATES, never on the outcome
# itself, so selection and outcome do not share a term (spec guard #3).
TARGETS = {
    "baseline (all cohorts, min=25)":
        (25, None),
    "emergency admissions only":
        (25, lambda d: d["admission_type_id"].isin([1])),
    "long stays (>=5 days)":
        (25, lambda d: pd.to_numeric(d["time_in_hospital"],
                                     errors="coerce") >= 5),
    "high prior utilisation (>=1 inpatient)":
        (25, lambda d: pd.to_numeric(d["number_inpatient"],
                                     errors="coerce") >= 1),
}

c_rows = []
for label, (mn, filt) in TARGETS.items():
    try:
        co = build_cohorts(mn, extra_filter=filt)
    except Exception as e:
        print(f"  [skip] {label}: {e}")
        continue
    if len(co) < 80:
        print(f"  [skip] {label}: only {len(co)} cohorts after filtering")
        continue

    scores, rare_pool, y_te, p_te, n_cal = fit_and_score(co, "gradboost")
    sweep = random_draw_sweep(scores, rare_pool, y_te, p_te, N=n_cal,
                              n_draws=N_DRAWS, seed=SEED)
    rho_collin = float(st.spearmanr(sweep["rare_count"],
                                    sweep["supp"]).correlation)
    degenerate = abs(rho_collin) > A2_RHO_LIMIT

    res = evaluate(scores, rare_pool, y_te, p_te, n_cal)
    b = res["trailing"]
    c_rows.append(dict(target=label, n_cohorts=len(co), n_test=len(y_te),
                       mean_rate=round(float(co["readmit_rate"].mean()), 2),
                       rate_sd=round(float(co["readmit_rate"].std()), 2),
                       rho_collinearity=round(rho_collin, 3),
                       A2_degenerate=bool(degenerate),
                       base_cov=b["cov"], base_lo=b["lo"], base_hi=b["hi"],
                       below_88=bool(b["cov"] < PREREG_HEADROOM_FLOOR)))
    r = c_rows[-1]
    flag = "  <-- A2 DEGENERATE, DISCARD" if degenerate else ""
    print(f"  {label:42s} n={r['n_test']:4d} rate={r['mean_rate']:5.2f}% "
          f"base={r['base_cov']:6.2f}% rho(rare,supp)={rho_collin:+.3f}{flag}")

C = pd.DataFrame(c_rows)
C.to_csv(f"{OUT}/task15_headroom_targets.csv", index=False)

# ── VERDICT against the pre-registered criteria ─────────────────────────────
print("\n" + "=" * 96)
print("VERDICT — evaluated against the criteria fixed before this ran")
print("=" * 96)

safe = A[A["noise_signal"] < 0.95]
print(f"\n  Thresholds with noise/signal < 0.95 (evidence-grade): "
      f"{sorted(safe['min_enc'].unique().tolist())}")
print(f"  Thresholds with noise/signal >= 1.0 (artifact-prone): "
      f"{sorted(A[A['noise_signal'] >= 1.0]['min_enc'].unique().tolist())}")

sep_safe = safe[(safe["sel_sep"]) | (safe["aug_sep"])]
sep_unsafe = A[(A["noise_signal"] >= 1.0) & ((A["sel_sep"]) | (A["aug_sep"]))]
print(f"\n  Separations at evidence-grade thresholds: {len(sep_safe)}")
if len(sep_safe):
    print(sep_safe[["min_enc", "model", "n_test", "base_cov", "sel_cov",
                    "sel_sep", "aug_cov", "aug_sep"]].to_string(index=False))
print(f"  Separations ONLY at artifact-prone thresholds: {len(sep_unsafe)} "
      f"(these do NOT count as findings)")

m15 = A[(A["min_enc"] == 15) & (A["model"] == "gradboost")]
if len(m15):
    r = m15.iloc[0]
    underpowered = not (r["sel_sep"] or r["aug_sep"])
    print(f"\n  UNDERPOWERED test (pre-registered): at min=15 gradboost, "
          f"n={int(r['n_test'])}")
    print(f"    baseline={r['base_cov']}  selector={r['sel_cov']} "
          f"[{r['sel_lo']},{r['sel_hi']}] separates={r['sel_sep']}")
    print(f"    -> underpowered = {underpowered}")
else:
    underpowered = None

valid_c = C[~C["A2_degenerate"]] if len(C) else C
low_head = bool(len(valid_c) and (valid_c["base_cov"] >= PREREG_HEADROOM_FLOOR).all())
print(f"\n  LOW-HEADROOM test (pre-registered): does any admissible target push "
      f"the baseline below {PREREG_HEADROOM_FLOOR}%?")
if len(valid_c):
    print(valid_c[["target", "base_cov", "below_88"]].to_string(index=False))
print(f"    -> low-headroom = {low_head}")

print("\n  DECISION (per the spec's committed table):")
if underpowered and not low_head:
    print("    underpowered=yes, low-headroom=no -> Study B is warranted")
elif underpowered and low_head:
    print("    both -> headroom binds first; do NOT run Study B, "
          "report as low-headroom")
elif not underpowered:
    print("    not underpowered -> Study A resolved it; no Study B")
print(f"\nSaved {OUT}/task15_healthcare_threshold_sweep.csv, "
      f"task15_headroom_targets.csv")
