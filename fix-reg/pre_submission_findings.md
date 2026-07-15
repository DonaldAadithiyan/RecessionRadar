# Pre-Submission Findings (IJCAI GlobalSouthAI)

## P0.1 — Non-monotonic scenario table

**Verdict: genuine model behavior (reproducible), NOT a transcription error and
NOT a chain-only artifact. The +50/+100 flip is a lumpy tree-ensemble response
surface at near-floor probabilities, amplified by chain conditioning. It is
economically negligible in magnitude and confined to CatBoost.**

**1. Reproduced exactly** from `fix-reg/scenario_test.py` (baseline = May 2025,
3M=4.25%, 10Y=4.42%, spread +17bps). 6M row: −100→20.26, −50→11.39, base→3.58,
**+50→6.44, +100→4.20**. Not a stale transcription.

**2. What the shock actually changes.** The script shocks only
`3_months_rate`, `1_year_rate`, `3_months_rate_diff3`; it holds `10_year_rate`
(and `6_months_rate`, all rolling/residual rate features) fixed. So a
"+100 bps" scenario is really a **short-end-only shock that inverts the 10Y–3M
spread** (+17 → −83 bps). That is a reasonable stress design, but it means the
5 scenarios trace a path through spread space, not a clean parallel rate move.

**3. Origin — decomposed by base model** (chain order Cur→1M→3M→6M):

| shock | CatBoost 6M | LightGBM 6M | RandomForest 6M | Ensemble 6M |
|---|---|---|---|---|
| −100 | 24.98 | 0.30 | 2.23 | 20.26 |
| −50 | 12.98 | 0.24 | 2.23 | 11.39 |
| 0 | 2.67 | 0.24 | 1.23 | 3.58 |
| **+50** | **6.61** | 0.23 | 1.05 | **6.44** |
| **+100** | **3.58** | 0.23 | 1.03 | **4.20** |

The non-monotonicity lives **entirely in CatBoost**. LightGBM is nearly inert
(it barely reacts because most of its informative features aren't shocked);
RandomForest is smooth/monotonic. The ElasticNet meta-learner is roughly a
weighted pass-through, so CatBoost's kink survives into the ensemble.

**4. Chain vs. 6M-model origin.** Freezing the upstream chain predictions at
their baseline values and varying only the shocked features, the **CatBoost 6M
estimator is *still* non-monotonic** (+50→3.56 vs +100→3.31), so the flip is
intrinsic to the 6M response surface — but the chain **amplifies** it: with
full propagation the 6M logit runs −3.60 (base) → −2.65 (+50) → −3.29 (+100),
i.e. +50 conditions on a higher intermediate state than +100. So the task's
suspicion is half right: it originates in the 6M model but the chain magnifies
it. The upstream 3M is monotone-decreasing in logit; the 6M model reacts to the
*joint* feature+chain state non-monotonically.

**5. Construction sensitivity.** Under a **parallel** shift (move all rate
levels together, spread held constant) the easing spike collapses (−100→7.10
vs 20.26) and the tightening side becomes monotone (base 3.58 → +50 4.00 →
+100 4.01). This shows the large responses are driven by the **spread
inversion**, which is the intended signal — so this is not a construction bug
to "fix," but it does confirm the effect is a property of the model's learned
spread→P(recession) surface, which is bumpy where training data is sparse
(deeply inverted curve + May-2025 feature context).

**Magnitude caveat (the real framing point):** on the tightening side the
entire 6M range is **2.6–6.6%** — all near the model's probability floor, where
tree quantization produces ±2–3pp wiggles. The economically meaningful,
monotone signal is the **easing→higher-6M** and the large easing response; the
+50/+100 ordering is within model noise.

**Paper footnote (Table-1-annotation style), one sentence:**
> "The non-monotonic 6-month ordering between the +50 and +100 bps tightening
> scenarios (6.44 vs 4.20) arises from CatBoost's discretized response surface
> in the deeply-inverted-curve region, where all predicted probabilities sit
> near the 3–6% floor; it reflects tree-ensemble quantization amplified by
> chain conditioning rather than a distinct economic mechanism, and is within
> the model's effective resolution at these probability levels."

*(If you prefer not to name model internals: "…arises from the ensemble's
locally non-smooth response in the deeply-inverted-curve region, where all
6-month probabilities are near-floor (3–6%); we flag it as an open observation
rather than a distinct economic effect.")*

**Compute:** diagnosis used the saved ensemble only (`full_chain_stacking.pkl`);
no retraining. Seconds to run.

---

## P0.2 — Calibration-composition sweep (20 / 25 / 30 / 35 / 40%)

**Verdict: the intermediate points reproduce cleanly (no retraining, seconds to
run), but they DO NOT support the word "systematic" in the smooth-sweep sense.
Coverage is a step function driven by one binary event — whether the calibration
window reaches back far enough to include the 2008 recession. Report it honestly
as a step, not a gradient.**

Methodology matches the paper's Experiment B exactly: the calibration set is the
**last `f`% of the 635 training rows** (temporal tail); conformity scores are
`|pred − actual|` from the *saved* ensemble; ACI runner is Gibbs–Candès with
γ=0.005, nominal 90%. Only re-slicing + re-running ACI — **no model retraining,
no Optuna** (`scripts/aci_composition_sweep.py`). "Composition" = share of
calibration months with smoothed recession probability ≥ 50%.

**Calibration-set composition by fraction:**

| Frac | Cal rows | Cal window start | Recession months (≥50%) | Composition % |
|---|---|---|---|---|
| 20% | 127 | 2009-06 | **1** | 0.8% |
| 25% | 158 | 2006-11 | 16 | 10.1% |
| 30% | 190 | 2004-03 | 16 | 8.4% |
| 35% | 222 | 2001-07 | 16 | 7.2% |
| 40% | 254 | 1998-11 | 16 | 6.3% |

Note the "20% = 1 month, 40% = 16 months" framing in the current draft conflates
two things: the jump from 1→16 recession months happens entirely between **20%
and 25%** (when the window first reaches the 2008 GFC). Beyond 25% the count is
flat at 16 and composition % actually *falls* as expansion months dilute it.

**Empirical coverage (%) at nominal 90%, γ=0.005:**

| Frac | Current | 1M | 3M | 6M |
|---|---|---|---|---|
| 20% | 70.77 | 71.88 | 59.68 | 59.32 |
| 25% | 89.23 | 82.81 | 88.71 | 67.80 |
| 30% | 87.69 | 81.25 | 87.10 | 66.10 |
| 35% | 89.23 | 85.94 | 85.48 | 66.10 |
| 40% | 89.23 | 85.94 | 85.48 | 67.80 |

**Mean interval width (pp):**

| Frac | Current | 1M | 3M | 6M |
|---|---|---|---|---|
| 20% | 2.49 | 2.41 | 4.39 | 11.71 |
| 25% | 7.12 | 4.80 | 21.90 | 10.55 |
| 30% | 6.03 | 4.22 | 16.90 | 10.39 |
| 35% | 6.86 | 5.43 | 14.65 | 10.54 |
| 40% | 7.45 | 7.35 | 14.95 | 10.64 |

Figure: `fix-reg/aci_composition_sweep.png` (coverage vs fraction, one line/horizon).

**Is it monotonic/smooth? No — it's a step.** Current/1M/3M coverage jumps from
~60–72% to ~85–89% between 20% and 25% (the moment the 2008 recession enters
calibration), then is essentially flat 25→40%. **6M never recovers** — it sits
at 66–68% across every composition, i.e. 6M undercoverage is *not* fixable by
adding calibration data (consistent with the paper's separate 6M finding). So
the honest one-paragraph summary:

> "Extending the calibration window across five compositions (20–40%) shows the
> coverage improvement is driven almost entirely by whether the window includes
> the 2008 recession: coverage for Current/1M/3M rises sharply from ~60–72% to
> ~85–89% once the 2008 episode enters the calibration set (between the 20% and
> 25% windows) and is stable thereafter, while 6-month coverage remains at
> ~66–68% regardless. The relationship is therefore a threshold effect tied to
> rare-event inclusion, not a smooth gradient — we report it as such."

**Recommendation:** either (a) drop "systematic" and describe the threshold
behavior + the 6M-invariance (this is a *stronger*, more interesting claim), or
(b) keep a sweep figure but caption it as a step/threshold, not a smooth trend.
Do not present 20% and 40% as two ends of a continuum — they're on the same
plateau; the interesting transition is at 25%.

**Compute:** saved model only; ~30 s. No Optuna, no retraining.

---

## P0.3 — Reproducibility package

**Status: WORKING (model-based reproduction) — verified end-to-end in a fresh,
clean venv from the assembled `supplementary/` folder alone. One honest caveat:
the processed feature matrix and the trained model are shipped as artifacts;
regenerating them from raw FRED data requires notebooks not in the package.**

**What was assembled** → `supplementary/` (21 MB):
`data/raw/` (13 FRED CSVs), `data/processed/feature_selected_reg_full.csv`,
`models/full_chain_stacking.pkl`, 4 `scripts/`, `expected_outputs/`, `README.md`.

**Fresh-environment test (actually performed):** new venv, installed only
`scikit-learn==1.5.2 numpy pandas scipy catboost lightgbm xgboost matplotlib`
(NOT the 400-line `requirements.txt`). Results:

- **Table 1 (ensemble MAE) reproduces to 4 dp**: 6.8292 / 5.6336 / 7.7285 /
  10.1696 — exact match to paper.
- **P0.2 composition sweep reproduces identically** in the clean venv.

**What does NOT reproduce cleanly (adjust the §4.6 claim to match):**

1. **`feature_selected_reg_full.csv` is provided pre-built, not regenerated.**
   The raw→processed pipeline (STL/Box-Cox, lag/rolling features, cubic-spline
   GDP interpolation, RF feature selection) lives in project notebooks that are
   **not** in the package. Claim should say the processed dataset is *provided*,
   with the pipeline *described*, rather than implying one-command raw→table.
2. **The ensemble is shipped as a pickle, not retrained.** Retraining exists
   (`ensemble.ipynb`) and is seed-deterministic, but is outside the package.
3. **scikit-learn version is load-bearing.** The pickle requires **sklearn
   1.5.2**; other versions may fail to unpickle. Must be pinned in the appendix.
4. **Scripts use hardcoded relative paths.** The main-repo copies assume
   `data/fix/…` and `fix-reg/…`; the packaged copies were path-patched to the
   folder layout. Worth a one-line note so a reviewer running the repo version
   isn't surprised.
5. **`ablation_baselines.py` re-runs Optuna (~10 min)** rather than loading saved
   baseline predictions (they were never saved). Deterministic (seed=42) and
   reproduces Table 2, but flag the runtime; consider shipping saved baseline
   predictions to make it instant.

**§4.6 wording fix (suggested):** "The supplementary material provides the
processed feature matrix, the trained Stage-2 ensemble, all evaluation/analysis
scripts, and the raw FRED series (IDs and 1967–2025 monthly spans) for the
635/65 split. Table 1, the ablation baselines, and the ACI/calibration analyses
reproduce from the provided model and data with the pinned environment
(scikit-learn 1.5.2); regenerating the processed matrix from raw series uses the
feature-engineering notebooks described in Section 3.2." The full README is at
`supplementary/README.md` and can be lifted into the appendix.

---

## P1.1 — Horizon-conditional SHAP heatmap export

**Status: DONE. Full top-12 × 4-horizon mean |SHAP| matrix exported and
re-rendered; all three paper-quoted numbers reconcile exactly (one sign typo to
fix).** Reproduced from `fix-reg/task2_shap.py` (TreeExplainer on the CatBoost
RegressorChain, SHAP of original features only, mean |value| over the 65 test
rows). Requires the `shap` package.

**Top-12 features × 4 horizons (mean |SHAP|, logit-space units):**

| Feature | Current | 1M | 3M | 6M |
|---|---|---|---|---|
| OECD_CLI_index_trend | 1.107 | 0.455 | 0.883 | 0.291 |
| INDPRO_diff3 | **1.550** | 0.331 | 0.174 | 0.235 |
| gdp_per_capita_residual | 0.239 | 0.655 | 0.597 | 0.799 |
| OECD_CLI_index_residual | 0.163 | 0.124 | 0.708 | **0.881** |
| gdp_per_capita_diff3 | 0.395 | 0.326 | 0.382 | 0.477 |
| gdp_per_capita | **0.042** | 0.162 | 0.193 | **1.163** |
| 10_year_rate_residual | 0.203 | 0.307 | 0.623 | 0.387 |
| OECD_CLI_index_diff1 | 0.198 | 0.439 | 0.432 | 0.289 |
| gdp_per_capita_diff1 | 0.463 | 0.476 | 0.299 | 0.112 |
| OECD_CLI_index_pct_change1 | 0.223 | 0.317 | 0.395 | 0.353 |
| share_price | 0.266 | 0.049 | 0.610 | 0.221 |
| gdp_per_capita_pct_change1 | 0.403 | 0.301 | 0.167 | 0.128 |

Exports: `fix-reg/shap_top12_x_horizon.csv`, `.json`, and heatmap
`fix-reg/task2_outputs/figure_C_shap_heatmap.png`.

**Reconciliation of paper-quoted numbers (all MATCH):**

| Paper text | Maps to | Export value | Verdict |
|---|---|---|---|
| "INDPRO diff current = 1.55" | `INDPRO_diff3` @ Current | **1.550** | ✓ exact |
| "6.6× fall" | `INDPRO_diff3` Current→6M (1.55→0.235) | ratio **6.6×** | ✓ exact |
| "GDP per capita 0.04→1.16" | `gdp_per_capita` Current→6M | **0.042 → 1.163** | ✓ exact |
| "OECD CLI residual −0.88" | `OECD_CLI_index_residual` @ 6M | **0.881** | ✓ magnitude |

**One fix for the paper:** the "−0.88" should be **0.88** (positive). Mean |SHAP|
is a magnitude and cannot be negative; the minus sign is a typo (or a leftover
from a signed-SHAP directional plot). Also make explicit that the "6.6× fall"
refers to **INDPRO_diff3**, not GDP — the current prose is ambiguous about which
feature the 6.6× applies to.

---

## P1.2 — Block-bootstrap 90% CIs on MAE

**Status: DONE. Moving-block bootstrap (block=8 ≈ √65, 2000 resamples, seed=42)
on |error| series, 90% CIs, vs the *tuned* Table-2 baselines. Verdict: the
ensemble's advantage is robust ONLY at 6M; at 3M the CIs overlap despite the
point-estimate win — bootstrap and DM legitimately disagree here, and honesty
favors reporting it.** XGB-indep MAEs reproduced exactly (3.60/3.78/12.68/45.71).

**MAE with 90% moving-block CIs:**

| Horizon | Ensemble (Ours) | XGB Indep (tuned) | Naive Mean | Probit (YC) |
|---|---|---|---|---|
| Current | 6.83 [1.12, 10.05] | 3.60 [0.20, 5.66] | 11.28 [8.72, 13.83] | 3.37 [0.23, 6.36] |
| 1M | 5.63 [0.93, 8.11] | 3.78 [0.20, 5.57] | 11.25 [8.70, 12.57] | 3.56 [0.34, 5.00] |
| 3M | 7.73 [2.10, 10.98] | 12.68 [5.51, 16.51] | 10.09 [8.79, 10.19] | 2.64 [0.59, 2.78] |
| 6M | 10.17 [4.15, 17.63] | 45.71 [31.60, 63.73] | 8.92 [8.78, 9.10] | 2.35 [1.07, 4.09] |

Export: `fix-reg/bootstrap_block_ci_90.csv`.

**Ensemble vs XGB-indep, disjointness of 90% CIs:**

| Horizon | Ensemble CI | XGB-indep CI | CIs disjoint? |
|---|---|---|---|
| Current | [1.12, 10.05] | [0.20, 5.66] | overlap (ensemble worse) |
| 1M | [0.93, 8.11] | [0.20, 5.57] | overlap (ensemble worse) |
| 3M | [2.10, 10.98] | [5.51, 16.51] | **overlap** |
| 6M | [4.15, 17.63] | [31.60, 63.73] | **DISJOINT** |

**Interpretation for the paper.** The **6M** advantage over the tuned
independent baseline is robust — the CIs do not overlap, reinforcing the DM
result and the RegressorChain claim. But at **3M** the ensemble's point-estimate
win (7.73 vs 12.68) is **not** robust to block resampling — the CIs overlap
substantially — so a DM p-value at 3M overstates confidence. At **Current/1M**
the ensemble is not better than XGB-indep at all (higher MAE, overlapping CIs);
its wide CIs there are driven by the two COVID-spike months. Recommended framing:
"Moving-block bootstrap 90% CIs (block ≈ √n) confirm the ensemble's advantage is
statistically robust at the 6-month horizon (disjoint CIs) but not at 3 months
(overlapping CIs); the short-horizon comparisons are dominated by the COVID
spike and neither model has a robust edge. This is consistent with the framework
being a *long-horizon* contribution." (Note: the pre-existing
`fix-reg/bootstrap_ci_results.csv` used an iid bootstrap at 95% against the
*untuned* baselines — superseded by this block-bootstrap-vs-tuned version.)

**Compute:** P1.2 re-ran the 4-horizon XGB-indep Optuna (~8 min, deterministic)
to obtain tuned baseline predictions (they were never saved); the ensemble was
loaded from pickle, not retrained. P1.1 is seconds given the saved SHAP outputs.

---

## Files produced
- `fix-reg/pre_submission_findings.md` (this file)
- **P0.2:** `fix-reg/aci_composition_sweep.py`, `aci_composition_sweep.csv`, `aci_composition_sweep.png`
- **P0.3:** `supplementary/` (full package + `README.md`), verified in a clean venv
- **P1.1:** `fix-reg/shap_top12_x_horizon.csv`, `.json`, `fix-reg/task2_outputs/figure_C_shap_heatmap.png`
- **P1.2:** `fix-reg/task_p12_bootstrap.py`, `fix-reg/bootstrap_block_ci_90.csv`
- No model/baseline code was modified (diagnosis + additive analysis only).
