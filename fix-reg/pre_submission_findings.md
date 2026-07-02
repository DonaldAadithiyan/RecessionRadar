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

## P1 — Not started

Per the task's tier ordering ("stop after each tier"), P1.1 (SHAP heatmap
export) and P1.2 (block-bootstrap CIs on Table 1) were not started — the mandate
was the P0 tier. Both are feasible from the saved model + data with no
retraining; say the word and I'll do them next.

## Files produced
- `fix-reg/pre_submission_findings.md` (this file)
- `fix-reg/aci_composition_sweep.py`, `aci_composition_sweep.csv`, `aci_composition_sweep.png`
- `supplementary/` (full package + `README.md`)
- No model/baseline code was modified (diagnosis + additive analysis only).
