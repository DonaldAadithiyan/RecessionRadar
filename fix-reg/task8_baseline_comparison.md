# Task 8 — Closing the Domain-Literature Baseline Gap

Healthcare and climate now have the same two-part structure the recession domain
has: (a) a point-prediction benchmark against the domain's own published
approaches, (b) the 8-strategy calibration comparison from Task 7.

**Headline: the calibration-diversity mechanism is a property of the score
distribution, not of the model that produced it.** The diversity-optimal
selector's gain over pooled/trailing ACI is essentially unchanged whether the
underlying scores come from a 2010 fixed-weight clinical index or a gradient
boosting regressor — +4.8pp vs +3.4pp in healthcare, +8.3–9.1pp vs +7.8–8.6pp in
climate. This is the claim Table 4 already makes; it now holds against two more
model families per domain.

Scripts: `fix-reg/task8_healthcare_literature.py`,
`fix-reg/task8_climate_literature.py`.
Data: `task8_{healthcare,climate}_literature_baselines.csv`,
`task8_{healthcare,climate}_point_prediction.csv`.
Figure: `figures/task8_model_independence.{pdf,png}`.
Paper tables: `fix-reg/task8_paper_tables.md`.

---

## Healthcare — LACE index and HOSPITAL score

### Field availability, audited before implementation

**LACE** (van Walraven et al., *CMAJ* 2010;182(6):551–557) — all four components
derivable from the existing UCI extract:

| Component | Source field | Published scoring |
|---|---|---|
| **L** ength of stay | `time_in_hospital` | 1d=1, 2=2, 3=3, 4–6=4, 7–13=5, ≥14=7 |
| **A** cuity of admission | `admission_source_id`=7 (ED) or `admission_type_id`=1 | 3 points |
| **C** omorbidity | Charlson index computed from `diag_1..diag_3` ICD-9 codes | CCI 0–3 → 0–3, ≥4 → 5 |
| **E** D visits prior 6mo | `number_emergency` | 1–4 visits → 1–4 (capped) |

**HOSPITAL** (Donzé et al., *JAMA Intern Med* 2013;173(8):632–638) — **5 of 7
components available. This is a substitution, and the reconstruction is not the
published score.**

| Component | Status | Handling |
|---|---|---|
| **H** emoglobin <12 g/dL | **ABSENT** | scored 0 |
| **O** ncology discharge | available (`medical_specialty`, 559 encounters) | 2 points |
| **S** odium <135 mEq/L | **ABSENT** | scored 0 |
| **P** rocedure during stay | available (`num_procedures`>0) | 1 point |
| **I** ndex admission urgent/emergent | available (`admission_type_id`∈{1,2}) | 1 point |
| **A** dmissions prior year | available (`number_inpatient`) | 0–1=0, 2–5=2, >5=5 |
| **L** ength of stay ≥5d | available (`time_in_hospital`) | 2 points |

This dataset carries no lab-value columns beyond `max_glu_serum` and
`A1Cresult`, so hemoglobin and sodium cannot be reconstructed by any means.
Both are scored 0, the documented handling for an unavailable component, which
is conservative — it can only lower a patient's score. **Maximum attainable
score falls from 13 to 11.** Likely effect on validity: both missing components
are among the weaker predictors in the original derivation (1 point each of 13),
but their absence removes the score's only physiological signal, leaving a
purely utilisation-based index. Every HOSPITAL number below should be read as
"5-component HOSPITAL variant", not as the published score.

### AUC reproduction check — passed, with an important nuance

The task specified checking against published figures on this dataset before
building calibration results on top. The published ~0.608 reference is a
**logistic regression fit on LACE components**, not the fixed-weight LACE index,
so both were computed:

| Quantity | AUC-ROC | Reference |
|---|---|---|
| LACE components, fitted logistic | **0.5820** | published ≈0.608 |
| LACE index, published fixed weights | **0.5691** | — |
| HOSPITAL (5-component variant) | **0.5875** | — |

**PASS.** The fitted comparator (0.582) lands near the published 0.608, and the
fixed-weight index sits just below it — exactly the expected ordering, since
published weights derived on a different population cannot adapt to this one.

**Why both fall short of LACE's original 0.684 validation** — measured, not
asserted. Per-component discrimination on this dataset:

| Component | AUC alone |
|---|---|
| L: length of stay | 0.5463 |
| C: Charlson index | 0.5384 |
| E: ED visits | 0.5318 |
| A: acute admission | 0.5131 |
| *(not a LACE component)* prior inpatient admissions | **0.6061** |

Every LACE component is individually weak here, and the single strongest
predictor in this dataset — prior inpatient admissions — **is not a LACE
component at all**. No fixed-weight combination of weak components can exceed
what those components carry. Two contributing factors: the ED-visit field is a
prior-**year** count, not LACE's prior-6-months, and the Charlson index is built
from three coded diagnoses rather than a full problem list.

### Point-prediction results (mirrors Table 6)

| Model | Origin | AUC (binary) | OOF MAE (cohort rate, pts) |
|---|---|---|---|
| LACE | van Walraven 2010 | 0.5691 | 4.900 |
| HOSPITAL (5-comp) | Donzé 2013 | 0.5875 | 4.776 |
| ridge | internal check | — | 4.51 |
| gradboost | internal check | — | 4.97 |

The literature scores are competitive with the internal models on the continuous
target — LACE's 4.90 sits between ridge (4.51) and gradboost (4.97) — despite
being fixed-weight indices designed for a different task.

### Calibration comparison (n=146 test cohorts)

| Strategy | LACE | HOSPITAL | ridge | gradboost |
|---|---|---|---|---|
| Pooled/trailing ACI | 86.99 | 86.99 | 88.36 | 90.41 |
| Mondrian | 86.99 | 91.10 | 91.78 | 91.78 |
| PID-conformal | 87.67 | 86.99 | 88.36 | 90.41 |
| EVT-tail | 86.99 | 86.99 | 89.04 | 90.41 |
| DtACI | 86.99 | 86.99 | 88.36 | 89.04 |
| AcMCP | 78.08 | 81.51 | 80.82 | 89.73 |
| Bellman CI | 71.23 | 68.49 | 69.18 | 66.44 |
| **Diversity-optimal (ours)** | **91.78** | **91.78** | **92.47** | **93.84** |

**Verdict: no strategy separates upward under any healthcare model.**
Diversity-optimal has the highest point estimate under all four (+4.79pp on both
literature scores), but the Wilson intervals overlap the baseline in every case.
This matches Task 7's healthcare finding and has the same cause: the baseline
already sits near nominal, so there is little headroom at n=146.

AcMCP and Bellman CI separate *downward* under the literature scores, as they did
under the internal models.

---

## Climate — persistence and climatology

### Why these baselines, and why not a tropical-cyclone model

The per-region-month storm-intensity target is a custom aggregate built for this
paper, not a standard task with an established point-prediction literature.
Persistence and climatology are the meteorological field's **own** baseline-tier
convention — used by WeatherBench (Rasp et al., 2020, *JAMES* 12(11)), TCBench,
and NOAA's hurricane guidance suite — not a novel choice made here.

A SHIPS/SHIFOR-style tropical-cyclone intensity model was **not** attempted:
those predict per-storm intensity from storm-centred predictors, and forcing one
onto a monthly regional aggregate would be a mismatch dressed up as a
comparison. Flagged explicitly, the same way CPTC was in Task 7.

**Leakage discipline:** the climatological normal is computed expanding over the
past only — each region-month is predicted from prior years' same-calendar-month
observations, and the current observation is added to the history only *after*
it has been predicted.

**Stability check (task guardrail):** 0 of 252 test region-months have fewer
than 5 years of history for their normal. The thin-history failure mode the spec
warned about does not occur here. Persistence loses 5 predictions (the first
month of each region's series), which are dropped and reported rather than
imputed.

### Point-prediction results

| Model | Origin | OOF MAE | Test MAE | Score support (p95−p5) |
|---|---|---|---|---|
| persistence | WeatherBench baseline tier | 3.560 | 2.515 | 10.559 |
| climatology | WeatherBench baseline tier | 3.054 | 3.032 | 8.119 |
| ridge | internal check | 2.93 | — | 7.39 |
| gradboost | internal check | 3.08 | — | 8.21 |

Climatology (3.054) is essentially level with gradboost (3.08) and slightly
behind ridge (2.93). **The internal ML models barely beat a climatological
normal on this target** — worth stating plainly in §6.4, since it bounds how
much the point-prediction layer is contributing in this domain.

### Calibration comparison (n=252 test region-months)

| Strategy | persistence | climatology | ridge | gradboost |
|---|---|---|---|---|
| Pooled/trailing ACI | 90.87 | 90.08 | 90.95 | 90.95 |
| Mondrian | 94.44 | 96.83 | 95.88 | 94.65 |
| PID-conformal | 90.48 | 89.68 | 89.71 | 88.89 |
| EVT-tail | 90.87 | 90.48 | 90.53 | 91.36 |
| DtACI | 90.08 | 90.08 | 90.12 | 90.95 |
| AcMCP | 93.25 | 89.68 | 88.89 | 88.07 |
| Bellman CI | 78.97 | 75.40 | 78.19 | 79.42 |
| **Diversity-optimal (ours)** | **99.21** | **99.21** | **99.59** | **98.77** |

**Verdict: diversity-optimal separates upward under both literature baselines**
(+8.34pp on persistence, +9.13pp on climatology; Wilson lower bounds 97.15 vs
baselines of 90.87/90.08). Mondrian also separates upward under both, consistent
with Task 7's finding that climate's balanced storm-season regimes let Mondrian
work where recession's degenerate test window does not.

As in Task 7, diversity-optimal **overcovers** at 99.21% — above nominal at
~1.8–2.0× width. That is a cost, not a clean victory.

---

## The actual point: model-independence of the mechanism

Diversity-optimal's gain over pooled/trailing, across all **eight** underlying
models now tested per the two domains:

| Domain | Model | Origin | Baseline | Div-opt | Gain | Width× | CI-separated |
|---|---|---|---|---|---|---|---|
| Healthcare | LACE | literature | 86.99 | 91.78 | +4.79 | 1.29 | no |
| Healthcare | HOSPITAL | literature | 86.99 | 91.78 | +4.79 | 1.20 | no |
| Healthcare | ridge | internal | 88.36 | 92.47 | +4.11 | 1.27 | no |
| Healthcare | gradboost | internal | 90.41 | 93.84 | +3.43 | 1.22 | no |
| Climate | persistence | literature | 90.87 | 99.21 | +8.34 | 2.02 | **yes** |
| Climate | climatology | literature | 90.08 | 99.21 | +9.13 | 1.76 | **yes** |
| Climate | ridge | internal | 90.95 | 99.59 | +8.64 | 1.72 | **yes** |
| Climate | gradboost | internal | 90.95 | 98.77 | +7.82 | 1.79 | **yes** |

Within each domain the gains cluster tightly regardless of model family
(healthcare +3.4 to +4.8pp; climate +7.8 to +9.1pp), and the literature/internal
split explains none of the variation. **The between-domain difference is far
larger than the between-model difference within a domain** — which is what the
mechanism predicts, since diversity is a property of the score distribution and
the domain determines how much room there is to widen it.

---

## Honest caveats

- **HOSPITAL is a 5-component variant, not the published score.** Two of seven
  components are unreconstructable from this dataset. Do not cite its numbers as
  the Donzé score's performance.
- **Healthcare separates nothing, again.** Four models, no CI separation for any
  strategy. The domain's baseline sits near nominal, so it confirms the mechanism
  transfers but cannot rank strategies. Point estimates are not evidence.
- **Climate diversity-optimal overcovers** (99.21%) at ~2× width. Reported as a
  cost.
- **Both LACE and HOSPITAL underperform their published AUCs** on this dataset
  (0.569/0.588 vs 0.608–0.684). The reconstruction check passed and the cause is
  measured (weak components, wrong ED window, truncated comorbidity coding), but
  these are weaker instantiations of the scores than their derivation cohorts
  produced.
- **The internal ML models barely beat climatology** in the climate domain
  (2.93 vs 3.05 MAE). The point-prediction layer is doing little work there.
- **Model-independence is demonstrated, not proven.** Four model families per
  domain is a stronger test than two, but all four consume the same target
  construction and the same cohort/panel definitions. A genuinely different
  target construction could still behave differently.
