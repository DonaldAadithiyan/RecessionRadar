# Task 24, Item 1 — Feature and Signal Audit (Healthcare, Climate)

**Scoping result: both domains have usable continuous features under existing
OOF discipline, and — contrary to the task's tentative expectation — both have a
*legitimate* ensemble-disagreement analog. It is structurally weaker than
recession's, and the difference is stated explicitly below rather than glossed.**

## Healthcare

Source: `fix-reg/domain_healthcare.py` (UCI Diabetes 130-US Hospitals).

| | |
|---|---|
| unit | patient cohort (≥25 encounters), target = 30-day readmission rate (%) |
| pool / test | 438 / 146 cohorts |
| features | **20 continuous** — 8 clinical counts (time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses) + cohort size + case-mix shares (change, diabetesMed, insulin, A1Cresult dummies averaged per cohort) |
| OOF discipline | rolling-origin folds, `SCORES[model] = |oof − y_pool|`, already established |
| target spread (p95−p5) | 22.514 |

Note: healthcare cohorts are **permuted** (`rng.permutation`) before splitting,
not temporally ordered. This matters for Item 4 — the autocorrelation-aware
permutation tests that Tasks 22–23 required are designed for temporal structure,
and healthcare has none by construction. Handled explicitly in Item 4.

## Climate

Source: `fix-reg/domain_climate.py` (NOAA Storm Events, region-month panel).

| | |
|---|---|
| unit | region-month, target = significant-event intensity |
| pool / test | 1377 / 243 region-months (1148 with finite OOF scores) |
| features | **21 continuous, all strictly lagged** — tgt_lag{1,2,3,6,12}, cnt_lag{1,2,3,6,12}, tgt_roll3, tgt_roll12, cas_lag1, dmg_lag1, sin_m, cos_m, 5 region dummies |
| OOF discipline | rolling-origin, `shift(1)`-lagged predictors — "no same-month information" enforced at construction |
| split | **temporal** (last 15% of panel) — genuine time structure |
| target spread (p95−p5) | 8.778 |

Climate is the larger and better-powered testbed: **243 test points vs
recession's 59**, with real temporal ordering.

## The disagreement-analog question — the decisive part of this audit

The task flagged that healthcare/climate have "two *independent* model choices
rather than an internal ensemble" and warned against manufacturing a signal.
Checking the actual code rather than assuming:

```python
MODELS = {"ridge": Ridge(alpha=1.0),
          "gradboost": HistGradientBoostingRegressor(...)}
...
for name, factory in MODELS.items():
    for tr, te in rolling_origin_folds(len(y_pool), n_folds=5):
        m = factory(); m.fit(Xp[tr], y_pool[tr]); oof[te] = m.predict(Xp[te])
```

Both models are trained on **the identical task**: same `X_pool`, same
`y_pool`, same rolling-origin folds, same target. They differ only in hypothesis
class (linear vs boosted trees). That is precisely the condition the guardrail
requires — "genuinely trained on the same task in a way that makes their
disagreement meaningful."

Empirical confirmation that the pair is non-degenerate:

| domain | corr(ridge, gb) on test preds | mean abs diff | ridge OOF MAE | gb OOF MAE |
|---|---|---|---|---|
| healthcare | **0.719** | 2.101 | 4.510 | 4.973 |
| climate | **0.621** | 1.128 | 2.933 | 3.080 |

Correlations of 0.62–0.72 mean the two models genuinely disagree on a
substantial fraction of points. A degenerate pair (corr ≈ 0.99) would have made
disagreement uninformative; that is not the case here.

**So the analog is legitimate and will be used.** Disagreement features per
domain-model pair: `disag_absdiff = |pred_ridge − pred_gb|`.

### The structural difference from recession, stated explicitly

This is **not** the same quantity recession had, in three ways:

1. **Two models, not three.** Recession's disagreement is `std`/`range`/`maxpair`
   over CatBoost/LightGBM/RandomForest — three numbers admitting a dispersion
   statistic. With two models only `|a−b|` exists; `std`, `range`, and `maxpair`
   all collapse to the same quantity up to a constant. So healthcare/climate get
   **one** disagreement feature where recession got three.
2. **Not an internal ensemble component.** Recession's base models feed an
   ElasticNet meta-learner, so disagreement is already computed inside the
   prediction path (`_engineer_meta_features`). Here ridge and gradboost are
   *reported alternatives*, not components of a combined predictor. Using their
   disagreement is a defensible extension, but it is a new construction rather
   than extraction of an existing internal quantity.
3. **Self-referential for the model being scored.** When β for
   healthcare-ridge uses `|pred_ridge − pred_gb|`, one term is the very model
   whose error β is predicting. This is not leakage — both predictions are
   functions of `X` only, no outcome — but it is a different relationship than
   recession's, where the meta-learner's error was predicted using disagreement
   among its *inputs*.

Point 3 is worth watching in Item 3. Task 23 found that when β loaded heavily on
disagreement (18.7%, 25.6% of weight), the perturbation check **failed** because
disagreement is a model *output*, not a direction the prediction is sensitive to.
The same mechanism could fire here, and more strongly, since the disagreement
term directly involves the scored model.

## Scoping conclusion

- Healthcare: 20 base features + 1 disagreement feature = 21.
- Climate: 21 base features + 1 disagreement feature = 21+1 = 22.
- Both proceed to Item 2 with disagreement included.
- Recession's β is not transferred; all four fits are fresh.
