# Task 25, Item 5 — Validation Gate, All Fits

**Result: 2 of 6 validated — both are RIDGE. All four tree fits (XGBoost and
LightGBM, both datasets) fail Check 2.**

Script: `task25_456.py` · Data: `task25_5_validation.csv`

## Full table (all fits reported, per the guardrail)

| dataset | model | fit r | held-out CAL r | angle PC1 | Check 1 | perturb pct | Check 2 | **validated** |
|---|---|---|---|---|---|---|---|---|
| insurance | xgboost | 0.126 | 0.089 | 70.8° | PASS | 0.410 | FAIL | No |
| insurance | lightgbm | 0.132 | 0.096 | 73.5° | PASS | 0.320 | FAIL | No |
| insurance | **ridge** | 0.124 | 0.126 | 84.9° | PASS | **1.000** | PASS | **Yes** |
| energy | xgboost | 0.576 | **0.410** | 77.8° | PASS | 0.465 | FAIL | No |
| energy | lightgbm | 0.550 | **0.437** | 80.8° | PASS | 0.520 | FAIL | No |
| energy | **ridge** | 0.602 | 0.417 | 81.7° | PASS | **1.000** | PASS | **Yes** |

## The ridge control arm earns its place

Ridge was included not as a third practical model but as a **control**, to
separate two explanations Task 24 and Item 1 could not distinguish on two
domains alone:

- *Hypothesis A:* tree-based β fails Check 2 because of model class.
- *Hypothesis B:* these particular new domains lack a real difficulty signal.

The result separates them cleanly. On both datasets, ridge passes at percentile
**1.000** — the maximum — while XGBoost and LightGBM fail at 0.32–0.52, all in
the middle of their own nulls. Same data, same features, same β-fitting
procedure, same split. **Hypothesis A is supported; B is ruled out.**

Energy is the sharpest case: LightGBM has the *highest* held-out correlation of
any fit in this task (**0.437**, above ridge's 0.417) and still fails Check 2 at
0.520. That is the Task 23 pattern again — correlational quality and
causal-adjacency are separate properties — now demonstrated on a fresh domain
with n=400 test points.

## What this establishes about Item 1's finding

Item 1 concluded that Check 2 is not broken for trees but is reporting something
real. That conclusion was drawn from two domains and one tree model
(gradboost). It now replicates across **two new domains and two different tree
implementations**, with a matched linear control passing on the same data:

| model class | fits | passed Check 2 |
|---|---|---|
| linear (ridge) | 4 (incl. Task 24) | **4 / 4** |
| tree (gradboost, xgboost, lightgbm) | 6 (incl. Task 24) | **0 / 6** |

Ten fits, a perfect split by model class. This is no longer a quirk of one
dataset — it is a systematic property. β, fit as a *linear* projection of
features onto expected error, is a strong lever on a linear predictor and a weak
one on a piecewise-constant tree ensemble, whose influence concentrates on a few
axis-aligned splits.

## Consequence

Per the Item 5/6 guardrails, only validated fits proceed to the comparison:
**insurance/ridge and energy/ridge**. The four tree fits are reported here and
excluded there.

This means **Task 25's headline question — does the method generalize beyond
ridge? — is answered NO**, for reasons Item 1 established are real rather than
instrumental. The domain-transfer question is answered separately and more
favourably in Item 6.
