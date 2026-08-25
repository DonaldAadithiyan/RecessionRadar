# Task 32 — Toward a Model-Agnostic Difficulty Signal

## **Neither candidate produces a model-agnostic difficulty signal. 0 of 8 tree fits validate. "Linear point-forecasters" is the method's honest scope — stated as a finding, not left open for a fourth attempt.**

## The cumulative record now settles this

Across Tasks 24, 25 and 32, Check 2 has been run on **three structurally
different** difficulty estimators:

| model class | β (linear projection) | A: local kNN | B: ensemble disagreement | **total** |
|---|---|---|---|---|
| **linear** | 4 / 4 | 2 / 2 | 0 / 2 | **6 / 8** |
| **tree** | 0 / 6 | 0 / 4 | 0 / 4 | **0 / 14** |

**Fourteen tree fits, zero passes, three mechanisms.** Tree perturbation
percentiles span 0.395–0.775 against a 0.95 bar — none is close.

This task was designed to remove the specific assumption blamed for β's failure:
a single global direction implying a smooth consistent error gradient. **Both
replacements dropped that assumption and both still failed.** That eliminates the
leading explanation and makes the linear-only scope a conclusion rather than an
untested limitation.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Literature | **Both candidates are prior work** | Ensemble-variance normalization is *the standard* difficulty estimator for tree ensembles; kNN-local conformal is an established family. No novelty available from this task. |
| 2 — Candidate A | **2/2 ridge, 0/4 tree** | Pointwise local difficulty; no global direction; still fails on trees. |
| 3 — Candidate B | **0/2 ridge, 0/4 tree** | Best held-out r on trees (0.396) of any candidate, still fails Check 2. |
| 4 — Validation | **0/8 tree** | Ridge control confirms the domains have signal. |
| 5 — Full pipeline | **Not run** | Nothing passed Item 4 on a tree domain. |

## What the ridge control established

Candidate A validates on **both** ridge arms — insurance (pct 0.965) and energy
(pct 1.000, held-out r 0.433, matching β's own performance there) — on the same
data, features and procedure as the failing tree arms. So this is not "these
domains lack signal" and not "Candidate A is a bad estimator." The failure is
specific to tree model class, exactly as Task 25's control design was built to
determine.

## The most informative single result

**Candidate B on insurance/xgboost: held-out r = 0.396, perturbation percentile
= 0.565.** Candidate A on the same fit: r = 0.021. Disagreement is a **13× better
error predictor**, matching what the literature reports for
variance-normalization on tree ensembles — and it still cannot clear Check 2.

Avoiding Task 23's output-as-input error *did* fix the correlational problem. It
did **not** fix causal adjacency. Knowing an ensemble disagrees at x tells you
the prediction is unreliable there; moving inputs along the disagreement gradient
still does not move the prediction more than a random direction. Third distinct
instance of this separation in the project (Task 23, Task 27's β_tail, now this).

## The unresolved question this leaves — stated honestly

Item 1 found the published literature reports variance-normalized conformal
prediction **working well on tree ensembles**. This project finds the same
quantity failing Check 2 there, 0/4. Both can be true, and the reconciliation
matters:

> **Check 2 is a stricter requirement than normalized conformal prediction needs.**
> Those methods require a difficulty estimate that *correlates* with error —
> which Candidate B plainly delivers (r up to 0.396). Check 2 additionally
> demands causal adjacency: that perturbing inputs along the signal moves the
> prediction more than a random direction. Tree predictors appear not to admit
> such a direction from any estimator tried.

So the honest scope statement has two parts, and the second is a limitation of
this project's own standard rather than of tree models:

1. **errordir's validated form requires a linear point-forecaster.** Settled by
   14 tree fits across 3 mechanisms.
2. **This may be a property of Check 2 rather than of difficulty estimation on
   trees.** A published method reported to work on tree ensembles cannot clear
   it. Whether Check 2 is *too* strict is a question about the gate — and per
   Task 25 Item 1, weakening a gate after seeing what it rejects is exactly what
   this project does not do. It stands as stated, with the caveat recorded.

## No fourth attempt is motivated

Three mechanisms, fourteen tree fits, a working ridge control, and a literature
check confirming the candidate space is well-explored. There is no new reasoning
available that would motivate a fourth attempt, and the guardrail is explicit
that this should be reported as scope rather than kept open.

## Files

- `task32_1_literature.md`
- `task32_2_local_difficulty.py` / `.md` — Candidate A (shared harness)
- `task32_3_disagreement_direct.md` — Candidate B, with the Task 23 contrast
- `task32_4_validation.csv` — full 12-row table
- `task32_5_fullpipeline.md` / `.csv` — not run, with reason
