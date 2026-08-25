# Task 32, Item 3 — Candidate B: Ensemble Disagreement as the Score Itself

**Result: fails everywhere — 0/4 on trees AND 0/2 on ridge.**

Script: `task32_2_local_difficulty.py` (both candidates share the harness) ·
Data: `task32_4_validation.csv`

## Why this is not Task 23's construction — stated before results, per the guardrail

Task 23 fed disagreement in as **one input feature to a linear β**. That asked β
to treat a model *output* as if it were an independent input cause, then fit a
direction through it. Check 2 broke exactly there: perturbing along a direction
loaded on disagreement (18.7% and 25.6% of β's weight) moved the prediction
*less* than a random direction, because disagreement is downstream of the
prediction, not upstream of it.

Here there is **no projection and no fitted direction**. Disagreement *is* the
difficulty score. Nothing is being asked to treat an output as an input, so Task
23's specific failure mode cannot arise by construction.

For tree models the ensemble is the model's own boosting members (staged
predictions at 20-round intervals). For the ridge control a 20-member bootstrap
ensemble is built, per the task's explicit allowance.

## Results

| domain | model | class | held-out r | angle PC1 | perturb pct | validated |
|---|---|---|---|---|---|---|
| insurance | xgboost | tree | **0.396** | 62.6° | 0.565 | **No** |
| insurance | lightgbm | tree | **0.306** | 84.6° | 0.600 | **No** |
| insurance | ridge | linear | 0.010 | 69.8° | 0.410 | **No** |
| energy | xgboost | tree | **0.378** | 79.8° | 0.540 | **No** |
| energy | lightgbm | tree | **0.347** | 81.3° | 0.645 | **No** |
| energy | ridge | linear | **0.415** | 86.6° | 0.175 | **No** |

## The informative part

**Candidate B has the best held-out correlation of any candidate on insurance
trees** — 0.396 and 0.306, versus Candidate A's 0.021 and 0.031, a **13×**
difference. Disagreement genuinely predicts where tree-model errors are large.
That is consistent with the literature Item 1 found, which reports
variance-normalization working well on tree ensembles.

And it still fails Check 2, at percentiles of 0.54–0.65.

So avoiding Task 23's output-as-input error **did** fix the correlational
problem — disagreement is a far better error predictor used directly than it was
as a β input. It did **not** fix the causal-adjacency problem. Disagreement
remains a model output: knowing the ensemble disagrees at x tells you the
prediction there is unreliable, but *moving inputs along the disagreement
gradient* does not move the prediction more than a random direction does.

That is a coherent finding rather than a contradiction, and it is the third
distinct time this project has separated correlational quality from causal
adjacency (Task 23's disagreement-as-input, Task 27's β_tail, now this).

Candidate B also fails on **ridge**, unlike Candidate A. So it is a weaker
candidate overall, not merely a tree-specific failure.
