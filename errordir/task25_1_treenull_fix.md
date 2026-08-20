# Task 25, Item 1 — Fixing Check 2's Perturbation Null for Tree Models

## Result: **the prescribed fix does not work, and the premise it rests on is incorrect. Reported as a negative finding rather than worked around.**

Script: `task25_1_treenull_fix.py` · Data: `task25_1_treenull_fix.csv`

## The premise, checked before building anything

Task 25 states the null fails for trees because "gradboost's step functions move
more under random perturbation, inflating the null's p95 threefold." Measuring
the null's actual shape on climate:

| model | mean | median | p95 | sd | frac zero | skew |
|---|---|---|---|---|---|---|
| ridge | 0.227 | 0.200 | 0.553 | 0.167 | 0.000 | 0.78 |
| gradboost | **0.908** | 0.900 | 1.103 | **0.119** | **0.000** | 0.36 |

The gradboost null is **not** lumpy, spiky, or zero-inflated — it has *lower*
variance and *less* skew than ridge's, with no zeros at all. Random directions
move a tree ensemble a lot, and do so **consistently**. The "step-function
jumpiness" mechanism is not what is happening.

A second candidate confound was tested and also rejected: β might be sparse
while random directions are dense, so a tree crosses fewer splits along β. It
isn't — β's participation ratio is **9.7 of 21** effective dimensions versus
**8.0** for random directions. Comparable.

**What is actually true: for a tree ensemble, β genuinely moves the prediction
less than a typical random direction does.** That is a property of the fitted
model, not an artifact of the test instrument.

## Three null constructions, all verified against the known climate/gradboost case

| domain/model | effect β | (A) same-class | (B) magnitude-matched | (C) sign-structure |
|---|---|---|---|---|
| climate/ridge | 0.851 | 1.000 PASS | 0.995 PASS | 1.000 PASS |
| **climate/gb** | 1.009 | **0.035 FAIL** | **0.030 FAIL** | **1.000 PASS** |
| healthcare/ridge | 3.200 | 1.000 PASS | 1.000 PASS | **0.910 FAIL** |
| healthcare/gb | 2.276 | 0.675 FAIL | 0.680 FAIL | 0.965 PASS |

**(A) Same-model-class null — the fix the task prescribes.** Critically, **Task
24 already did this**: each fit's null was built by perturbing that fit's own
model. Verified explicitly rather than assumed. So "use the same model class"
was already in force and cannot change any verdict. climate/gb still fails at
0.035.

**(B) Magnitude-matched null.** Rescales each random direction so its
input-space displacement equals β's, isolating direction from step-size — the
most plausible genuine repair for a non-smooth model. It changes essentially
nothing (0.030 vs 0.035). This rules out step-size as the explanation.

**(C) Sign-structure null.** Replaces "does β move the model far" with "does β's
projection track realized error," measured on the held-out CAL split. This
*does* let climate/gb pass (1.000).

## Why (C) is not adopted, despite being the only one that "works"

The guardrail says the fix must resolve the known climate/gradboost case. (C)
does. But adopting it would be wrong, for two reasons:

**1. It is a near-trivial bar that re-tests what Check 1 and held-out r already
cover.** β is fit by regression to predict |error|. (C) asks whether that fitted
regression beats *random projections* at correlating with |error| — the exact
objective it was optimized for. Any β that generalizes at all passes. It is a
test of non-zero generalization, not the causal-adjacency test Check 2 exists to
provide. Task 21 introduced Check 2 precisely because "correlational fit alone
is not sufficient evidence β captures something causal-adjacent."

**2. It is not strictly better — it breaks a case that currently passes.**
(C) *fails* healthcare/ridge (0.910), which passes under both A and B. So it is
not a repair; it is a different test with a different, partly worse, pattern of
outcomes. Adopting a test that flips one result to pass and another to fail,
chosen after seeing which cases it rescues, is exactly the post-hoc selection
this project's discipline forbids.

Deleting Check 2's distinctive content would also discard the finding it
produced in Task 23 — that heavy disagreement-loading breaks causal adjacency
while *improving* correlation. That was the most informative result of that
task, and only Check 2 could see it.

## Conclusion

**Check 2 is not broken for tree models. It is reporting something real:
β is a weaker direction of influence for a tree ensemble than for a linear one.**

Task 24's report speculated the failure was a test-design mismatch; that
speculation is now tested and does not hold. The honest statement is that
tree-based β's fail Check 2 on their merits under every non-degenerate null
construction tried.

Note that climate/gb's effect (1.009) is *larger in absolute terms* than
climate/ridge's (0.851). It fails anyway because its own model's null is much
higher (p95 1.816 vs 0.576). Both facts are true simultaneously and neither is
an error.

**Consequence for this task:** Item 1's prerequisite is not satisfiable as
specified. This blocks the tree-model path — see `task25_2_gradboost_retest.md`
and the summary for what that means for Items 3–6.
