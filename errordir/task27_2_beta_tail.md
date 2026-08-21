# Task 27, Item 2 — Tail-Targeted β and the Unmodified Validation Gate

**Result: β_tail validates on energy but FAILS on climate — a domain where
β_mean passes. Reported as a negative result for tail-targeting on climate, per
the guardrail.**

Script: `task27_3_comparison.py` · Data: `task27_2_validation.csv`

## Target definition

    d_i = 1{ s_i > q_α }        q_α = the (1−α)=0.90 quantile of FIT-split scores

The **binary indicator** is used rather than a continuous tail-proximity score,
because it is exactly the event coverage is defined on: ACI consumes the
(1−α) quantile, so the operative question is "is this point beyond it," not "how
far along the severity scale." β_tail is fit by **logistic regression** of `d_i`
on the same features β_mean uses.

`q_α` is computed on the **FIT split only** — never CAL or TEST — so the label
definition itself carries no calibration or test information. Realized label
rate: **0.1005** (climate), **0.1003** (energy), confirming the threshold lands
where intended.

## Gate results — Task 21's two-part gate, unmodified

| domain | β | held-out CAL r | angle PC1 | Check 1 | perturb effect | perturb pct | Check 2 | **validated** |
|---|---|---|---|---|---|---|---|---|
| climate | β_mean | 0.333 | 84.6° | PASS | 0.625 | 0.985 | PASS | **Yes** |
| climate | **β_tail** | 0.281 | 81.6° | PASS | **0.0155** | **0.030** | **FAIL** | **No** |
| energy | β_mean | 0.417 | 81.7° | PASS | 2.330 | 1.000 | PASS | **Yes** |
| energy | **β_tail** | 0.348 | 83.7° | PASS | 2.475 | 1.000 | PASS | **Yes** |

No new gate was added. β_tail had to clear the bar built for β_mean, and on
climate it does not.

## The climate failure is dramatic, not marginal

β_tail's perturbation effect on climate is **0.0155 against β_mean's 0.625 — a
40× collapse**, landing at the 3rd percentile of its own model's null. This is
not a borderline miss.

The mechanism is visible in the two numbers together. β_tail's held-out
correlation with realized error is respectable (0.281, versus β_mean's 0.333), so
it *does* carry information about where errors are large. But moving inputs along
it barely moves the ridge prediction at all.

Logistic regression on a rare binary label (10% positives) optimizes for
separating tail from non-tail cases. The direction that best separates those
classes need not be a direction the *prediction function* is sensitive to — it
can be dominated by features that discriminate tail membership without driving
the point forecast. That is precisely the correlational-vs-causal-adjacent gap
Task 21's Check 2 exists to detect, and it is the third time this project has
seen it (Task 23's disagreement feature, Task 25's tree models, now β_tail).

## Why this matters for the task's framing

The task anticipated that a new "tail relevance" gate would be circular and
required the existing gate instead. That decision is vindicated: a
tail-specific gate (does moving along β_tail change P(tail|x)?) would very
likely have **passed** climate — β_tail was fit to predict exactly that — and
would have hidden a 40× collapse in the quantity that actually matters for
building an interval.

## Consequence

Only **energy** carries a validated β_tail. Item 3's head-to-head is therefore
decided on energy for the tail arm; climate's tail arm is reported in the
comparison table for completeness but does not count toward the pre-registered
criterion, since an unvalidated β cannot supply a valid result.
