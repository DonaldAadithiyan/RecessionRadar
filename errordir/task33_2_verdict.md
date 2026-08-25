# Task 33, Item 2 — Verdict: Does Item 3 Run?

## **Outcome 2 — a meaningful fraction of perturbations produce a response. In fact essentially all of them do. Item 3 IS warranted, and runs.**

## The measured rate, stated plainly as the guardrail requires

**Tree zero-response rate: 0.00003** — three thousandths of one percent of
(point, direction) pairs produce no change in the forecaster's output. The
per-fit maximum is 0.00013 (insurance/xgboost); four of five tree fits are
exactly 0.00000.

**Fraction of test points that are "dead"** (≥95% of directions producing zero
change): **0.000 in every tree fit.**

Outcome 1 required "most perturbations produce zero forecaster response." The
measured rate is not merely below that threshold — it is three orders of
magnitude below any reasonable reading of it, and trees move **2.7× more** than
the ridge control.

## What this means for Task 32's finding

Task 32 concluded that linear-only is the honest scope, based on 0/14 tree fits
across three mechanisms. That empirical record stands unchanged — no candidate
has passed Check 2 on a tree.

But Task 32's *explanation* was wrong, and this correction matters:

- **Task 32 implied** (and this task's hypothesis stated) that tree predictors
  don't admit a causal-adjacent direction because their surface is
  piecewise-constant.
- **Item 1 shows** they respond to perturbation more than ridge does. The
  surface is not the obstacle.

The real obstacle is that Check 2's random-direction null is **high and flat** on
trees: they respond to any direction, so the 95th percentile bar is high and no
single direction clears it. Task 25 measured this directly (gradboost null p95 at
3× ridge's, with *lower* variance) but read it as a property of step functions.
It is better read as: an additive ensemble of hundreds of trees is sensitive in
essentially every direction, which is the opposite of the stated hypothesis.

## Why Item 3 is now a real test rather than a repeat

The three mechanisms tried so far — global β, local kNN, direct disagreement —
all produce, at the moment of the check, **a single direction per fit** (β's
coefficient vector, or the fitted gradient of the pointwise signal). Against a
flat high null, one global direction is exactly the wrong shape of candidate.

A **locally-varying** difficulty function supplies a *different direction at each
test point*, ∇D(x). If the tree's sensitivity is genuinely local — strong but
direction-varying across feature space — then a per-point direction could clear a
null that a single averaged direction cannot. That is a mechanistically distinct
proposition, not a fourth flavour of the same thing.

## Scoping note

Item 3's guardrail says to restrict testing to regions Item 1 found responsive.
**Item 1 found responsiveness is uniform** — zero dead points, near-zero
zero-rate everywhere. There is no unresponsive region to exclude, so Item 3 runs
across the full test set rather than a subset. This is recorded so the absence of
regional restriction is a stated consequence of the measurement rather than an
omission.

---

## Addendum, written after Item 3 ran

The verdict above was written **before** Item 3 executed, as a go/no-go decision
based solely on Item 1's measured rate. Recording what followed, so the decision
and its outcome are both on the record:

**Item 3 returned 0/6, and the ridge control failed as well** (percentiles 0.921,
0.890). Two consequences for how this verdict should be read:

1. **The decision to run Item 3 was correct on the evidence available.** Item 1
   falsified the unresponsiveness hypothesis decisively, and a per-point
   direction against a flat high null was a genuinely distinct proposition. The
   go/no-go logic holds regardless of the outcome.

2. **But Item 3 did not actually test the locality hypothesis.** The
   median-heuristic bandwidth (h = 4.21-4.36) turned out to be the same order as
   the median inter-point distance in 12-21 dimensions (sqrt(2d) ~ 4.9-6.5), so
   D(x) averaged over most of the sample and grad D(x) barely varied by point.
   The mechanism was configured, by its own principled default, into a nearly
   global regime -- the one regime Item 2 argued was the wrong shape of
   candidate.

So the honest status is **not** "locality was tried and failed." It is
"locality was specified, but the fixed-in-advance bandwidth rendered it global,
and re-choosing the bandwidth after seeing the result would be test-set tuning."
The follow-up named in the summary -- the same mechanism with a
fit-period-only bandwidth selection rule -- is what would settle it.
