# Task 33 — Is the Tree-Model Failure About the Forecaster, Not the Mechanism?

## **Item 1 falsifies the hypothesis. Tree forecasters are NOT unresponsive to perturbation — zero-response rate 0.00003, and they move 2.7× MORE than ridge. Task 32's "linear-only" scope stands empirically, but its stated explanation was wrong.**

## Item 1 — the measurement

Perturbation responsiveness of the forecasters alone (no difficulty function
involved), using Check 2's exact perturbation: 0.5 SD, 200 random directions,
40 test points per fit.

| class | **zero-response rate** | median relative change | dead points |
|---|---|---|---|
| **tree** (5 fits) | **0.00003** | **0.166** | **0.000** |
| linear (3 fits) | 0.00000 | 0.061 | 0.000 |

Out of 8,000 (point, direction) pairs per fit, essentially none produce zero
change, and **not one test point in any fit is "dead."** Four of five tree fits
have a zero-rate of exactly 0.00000.

The reason is structural: these are *boosted ensembles of 200–250 trees*.
Piecewise-constancy is a property of a **single** tree. A 0.5 SD perturbation
across 12–21 features will cross a split in at least one member with
overwhelming probability.

## What this corrects

Task 32 concluded linear-only scope from 0/14 tree fits across three mechanisms,
and attributed it to piecewise-constant prediction surfaces. **The empirical
record stands; the explanation does not.**

The real mechanism, which the measurement reveals: Check 2 compares the signal's
direction against a **random-direction null**. Trees respond strongly to
*essentially any* direction, so that null is **high and flat**. Task 25 measured
this directly (gradboost's null p95 at 3× ridge's, with *lower* variance) but
read it as step-function spikiness. It is the opposite — trees are sensitive
almost everywhere, so no single direction distinguishes itself.

**Task 32's 0/14 is a signal-to-noise problem in the null, not a dead
forecaster.**

## Item 2 — verdict

Outcome 2, stated with the measured rate: **0.00003 zero-response**, three orders
of magnitude below any reading of "most perturbations produce zero response."
Item 3 warranted and run.

Scoping note: Item 1 found responsiveness **uniform** — zero dead points
everywhere — so there was no unresponsive region to exclude, and Item 3 ran
across the full test set. Recorded as a consequence of the measurement, not an
omission.

## Item 3 — locally-varying D(x): 0/6, and the test was compromised

Nadaraya-Watson kernel regression with a **closed-form** gradient, applying
Check 2 **per-point** (each point perturbed along its own ∇D(x), against its own
null) — the mechanistic difference from every prior attempt.

| class | validated | mean percentile |
|---|---|---|
| tree | 0/4 | 0.497–0.693 |
| **ridge control** | **0/2** | **0.890, 0.921** |

Two things prevent reading this as a tree-model finding:

**The ridge control fails.** In Task 32 the local-kNN candidate validated on both
ridge arms (0.965, 1.000), which is what licensed attributing tree failures to
model class. Here ridge fails too, so this mechanism is weak on linear models
as well and its tree results say nothing tree-specific. The control did its job.

**The bandwidth made D(x) effectively global.** With d = 12–21, median pairwise
distance is √(2d) ≈ 4.9–6.5; the fitted `h` was **4.21–4.36** — the same order.
The kernel averages over most of the sample, so ∇D(x) barely varies (only 3–10%
of tree points exceed 0.95, the signature of a near-constant gradient). **The
locality hypothesis was not genuinely tested.**

I did not re-run with a smaller `h`. The median heuristic was fixed in advance
from a stated rationale, and picking a bandwidth after seeing the first one fail
is the test-set tuning this project has refused since Task 22.

Check 1 also fails on energy (angles 22–28°, below the 30° bar) for ridge as
well as trees — another symptom of the over-wide bandwidth: D(x) is tracking the
dominant variance direction rather than a distinct difficulty direction.

## Where this leaves tree-model support

| question | answer |
|---|---|
| Is the tree failure forecaster-level (unresponsive predictions)? | **No — falsified by direct measurement.** |
| Is Task 32's linear-only scope still the empirical record? | **Yes — 0/20 tree fits now, across four mechanisms.** |
| Is it a *mechanism*-level limitation? | **Still open.** Item 3 was meant to test this and did not, for a stated configuration reason. |
| Is it a property of Check 2's null on trees? | **Most likely** — this is the explanation the measurement supports. |

The honest scope statement changes in one specific way: *"errordir's validated
form requires a linear point-forecaster"* remains true empirically, but the
reason is **not** that tree predictions don't respond. It is that Check 2's
random-direction null is uninformative when a model responds to every direction.

## The well-defined follow-up, if one is wanted

Not a fifth mechanism. The same locally-varying D(x) with a bandwidth selected by
a principled **fit-period-only** criterion — leave-one-out likelihood on the FIT
split, or a dimension-adjusted Silverman rule — fixed in advance. That would
actually test the locality hypothesis this task set out to test.

A second, arguably more valuable option: given that the diagnosis is now about
**Check 2's null on high-responsiveness models**, a null construction that
conditions on the model's overall responsiveness (e.g. comparing against
directions matched for induced prediction change, not just unit length) would
test whether the gate itself can discriminate on trees at all.

## Files

- `task33_1_responsiveness.py` / `.md`, `task33_1_responsiveness.csv`
- `task33_2_verdict.md`
- `task33_3_local_difficulty.py` / `.md`, `task33_3_local_difficulty.csv`
