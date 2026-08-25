# Task 36, Item 2 — Climate/ridge/β Under DS: NOT RUN

**Blocked by Item 1, exactly as the guardrail specifies.**

Item 1 found all four tested ridge fits degenerate: δᵢ takes **one unique value
across 400 test points** (CV ≈ 2×10⁻¹⁵), because the ridge prediction path is a
composition of affine maps with no clipping or bound, giving
δᵢ = |2h·(a·v)| with no xᵢ dependence.

Climate/ridge/β uses the identical construction — `Ridge(alpha=1.0)` with
`StandardScaler` transforms and no post-hoc nonlinearity — so its δᵢ would be
constant for the same structural reason. Running DS there would produce a number
determined by Spearman tie-breaking on a constant vector, not a measurement.

The guardrail: *"Do not run this item if Item 1 finds degeneracy — testing
climate under a broken instrument would just produce a second uninformative
number."* Item 1 found degeneracy. Item 2 does not run.

## What this leaves unresolved, stated plainly

The question Item 2 existed to answer — *does the mechanistic claim (β identifies
a genuine difficulty direction) hold on both ridge fits behind the positive
result?* — **remains open**. DS cannot answer it. Neither energy/ridge/β nor
climate/ridge/β has been tested by any instrument capable of measuring
directional specificity on a linear model.

What is now known:

- Energy/ridge/β's DS "failure" (percentile 0.204) is **not** evidence against
  the mechanism. It is an artifact.
- Climate/ridge/β is untested, not exonerated.
- Check 2's passes on both fits (0.995, and 1.000 for climate in Tasks 24–25)
  stand unchallenged by this task.

## What a valid instrument would need

DS asks whether a direction's *induced prediction change* varies across points in
a way that tracks error. On a linear model that quantity is constant by
construction, so the question is ill-posed as framed.

A linear-appropriate analogue would have to test something else — for example
whether the *projection* β·xᵢ (rather than the induced change) correlates with
realized error across points. That is a different statistic, and notably it is
close to what this project already reports as `r_cal_heldout` (energy/ridge β:
**0.417**, climate/ridge β: **0.333** — both positive and substantial).

That existing evidence is correlational, which is precisely why Check 2 was
introduced. But it means the mechanistic claim is not unsupported; it is
supported by held-out correlation and untested by any causal-adjacency
instrument valid on linear models. Designing one is a well-posed follow-up, not
part of this task.
