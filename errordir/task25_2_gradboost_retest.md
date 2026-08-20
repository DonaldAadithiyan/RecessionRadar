# Task 25, Item 2 — Climate/Gradboost Retest

## Result: **still does not validate. Task 24's open item is now resolved — as a negative, not a rescued positive.**

| null construction | climate/gb percentile | verdict |
|---|---|---|
| (A) same-model-class *(what Task 24 used)* | 0.035 | FAIL |
| (B) magnitude-matched | 0.030 | FAIL |
| (C) sign-structure | 1.000 | PASS — but rejected as a valid test, see Item 1 |

Under both defensible null constructions climate/gradboost fails Check 2
decisively — at the **3rd percentile** of its own model's null, not marginally.

## Reconciling this with the high held-out correlation

Task 24 flagged climate/gradboost as having "the highest held-out correlation of
any fit (0.374)" and therefore likely a real signal behind a broken test. Both
halves of that can be true, and the resolution is the same one Task 23 found:

- **Held-out r = 0.374** says β's projection correlates with realized error
  out-of-sample. Real, and the best of the four Task 24 fits.
- **Check 2 percentile = 0.035** says perturbing inputs along β changes the
  gradboost prediction *less* than a random direction does.

These measure different things, and Check 2 exists to catch exactly this gap.
Task 23 saw the same pattern when the disagreement feature improved correlation
at every horizon while collapsing the perturbation percentile at two of them.

For a tree ensemble the gap has a plausible structural cause: the prediction
surface is piecewise-constant over axis-aligned splits, so influence concentrates
on a few high-importance features. β, fit to predict error magnitude, spreads
weight across ~9.7 effective dimensions and need not align with those splits. It
can track error well while being a weak lever on the prediction.

**No comparison was run** for climate/gradboost. Per the Item 2 guardrail
("a passed validation gate is not itself a result"), and here the gate was not
even passed, so running the anti-gaming comparison would be reporting a result
for an unvalidated fit.

## What this settles

Task 24 left this genuinely open and suggested it might be "a second real result
sitting in already-computed data, just behind a broken test." It is not. The
cheapest possible check — the one Item 2 correctly ordered before any new
data-fetching — came back negative, which is exactly what that ordering was for.
