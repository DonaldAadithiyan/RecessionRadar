# Task 37, Item 5 — Record Update

## No superseded banners are warranted. Item 5's condition was not met.

Item 5 specified: *"If Item 1 finds Check 2 was also construction-trivial on
ridge, add a superseded banner to every task report that cited a ridge Check 2
pass as meaningful evidence (Tasks 21, 24–27, 30)."*

**Item 1 found the opposite.** Fitted-but-meaningless directions
(noise-target, permutation-target) score Check 2 percentiles of 0.18–0.48 and
pass **0 of 20 times**, while real β scores 0.995 on both domains. Check 2's
ridge pass is not an artifact of fitting.

So Tasks 21, 24–27 and 30 cited ridge Check 2 passes as meaningful evidence, and
**that citation stands.** No banner is added to any of them.

## What the record now says, replacing "untested"

Task 36's summary left the mechanistic question explicitly open:

> *"Climate/ridge/β remains **untested, not exonerated**... the mechanistic claim
> is not unsupported; it is supported by held-out correlation and untested by any
> causal-adjacency instrument valid on linear models."*

**That status is now superseded by an actual answer.** Item 4 applied a
verified, linear-valid instrument to both fits:

| fit | HOS(β) | null mean | percentile | verdict |
|---|---|---|---|---|
| climate/ridge/β | 0.3583 | 0.1175 | **1.000** | **validated** |
| energy/ridge/β | 0.4023 | 0.1499 | **0.990** | **validated** |

The standing evidence for the mechanistic claim is therefore:

1. **Check 2** passes on both ridge fits (0.995), and Item 1 confirms that pass
   discriminates real from fitted-but-irrelevant directions.
2. **HOS** validates both fits (1.000, 0.990) against a permutation-fitted null,
   verified in both directions on synthetic ground truth.
3. **DS** is void on linear models and says nothing either way (Task 36).

Two independent non-degenerate instruments support the claim; the one that
failed was structurally incapable of measuring it.

## Correction applied to Task 36

Task 36's summary states the mechanistic question is "open rather than settled in
either direction." That was accurate when written and is now outdated. A pointer
has been added to `task36_summary.md` directing readers to this task's Item 4
result.

Task 36's substantive findings are unaffected: DS's degeneracy on linear models,
the withdrawal of Task 34's "magnitude alignment" claim, and the Task 34 banners
all stand.
