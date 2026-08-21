# Task 26 — Is It Temporal-vs-Random-Split, or Just Coincidence?

## **Outcome 2: no difference between conditions in the predicted direction — the pre-registered prediction is not supported, and Task 25's n=4 pattern should be treated as probably coincidental to which domains were chosen.**

More precisely, the result is slightly *worse* for the hypothesis than plain
outcome 2. errordir beat Mondrian in **both** conditions, and the margin was
**5.6× larger under the random split** (−3.840) than the temporal one (−0.687) —
the opposite of what the hypothesis predicts. It is not a clean outcome 3
(reversal) either, because neither margin is statistically significant and the
sign never flipped. The honest classification is **outcome 2**, with the caveat
that what directional signal exists points *against* the hypothesis rather than
merely being absent.

## The pre-registered prediction, and what actually happened

| | predicted | observed |
|---|---|---|
| temporal condition | errordir **beats** Mondrian | errordir beats Mondrian by **0.687** (win rate 27.8%, q=0.547) |
| random condition | errordir **loses**, or margin narrows substantially | errordir beats Mondrian by **3.840** (win rate 40.2%, q=0.239) |

The prediction required the margin to shrink or reverse under random assignment.
It **grew**.

## Why this is a credible test rather than a weak one

The design controls the confound that a fifth domain could not:

- **Identical rows.** Both conditions use the same 3,600 rows from Beijing PM2.5.
- **Identical counts.** 1800 fit / 1200 cal / 600 test in both, verified.
- **Only the assignment differs**, and it demonstrably differs: lag-1
  autocorrelation of paired Winkler differences is **0.187–0.500** under
  temporal versus **0.027–0.150** under random. The shuffle really did destroy
  the temporal structure.
- **Both conditions validated** the gate (perturbation percentile 0.970 and
  0.985), so neither result is an artifact of one condition failing to produce a
  usable β.

So this is not a null from a broken or underpowered experiment. The manipulation
worked and the predicted effect did not appear.

## The most informative detail

β generalizes **better** under the temporal split (held-out r **0.431** vs
**0.256**) while performing **worse** against Mondrian there. Signal quality and
competitive standing move in opposite directions on the same data.

That decouples the two things the hypothesis conflated. The hypothesis assumed
temporal splits favour errordir *because* distribution shift is where a
difficulty signal helps. The signal is indeed stronger under temporal shift —
but Mondrian and DtACI evidently benefit from that structure at least as much.
Under the temporal condition **DtACI wins outright** (107.469 vs errordir's
113.739, q=0.976), which no prior domain showed.

## What this means for Tasks 24–25's pattern

Four domains produced: errordir beats Mondrian on climate and energy (temporal),
loses on healthcare and insurance (random). That looked structured. This test
says the structure is more likely **domain-coincidental** — properties of those
particular datasets (field, error distribution, tail shape) rather than the split
methodology itself.

The pattern should not be reported as a finding. It can be reported, if at all,
as an observation across four domains that a matched within-domain test failed
to reproduce.

## Standing claim, unchanged by this task

This task tested an explanatory hypothesis, not the method. The Tasks 21–25
result stands as it was:

> The error-direction method achieves the best combined coverage-width tradeoff
> among nine methods on climate (n=243) and energy (n=400), significantly and
> without gaming; it does not dominate any single axis, does not validate for
> tree-based models (0/6 fits), and its advantage over Mondrian is not explained
> by temporal versus random splitting.

On this fifth domain it places 2nd of 9 (temporal) and 3rd of 9 (random) — solid
but not leading, and behind DtACI in both.

## One implementation note

`r_cal` initially reported `NaN` because `rolling_origin_folds` leaves the first
fold unassigned and the correlation was computed over an array containing NaNs.
Fixed with a finite mask. The bug affected only that diagnostic column — the
calibration scores, β fit, both checks, and every interval and Winkler number
were already finite-masked, so no verdict depended on it.

## Files

- `task26_1_domain_selection.md` — criteria, four candidates rejected, final choice
- `task26_2_splits.py` / `.md`, `task26_2_splits.csv` — matched splits, verified
- `task26_3_comparison.py` / `.md`, `task26_3_comparison.csv`,
  `task26_3_significance.csv`, `task26_3_validation.csv`
