# Task 33, Item 1 — Direct Measurement of Tree-Forecaster Responsiveness

## Result: **the hypothesis is falsified. Tree forecasters are not unresponsive — they respond to Check 2's perturbations at essentially a 100% rate, and move 2.7× MORE than ridge.**

Script: `task33_1_responsiveness.py` · Data: `task33_1_responsiveness.csv`

Measurement of the forecasters alone. No β, no kNN, no disagreement — no
difficulty function is involved anywhere in this item.

## Method

Identical perturbation to Check 2's: standardized feature space (fit on FIT),
magnitude 0.5 SD, 200 random unit directions, 40 sampled test points per fit.
For each (point, direction) pair, record whether
`|pred(x + 0.5v) − pred(x − 0.5v)|` is exactly zero.

## Results

| domain | model | class | **zero rate** | near-zero rate | median rel. change | dead points |
|---|---|---|---|---|---|---|
| insurance | xgboost | tree | **0.00013** | 0.00013 | 0.187 | **0.000** |
| insurance | lightgbm | tree | **0.00000** | 0.00000 | 0.141 | **0.000** |
| insurance | ridge | linear | 0.00000 | 0.00000 | 0.027 | 0.000 |
| energy | xgboost | tree | **0.00000** | 0.00000 | 0.104 | **0.000** |
| energy | lightgbm | tree | **0.00000** | 0.00000 | 0.088 | **0.000** |
| energy | ridge | linear | 0.00000 | 0.00000 | 0.081 | 0.000 |
| climate | gradboost | tree | **0.00000** | 0.00000 | 0.312 | **0.000** |
| climate | ridge | linear | 0.00000 | 0.00000 | 0.074 | 0.000 |

**By class:** trees zero-rate **0.00003** (three thousandths of one percent),
median relative change **0.166**; ridge zero-rate 0.00000, median relative change
**0.061**.

## What this settles

The pre-registered hypothesis was that tree predictions are piecewise-constant,
so small perturbations mostly land inside the same leaf and produce zero output
change. **That is false at this perturbation magnitude.** Out of 8,000
(point, direction) pairs per fit, essentially none produce zero change, and
**not a single test point in any fit is "dead"** (≥95% of its directions
producing no change).

The reason is structural and, in hindsight, obvious: these are *boosted
ensembles* of 200–250 trees. A perturbation of 0.5 SD across ~12–21 features
simultaneously will cross at least one split in at least one tree with
overwhelming probability. Piecewise-constancy is a property of a *single* tree,
not of a large additive ensemble of them.

**Trees are in fact more responsive than ridge — 2.7× larger median relative
change.** So the failure of Check 2 on trees cannot be attributed to the
forecaster failing to move.

## The actual mechanism, which this measurement reveals

Check 2 compares the effect along the *signal's* direction against a
**random-direction null**. Trees respond strongly to essentially *any* direction,
which makes that null **high and flat**. A candidate direction then has to beat a
much higher bar to reach the 95th percentile — and with the null concentrated
(Task 25 measured gradboost's null at 3× ridge's p95 with *lower* variance), no
single direction stands out.

So Task 32's 0/14 is a **signal-to-noise problem in the null**, not a dead
forecaster. Trees don't respond too little; they respond too indiscriminately for
one direction to distinguish itself.

This is the outcome-2 branch of Item 2, and it means a locally-varying difficulty
function is a genuinely motivated next step rather than a predictable repeat.
