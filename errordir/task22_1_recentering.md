# Task 22, Item 1 — Rolling Recentering Fix

**Result: implemented and leak-free, but it does NOT fix the confound. Mean
multiplier moved only from 1.156 to 1.143 against a design target of 1.000.
The residual gap has a specific, identified cause.**

Script: `task22_1_recentering.py` · Data: `task22_1_recentering.csv`,
`task22_1_leakcheck.txt`

## What was changed

Task 21 froze the projection CDF once, on the fit/CAL period. Task 22 recomputes
it online: at test step *t* the reference window is

    254 CAL projections  +  test projections from steps STRICTLY < t

i.e. a strictly backward-looking expanding window.

**Why this is legitimate and not the leak Task 21 rejected.** A projection is
β·x — it needs *input features only*, never the outcome. At deployment, the
inputs for months up to *t−1* are published macro data and genuinely available.
Task 21's rejected version would have used the *whole* test period's
projections, including future ones, to set the CDF. This version cannot see step
*t* or beyond.

## Leakage check (the guardrail's requirement)

Same class of check as Task 20's audit — corrupt the present and all future
projections, confirm nothing upstream changes:

| Probe step | 0 | 5 | 15 | 30 | 45 | 64 |
|---|---|---|---|---|---|---|
| reference window identical after corrupting steps ≥ t | True | True | True | True | True | True |

- reference windows unaffected by present/future: **PASS**
- projection path touches any test outcome: **NO (inputs only)**

*(An initial version of this check reported "YES — LEAK". That was a flaw in the
check, not the code: a naive substring grep matched `y_pool` — the training
outcome legitimately used to fit β on the FIT split — and matched its own
source line. Corrected to ask the precise question: does the projection path
touch any *test* outcome. It does not.)*

## Why the fix underperforms

| | mean rank | mean multiplier | frac saturated high | frac saturated low |
|---|---|---|---|---|
| Task 21 (frozen CDF) | 0.812 | 1.156 | 15.4% | 15.4% |
| Task 22 (rolling) | 0.786 | **1.143** | **3.1%** | 4.6% |

Saturation collapses from 15.4% to 3.1% — the rolling window genuinely fixes the
*clipping* problem. But the mean barely moves, and the reason is visible by
period:

| test period | mean rank | mean multiplier |
|---|---|---|
| steps 0–22 | **0.467** | **0.983** |
| steps 22–44 | 0.967 | 1.233 |
| steps 44–65 | 0.932 | 1.216 |

The first third recenters almost perfectly (0.983 ≈ the 1.0 target). Then it
fails, because the **expanding window permanently accumulates the extreme
early-2020 projections** (−110 to −96, far below anything in CAL). Once those
enter the reference set they never leave, so every subsequent month ranks above
them and the mean rank is dragged upward.

## A tuning opportunity deliberately declined

A bounded (rolling-window) reference would partly fix this:

| window | ∞ (expanding) | 400 | 254 | 120 | 60 |
|---|---|---|---|---|---|
| \|mean mult − 1\| | 0.143 | 0.143 | 0.140 | 0.126 | **0.094** |

Window=60 is the best of these — and selecting it *because* it minimises
deviation on the test period would be tuning a parameter against test data, the
exact violation this task exists to remove. The expanding window is the spec's
stated design and has no free parameter, so it is what ships. The residual
deviation is reported rather than optimised away.
