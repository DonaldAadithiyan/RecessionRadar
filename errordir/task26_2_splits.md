# Task 26, Item 2 — Matched Split Construction

**Both conditions draw from an identical 3,600-row set with identical
fit/cal/test counts. Verified, not assumed.**

Script: `task26_2_splits.py` · Data: `data/domains/task26/pm25.csv`,
`task26_2_splits.csv`

## The two constructions

| | fit | cal | test |
|---|---|---|---|
| **temporal** (chronological) | 1800 | 1200 | 600 |
| **random** (shuffled) | 1800 | 1200 | 600 |

Both are carved from the **same** most-recent 3,600 usable rows, so the only
difference between conditions is how rows are *assigned*, never which rows exist
or how many.

## Verification (the guardrail)

| check | result |
|---|---|
| matched sizes across conditions | **True** |
| identical underlying row set | **True** |
| fit/cal/test disjoint (temporal) | **True** |
| fit/cal/test disjoint (random) | **True** |
| temporal TEST strictly after FIT | **True** |
| random TEST interleaved with FIT | **0.50** fraction before FIT median (≈0.5 expected) |

The last two rows are the ones that make the experiment meaningful: the temporal
condition genuinely tests the future from the past, and the random condition
genuinely interleaves.

## Feature construction, and why it matters more here than usual

All predictors are **strictly lagged** (`shift ≥ 1`): PM2.5 at lags 1/2/3/24,
24-hour and 168-hour rolling means, lag-1 weather (dew point, temperature,
pressure, wind, snow, rain, wind direction), plus hour/month cyclical encodings.

This is essential for the **random** condition specifically. Without lagging, a
random shuffle would place hour *t−1* in the fit set and hour *t* in the test
set, letting contemporaneous weather leak the neighbouring target. The random
condition would then look artificially easy and the whole comparison would be
meaningless. Lagging removes that channel.

22,914 rows survive lagging; the most recent 3,600 are used.

## Post-hoc confirmation the manipulation worked

Item 3 measures the lag-1 autocorrelation of the paired Winkler differences in
each condition — a direct check that the shuffle did what it was supposed to:

| condition | lag-1 autocorr of paired differences |
|---|---|
| temporal | **0.187 – 0.500** |
| random | **0.027 – 0.150** |

The temporal condition retains substantial serial dependence; the random
condition has little. The two conditions are genuinely different experiments on
identical data, which is exactly what this task needed.
