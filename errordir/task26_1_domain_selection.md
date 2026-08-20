# Task 26, Item 1 — Domain Selection

**Selected: Beijing PM2.5 air-pollution severity (UCI, 2010–2014 hourly).
Committed before any evaluation against the method.**

## Criteria (Task 25's four, plus this task's new fifth)

1. Publicly available, no access barrier.
2. Genuine rare/severe-event structure.
3. Real named practical domain where these models are standard.
4. Large enough to avoid repeating the n=59 power problem.
5. **NEW:** enough data *and* a genuine chronological ordering to support
   constructing both a temporal split and a matched random split from the same
   rows — ruling out domains where "temporal" was never a natural property.

## Candidates considered

| Candidate | Source | Verdict |
|---|---|---|
| **Beijing PM2.5** | UCI PRSA 2010–2014 | **SELECTED** |
| `sulfur` (OpenML 23515) | OpenML | **REJECTED — criterion 3/5.** 10,081 rows but columns are anonymized (`a1..a5`, `y1`, `y2`) with no interpretable timestamp. Not a named practical domain and has no usable chronological order. |
| NOAA storm events (daily re-cut) | already in this project | **REJECTED — confounded.** Climate is one of the four domains that *generated* the hypothesis. Re-cutting the same source at finer granularity would not be an independent test. |
| `nyc-taxi-green` (OpenML 41187) | OpenML | **REJECTED — criterion 2/4.** 2,225 rows; the target is CO2, smooth rather than rare-event. |
| `wind-power` (OpenML 42712) | OpenML | **REJECTED — criterion 2.** Resolves to bike-sharing-style hourly demand, already rejected in Task 25 for having smooth diurnal/seasonal structure rather than rare-event structure. |

## The selected domain

| | |
|---|---|
| **Target** | hourly PM2.5 concentration (µg/m³) |
| **Rows** | 43,824 hourly records; **41,757** with a non-missing target |
| **Chronology** | explicit `year`/`month`/`day`/`hour` columns, contiguous 2010-01-01 → 2014-12-31 |
| **Features** | DEWP (dew point), TEMP, PRES (pressure), cbwd (wind direction), Iws (cumulated wind speed), Is (snow hours), Ir (rain hours), plus strictly lagged PM2.5/weather terms |
| **Rare-event structure** | p50 = 72, p90 = 221, p99 = 420, max = 994. **Tail ratio p99/p50 = 5.8**; **2.11%** of hours exceed 5× the median |
| **Practical fit** | air-quality forecasting is a standard tabular-regression application; severe-episode prediction is the operational question |

## Why this domain suits the specific test

The hypothesis under test is that **split methodology** — temporal vs random —
is the operative variable behind Mondrian's win/loss pattern. That requires a
domain where a temporal split is genuinely meaningful (there must be real
distribution shift over time for the temporal condition to differ from the
random one). Beijing PM2.5 has strong seasonal structure (winter heating
episodes) and multi-year drift across 2010–2014, so the two conditions should be
genuinely different rather than nominally different.

It is also fully independent of all four domains that generated the hypothesis
(recession, climate, healthcare, insurance/energy) — different field, different
data source, different error distribution.

## Commitment

Per the Item 1 guardrail, this domain is **committed now, before evaluation
against the method**. No swap to another candidate will be made after seeing a
weak or inconvenient result. Both split conditions will be reported regardless
of outcome.
