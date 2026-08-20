# Task 22, Item 3 — Diagnosis of the 2022 Blind Spot (No Fix)

**Verdict: CASE 2 — a pattern within the fit distribution that β should have
caught and didn't. This is a real miss, not an extrapolation failure.**

Script: `task22_3_2022.py` · Data: `task22_3_2022.csv`

## The evidence

2022 had the highest mean realized error of any test year (30.69 vs 5.9 average
elsewhere) yet β flagged zero cases. Comparing 2022's inputs against the
291-month fit-period feature distribution:

| year | n | mean error | cells beyond 3 fit-SD | centroid dist/feature | proj within fit range |
|---|---|---|---|---|---|
| 2020 | 12 | 9.98 | 49.2% | 6.64 | 33.3% |
| 2021 | 12 | 1.78 | 36.4% | 10.15 | 41.7% |
| **2022** | 12 | **30.69** | **29.9%** | **9.41** | **25.0%** |
| 2023 | 12 | 3.74 | 26.5% | 8.60 | 8.3% |
| 2024 | 12 | 8.05 | 28.4% | 8.16 | 41.7% |

Against the other test years on every relative measure:

| criterion | 2022 | other years | verdict |
|---|---|---|---|
| cells beyond 3 fit-SD | 0.299 | 0.340 | **in line** (actually *fewer*) |
| centroid distance/feature | 9.410 | 8.453 | **in line** |
| projections within fit range | 0.250 | 0.250 | **in line** (identical) |

2022 is not an outlier on any of them. Its inputs look like every other test
year's. What differs is that **the base ensemble was wrong on ordinary-looking
inputs** — and β, which is fit to predict error magnitude from inputs, had no
input-side signal to work with.

## Two corrections to the analysis worth recording

**1. Degenerate features destroyed the first pass.** Two features
(`CPI_anomaly`, `share_price_anomaly`) are *constant* over the fit period, so any
nonzero test value produces an infinite z-score. The first run reported
`max_abs_z = 1e12` and centroid distances of ~7×10¹⁰, which made every
comparison meaningless. They are excluded from the continuous metrics and
reported separately — their activation rate is itself a novelty signal (2022:
33.3% vs 17.8% elsewhere, the one measure where 2022 does stand out mildly).

**2. The automated verdict initially fired CASE 1, wrongly.** The first
criterion included an absolute test (`proj_within_fit_range < 0.5`) — but *every*
test year fails that (8–42%), so it flagged all of them and discriminated
nothing. Corrected to compare 2022 against the other test years, which reverses
the verdict to CASE 2. This is worth flagging because the wrong verdict was the
convenient one: CASE 1 would have excused the blind spot as unavoidable
extrapolation.

## What this means, and the recommendation

CASE 2 means the blind spot is **potentially correctable but not by this
method's current shape**. β maps inputs → expected error magnitude. When the
base model fails on inputs that look ordinary, there is nothing in *x* to signal
it, so no β fit on *x* alone can anticipate it — regardless of ceiling or
recentering.

A fix would need a signal source β currently ignores: recent realized residuals
(which is Task 20's Signal B, already found underpowered on this testbed),
model-internal disagreement (ensemble spread, available in this project), or
base-model confidence. **Per the guardrail, no fix is attempted here.** The
recommendation for a future task is ensemble-disagreement as the most promising
input, because it is already computed in this project (Task 4) and is
outcome-free at prediction time — but that belongs in a scoped, guarded task,
not bolted onto this one.

**A caveat on the whole diagnosis:** n=12 months for 2022. These are
year-level aggregates over a single year, and the "in line" conclusions rest on
comparisons between 12-month buckets. The direction is consistent across all
three independent measures, which is why the verdict is stated, but it is not a
high-powered test.
