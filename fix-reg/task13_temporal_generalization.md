# Task 13 — Temporal Generalization

**Verdict: the mechanism is TEMPORALLY GENERAL. Diversity outpredicts
rare-event count in 12 of 12 (cutoff × horizon) cells, across three independent
recession episodes spanning 1990 to 2020. The pre-registered "temporally
general" criterion is met.**

This closes the sharpest untested objection to the paper: that the
calibration-diversity finding might be an artifact of the specific post-2020
episode (COVID + tightening) it was validated on.

Script: `fix-reg/task13_temporal_generalization.py`.
Spec: `task13_temporal_generalization_spec.md`.
Data: `task13_temporal_cutoffs.csv`, `task13_selector_by_cutoff.csv`.

---

## Result

`gap = ρ(support) − ρ(rare-count)`; positive means diversity dominates.

| Horizon | 1989 cutoff (1990–91 recession) | 2006 cutoff (2007–09 crisis) | 2020 cutoff (COVID) |
|---|---|---|---|
| Current | +0.138 | +0.095 | +0.016 |
| 1M | +0.206 | +0.242 | +0.093 |
| 3M | +0.364 | **+0.484** | +0.297 |
| 6M | **+0.434** | +0.389 | +0.353 |

**12 of 12 cells positive.** The relationship is not episode-specific.

Full detail, showing that each cutoff tested a genuinely different regime:

| Cutoff | Train months | Rare in train | Rare in test | N_cal | 6M ρ(supp) | 6M ρ(rare) |
|---|---|---|---|---|---|---|
| 1989-01 | 263 | 29 | 5 | 105 | 0.601 | 0.167 |
| 2006-01 | 467 | 34 | **16** | 186 | 0.695 | 0.306 |
| 2020-01 | 635 | 50 | 2 | 254 | 0.688 | 0.335 |

### The 2006 cutoff is the most important row

It has **16 rare months in its test window** — eight times the published split's
2 — making it the best-powered episode in the paper. Its 3M gap (**+0.484**) is
the *largest* of any cutoff, exceeding the published split's +0.297.

That matters because the natural worry was that COVID's extremity manufactured
the effect. The opposite holds: the effect is *stronger* in the
financial-crisis window, which contains far more rare-event mass and is the
episode most representative of an ordinary recession.

The within-tertile redundancy check also holds everywhere (0.109–0.334): once
support width is fixed, rare-event count retains only weak predictive power at
every cutoff.

## The selector's temporal stability

| Cutoff | Horizon | Trailing | Selector | Gain | Width |
|---|---|---|---|---|---|
| 1989 | 6M | 98.46 | 100.00 | +1.54 | 1.61× |
| 2006 | 3M | 69.23 | 83.08 | **+13.85** | 4.45× |
| 2006 | 6M | 70.77 | 80.00 | +9.23 | 4.73× |
| 2020 | 6M | 77.97 | 89.83 | **+11.86** | 2.79× |

The selector improves coverage at **12 of 12** cutoff × horizon combinations,
with the largest gains where the baseline is worst (2006 3M: 69.23 → 83.08).
This is the mechanism working as designed — the selector helps most exactly
where undercoverage is most severe.

**The 1989 cutoff is close to degenerate for the selector**: its trailing
baseline already sits at 92–98%, so there is little headroom, and the selector
pushes to 100% at modest width. Read that column as "does no harm", not as
evidence of benefit.

## Methodological notes that constrain the reading

**Every cutoff refits from scratch, and the saved ensemble is never used.** It
was fit through 2019-12, so scoring a 1990 test window with it would leak thirty
years of future information. All three cutoffs use a surrogate RegressorChain
refit on their own training rows — including 2020, so the comparison is
like-for-like.

**Consequence, as the spec anticipated:** the 2020 arm does not reproduce the
published Phase 1 numbers exactly.

| Horizon | Published ρ(supp) / ρ(rare) | Task 13 2020 arm |
|---|---|---|
| 3M | 0.683 / 0.331 | 0.567 / 0.270 |
| 6M | 0.913 / 0.098 | 0.688 / 0.335 |

Direction agrees at every horizon; magnitudes differ because the published
numbers use the saved ensemble with **in-sample** scores while this uses a refit
surrogate with **out-of-fold** scores. Both differences push the same way and
were expected. The reproduction gate was defined on direction, not magnitude,
and it passed.

**STL features are computed leak-free per cutoff.** STL is a global smoother, so
one fit over the whole frame would let post-cutoff months shape pre-cutoff
features *and* let future months inform test rows. Instead the decomposition is
fit on training rows for training features, then **expanded one month at a
time** across the test window, so each test row uses only data available by that
month. Cost was measured before adopting this (~0.1 minutes for 2,340 fits), so
the honest construction was effectively free.

## Why only three cutoffs

The spec's original fourth cutoff (1999, testing the 2001 recession) **was
dropped after checking the data**: the 2001 recession peaks at **31.3%** in this
smoothed series, so its 65-month test window contains **zero** months above the
paper's ≥50 rare threshold. A window with no rare events cannot discriminate
between diversity and rare-count.

Rare months (≥50) by decade, which shows how unevenly they fall:

| 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|
| 0 | 12 | 17 | 5 | 16 | 0 | 2 |

This also rules out evenly-spaced cutoffs: the 2010s contain no rare months at
all. Cutoff spacing must follow the rare events, not the calendar.

## Honest caveats

- **Three episodes is a small sample.** It is what the series supports at the
  paper's own rare-event definition; claiming more would mean lowering the
  threshold mid-paper and changing what "rare" means.
- **The 1989 cutoff is underpowered** — 263 training months, 5 rare months in
  test, and a baseline already near nominal. Its positive gaps are directionally
  consistent but should not carry weight independently.
- **Cutoffs are not independent.** They share the same underlying series and
  overlapping training data (1989's training period is a subset of 2006's), so
  these are not three independent replications in the statistical sense.
- **Correlations, not confidence intervals.** The gaps are point estimates from
  200 draws each; no interval is placed on the gap itself. The consistency of
  sign across 12 cells is the evidence, not any single value.
- **This tests the diagnostic, not the full baseline horse race.** Running all
  nine strategies at every cutoff was out of scope and would not have answered
  the question this task exists for.
- **The 2020 arm's numbers here supersede nothing.** The published Table 8
  figures remain correct for the published pipeline; these are a differently
  constructed comparison built for internal consistency across cutoffs.
