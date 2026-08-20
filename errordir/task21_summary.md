# Task 21 — An Error-Direction Signal for Dynamic Interval Width

## Outcome: **3 (no supportable improvement)** — with one genuine positive finding inside it

The hypothesis under test was that a learned error-direction signal improves
coverage and width simultaneously against both the diversity-optimal selector
and DtACI. **It does not.** Against DtACI — the method that actually had to be
beaten — this is a tie that slightly loses (Winkler 116.92 vs 116.71, coverage
88.14% vs 89.83%). Against the selector it looks like outcome 2, but that
comparison is confounded by covariate drift and does not survive testing: every
p-value in the family is ≥ 0.71 nominal and ≥ 0.77 after BH correction.
**Outcome 1 is firmly rejected**, consistent with the paper's own
coverage-sharpness frontier framing.

This is reported as outcome 3 rather than dressed as outcome 2 because the
selector comparison — the only one that looks good — is exactly the one with a
mechanical explanation that isn't the proposed mechanism.

**The one real positive result:** where β was confident enough to *narrow* an
interval, it was right to. The method narrowed 12 of 59 months, and those months
had mean realized error **1.35 vs 13.33** elsewhere — a 10× separation. The
method's distinctive new risk never materialized.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Fit and validate β | **Run, gates hard** | Validates at **6M only** (1/4 horizons). Check 1 (not-just-variance) passes everywhere; Check 2 (perturbation) fails at Current/1M/3M. |
| 2 — Width scaling | **Run, pre-registered** | Quantile-mapped, ±25% band, mean multiplier pinned to 1.0 so width cannot be won by uniform shrinkage. |
| 3 — The test | **Run, outcome 3** | No supportable gain over DtACI; selector gain confounded by drift; nothing survives correction. |
| 4 — Failure stress-test | **Run** | Narrowing correctly targeted (10× error separation). Separate failure mode: 100% clustered in COVID onset. |

## What actually decided this

**Item 1's gate did most of the work.** Three of four horizons never got a
validated β. The pattern is informative rather than arbitrary: Check 2's
percentile rises monotonically with horizon (0.370 → 0.595 → 0.925 → 0.975),
tracking held-out correlation (0.109 → 0.151 → 0.349 → 0.367). Error magnitude is
genuinely more predictable at longer horizons, and only 6M clears the bar. The
fit-vs-held-out gap at Current (r = 0.912 → 0.109) shows how badly a
fit-correlation-only report would have misled — which is why the three-way
separation was built before anything else.

The design choice the proposal rests on **did work**: fitting β as a continuous
regression on all 291 FIT months, rather than a classifier on 2–3 rare labels,
avoided the scarcity trap. The method failed for a different reason.

**Covariate drift broke the design's central safeguard.** Item 2 pinned the mean
multiplier to 1.0 specifically so a width win could not come from uniform
shrinkage. On the test period the mean multiplier came out at **1.156**: test
projections have mean rank 0.812 instead of ~0.5, with 15.4% saturating above
every calibration month. β's difficulty scale shifted between the pre-2020
calibration window and the COVID-era test window. So the coverage gain over the
selector is partly bought with ~16% uniform extra width — the mechanism the
design existed to exclude. It cannot be fixed without re-centring on test
projections, which would leak the test distribution into interval construction.

**The most informative single number is 40.7%** — the share of individual months
where the method beats the selector, despite a mean Winkler difference of −5.85
in its favour. It loses in the majority of months and is carried by a few large
wins. A mean-only report would have called this a win.

## What would be needed to revisit this

Not a wider multiplier band and not a different scaling function — the binding
constraint is that β only validates at one horizon, and that the rank map's
meaning is not stable across a regime break. A serious version would need a
drift-aware mapping (recalibrating F on a rolling basis, which needs resolved
outcomes and so changes the guarantee story), and enough post-break data to fit
and test β within a single regime. Neither is available here.

## Files

- `task21_1_beta.py` / `.md` — β fit, three-way separation, both checks
- `task21_1_beta.csv`, `task21_1_perturbation.csv`
- `task21_2_scaling.py` / `.md` — pre-registered scaling design
- `task21_3_comparison.py` / `.md` — three-way comparison
- `task21_3_comparison.csv`, `task21_3_significance.csv`, `task21_3_drift.csv`
- `task21_4_failure.py`, `task21_4_failure_analysis.md`, `task21_4_failures.csv`
