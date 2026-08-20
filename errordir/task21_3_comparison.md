# Task 21, Item 3 — The Test: Frontier Escape, Frontier Movement, or Nothing?

## Outcome: **3, with a qualification — no statistically supportable improvement over DtACI, and the one apparent gain is confounded by covariate drift.**

Against the diversity-optimal selector the method looks like outcome 2 (better
coverage +3.39pp, narrower width −0.78) — but that comparison is not clean, and
against **DtACI, the method it actually has to beat, it is essentially a tie that
slightly loses** (Winkler 116.92 vs 116.71). Nothing survives significance
testing: every p-value in the family is ≥ 0.71 nominal, ≥ 0.77 after BH
correction. **Outcome 1 (frontier escape) is firmly rejected.**

Script: `task21_3_comparison.py` · Data: `task21_3_comparison.csv`,
`task21_3_significance.csv`, `task21_3_drift.csv`

Runs at **6M only** — the sole horizon where Item 1 validated β.

## Results

| Method | coverage | mean width | Winkler mean | Winkler median |
|---|---|---|---|---|
| diversity_optimal | 84.75 | 53.279 | 122.774 | 57.035 |
| pooled_trailing | 84.75 | 53.279 | 122.774 | 57.035 |
| **dtaci** | **89.83** | 53.289 | **116.712** | 57.185 |
| **errordir (this method)** | 88.14 | **52.498** | 116.923 | 58.261 |

The method beats the selector on coverage (+3.39pp) *and* width (−0.78), which
in isolation reads as outcome 1 against that baseline. It does not beat DtACI:
lower coverage (−1.69pp), essentially identical Winkler (+0.21, worse), and a
worse Winkler median.

## Significance — nothing survives

| Comparison | mean Winkler diff | % months better | Wilcoxon p | BH q | perm (shift) | perm (sign-flip) |
|---|---|---|---|---|---|---|
| vs diversity_optimal | −5.851 | 40.7% | 0.7115 | 0.7742 | 0.983 | 0.231 |
| vs dtaci | +0.210 | 33.9% | 0.7742 | 0.7742 | 0.530 | 0.524 |
| vs pooled_trailing | −5.851 | 40.7% | 0.7115 | 0.7742 | 0.983 | 0.211 |

The **% months better** column is the most telling number in this report. Even
against the selector, where the mean Winkler difference is −5.85 in the method's
favour, the method is better in only **40.7%** of individual months — i.e. it is
*worse* in the majority, and the mean is carried by a few large wins. That is the
signature of a high-variance effect, not a reliable improvement.

## Why the apparent gain over the selector is not trustworthy

**Covariate drift breaks the design's central constraint.** Item 2 pinned the
mean multiplier to 1.0 so the method could not win by uniform widening. On the
test period:

- test projections have mean rank **0.812**, not ~0.5
- **15.4%** of test months project above *every* calibration month (rank
  saturates at 1.0); 15.4% below all
- realized mean multiplier: **1.156**, not 1.0

So the method widened intervals by ~16% on average. Its +3.39pp coverage gain
over the selector is therefore **partly bought with uniform extra width, not
differential allocation** — the exact mechanism the design was built to exclude.
The width still came out marginally narrower only because ACI's α adapted
downward in response.

This is a genuine distribution shift (β's difficulty scale changed between the
pre-2020 calibration period and the COVID-era test period), not a coding error.
It cannot be corrected without re-centring the multiplier on test projections,
which would leak the test distribution into interval construction.

## A redundancy worth recording

An `errordir_selector` arm (scaling applied on the selector's calibration subset
rather than the full CAL set) produced **byte-identical** results to
`errordir_pooled`. At 6M the two calibration bases yield the same operative
quantile, so they are the same method. It is omitted rather than reported twice
as though it were independent corroboration.

## Skepticism applied, per the guardrail

The guardrail demanded that outcome 1 be met with the same skepticism Task 20
applied to its own tempting result. The selector comparison did initially look
like outcome 1, so: BH correction across the family (min q = 0.774), two
permutation nulls respecting temporal autocorrelation (both non-significant),
a held-out *time* split rather than cross-validation (2020+ never seen during
fitting), and an explicit search for what could be wrong — which found the drift
confound above. It does not survive.
