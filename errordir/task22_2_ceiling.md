# Task 22, Item 2 — Raising the Bounded-Compensation Ceiling

**New ceiling: HI = 2.79, up from 1.25. Derived from fit-period data only,
before any test comparison.**

Script: `task22_2_ceiling.py` · Data: `task22_2_ceiling.csv`

## The stated rationale, fixed before computing

The multiplier scales the ACI half-width `Q_C`. To cover an error of size `E`
the method needs `HI · Q_C ≥ E`, so the natural fit-period-derived ceiling is

    HI = p99(fit-period |error|) / p90(fit-period |error|)

- **p99 numerator** — the ceiling must reach the fit period's own near-worst
  errors. p99 rather than max, because max is a single unstable point at n=291.
- **p90 denominator** — ACI at α=0.10 operates near the 90th percentile of the
  score distribution, so p90 is what `Q_C` is typically near. The ratio is
  dimensionless: "how many typical half-widths to reach a near-worst error?"

Both quantities come from the **291-month FIT split only**. No CAL data, no test
data, no knowledge of the 2020 cluster.

## The result

| | value |
|---|---|
| p99(fit \|error\|) | 96.32 |
| p90(fit \|error\|) | 34.51 |
| **HI_new = p99/p90** | **2.7909** |
| HI_old | 1.25 |
| LO (unchanged) | 0.75 |

**LO is left at 0.75.** Only the ceiling was diagnosed as the bug, and Task 21
Item 4 found the narrowing risk *correctly targeted* (narrowed months had 10×
lower realized error), so there is no evidence-based reason to move the floor.

## Observation, reported as an observation

The worst Jan–Apr 2020 miss (46.98) against a 6M half-width of ~26.6 would have
needed **HI ≈ 1.76**. The fit-derived ceiling of **2.79 clears that**.

Per the guardrail this is stated as an observation, not validation: the ceiling
was derived from the 291-month fit period with no reference to those four
months, and it landing above what they needed is a stronger result than a value
chosen to land there. It is not evidence that 2.79 is *correct* — only that an
independently-derived ceiling would have had the range to respond.

## The property this breaks, stated explicitly

Raising HI without moving LO **destroys** Task 21's "mean multiplier = 1.0 under
uniform ranks" property. The midpoint is now (0.75+2.79)/2 = 1.77. This was the
safeguard that made a width win impossible by uniform inflation — so with it
gone, the width column of Item 4 is where any inflation must be read, and it is
reported there prominently rather than assumed away.
