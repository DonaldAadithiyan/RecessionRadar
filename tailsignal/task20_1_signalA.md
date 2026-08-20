# Task 20, Item 1 — Signal A: Structural Supply-Demand Margin

**Status: built, exactly verified. (Not validated as a miss predictor — see Item 3.)**

Script: `task20_1_signalA.py` · Verification data: `task20_1_signalA_verify.csv`

## Definition

At each prediction step, with active calibration budget `N` and ACI's current
target quantile `q_t = 1 - alpha_t`:

| Quantity | Definition |
|---|---|
| **Demand** | `ceil((1 - q_t) * N)` — top-ranked points that determine the `q_t`-quantile |
| **Supply** | upper-tail points the active selection rule guarantees |
| **Margin** | `Supply / Demand` |

Supply is *computed, not assumed*, per the spec. For the alternating-tail
selector it is `floor(N/2)`, because the rule takes points alternately from the
low and high tails. For the trailing/pooled baseline no structural guarantee
exists, so it is derived from that set's own score ranking: the count of its
points at or above the pool median.

## Hand-verification (the guardrail's requirement)

**1 — Claimed supply vs. what `selector_lib.support_width_selector` actually picks.**
Run the real selector on a known pool and count picks landing in the upper half:

| N | 20 | 30 | 50 | 80 | 120 | 180 | 254 |
|---|---|---|---|---|---|---|---|
| claimed `floor(N/2)` | 10 | 15 | 25 | 40 | 60 | 90 | 127 |
| actual upper-tail picks | 10 | 15 | 25 | 40 | 60 | 90 | 127 |

All match. Supply is exact, not an approximation.

**2 — Against Section 4.4's own ceiling-ratio sweep** (`task12a_small_n_ceiling.csv`),
84 cells across the operative quantiles {0.868, 0.900, 0.927}. Wherever that
sweep reports the guarantee attained (`ratio = 1.0`), the margin must be >= 1 —
a margin below 1 would mean the selector cannot supply the points the quantile
needs. **Inconsistent cells: 0 / 84.**

Worked examples for hand-checking:

```
N=254, q=0.900: demand=ceil(0.100*254)= 26, supply=floor(254/2)=127, margin=4.885
N= 20, q=0.900: demand=ceil(0.100* 20)=  2, supply=floor( 20/2)= 10, margin=5.000
N=254, q=0.868: demand=ceil(0.132*254)= 34, supply=floor(254/2)=127, margin=3.735
```

## Boundary behaviour

The spec asks the signal to flag proximity to the proof's boundary. It does so
by construction — at `N=254`:

| q | 0.99 | 0.95 | 0.927 | 0.90 | 0.868 | 0.80 | 0.70 | 0.60 | 0.55 | 0.51 | 0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| margin | 42.33 | 9.77 | 6.68 | 4.89 | 3.74 | 2.49 | 1.65 | 1.25 | 1.10 | 1.02 | **1.000** |

The margin reaches exactly 1.000 at `q = 0.5` — precisely where Section 4.4 says
the guarantee degrades. This was not fitted; it falls out of
`ceil((1-q)N)` meeting `floor(N/2)` at `q=0.5`, and is independent confirmation
the derivation is right.

## The limitation Item 3 exposed

In live operation ACI's alpha stays in a narrow band, so `q_t` never approaches
0.5. Observed margin ranges across all horizons: **4.10 to 6.89** — comfortably
in the regime where the proof is exact, never near the boundary where the signal
would be informative. Signal A is a correct instrument pointed at a condition
that does not occur on this testbed. See `task20_3_validation.md`.

No fitting, no free parameters.
