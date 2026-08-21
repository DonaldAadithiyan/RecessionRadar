# Task 29 — The Decisive Comparison: Original errordir vs RLCP

## **Outcome 2: errordir ties RLCP.** On climate the two are statistically indistinguishable (Winkler 11.220 vs 11.292, difference 0.6%, win rate 54.7%, BH q=0.556). errordir has independently converged on a position RLCP already occupies. This is not "beats current methods."

The energy domain does not overturn this. errordir is clearly better there
(86.25% coverage, zero unbounded intervals, versus RLCP's 74–77% coverage among
bounded intervals with 27 of 400 unbounded) — but RLCP **breaks down** on that
domain rather than losing a fair contest, and a Winkler mean cannot even be
computed for it. A method failing under distribution shift is a real and
reportable difference, but it is not the same as winning the combined tradeoff
against a functioning competitor.

## The three outcomes, resolved

| | | |
|---|---|---|
| 1. errordir beats RLCP | ✗ | Climate difference is 0.6% at q=0.556. Energy is a breakdown, not a win. |
| **2. errordir ties RLCP** | **✓** | **Climate: −0.072 on ~11.2, win rate 54.7%, q=0.556.** |
| 3. errordir loses to RLCP | ✗ | errordir is never significantly behind. |

## A correction that changes what Task 28 reported

**Task 28's "RLCP" was not RLCP.** Checked against the paper, it omitted the two
features that define the method:

- the **randomization step** (sampling `X̃ ~ H(X_{n+1},·)` and centring weights
  there) — the paper's title is "randomization enables robust guarantees"
- the **`+∞` atom** and normalisation over **n+1** rather than n points

Without these it is **baseLCP**, the un-randomized variant the paper contrasts
RLCP against. I implemented the published algorithm (verified: exactly 90.0%
coverage on a synthetic control) and kept Task 28's version alongside it.

This matters for the record: on climate, baseLCP scores **10.723** — better than
both errordir (11.220) and true RLCP (11.292). Task 28's reported "tie with RLCP"
was a tie against a sharper, guarantee-free estimator. The published method,
which is what prior work actually offers, is slightly *worse* on Winkler because
its guarantees cost sharpness.

## What this means for the project's claim

The honest position, stated at this project's precision standard:

> On climate, errordir reaches a coverage-width frontier position statistically
> indistinguishable from RLCP — a published method that achieves it by a
> different mechanism and carries marginal-validity and covariate-shift
> guarantees errordir lacks. errordir is more robust than RLCP under the
> distribution shift present in the energy domain, where RLCP produces unbounded
> intervals and under-covers.

Two things remain genuinely errordir's own, and neither is affected by this
result:

1. **The data-efficiency result (Task 28 Item 2)** — errordir at n_cal=50 beats
   every baseline at full calibration (11×/32×). RLCP was not in that sweep; a
   follow-up should add it, since kernel localization plausibly degrades faster
   under scarcity.
2. **Robustness under regime change** — errordir's ACI anchor adapts online.
   RLCP, CQR (Task 27) and q_α(z) (Task 28) all fail on energy for the same
   structural reason: they freeze a calibration-period estimate.

What is **not** available is "errordir beats current methods on the combined
tradeoff." On the one external method close enough in mechanism to be decisive,
it ties.

## Files

- `task29_1_rlcp.py` / `.md` — published RLCP, synthetic control, bandwidth policy
- `task29_1_rlcp_verify.csv` — bandwidth sensitivity, both domains
- `task29_2_comparison.py` / `.md`, `task29_2_comparison.csv`,
  `task29_2_significance.csv`
