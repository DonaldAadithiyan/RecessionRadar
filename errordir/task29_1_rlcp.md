# Task 29, Item 1 — RLCP Implemented Per the Published Method

## Task 28's "RLCP" was not RLCP. It was baseLCP — the un-randomized variant the paper explicitly contrasts against.

Script: `task29_1_rlcp.py` · Data: `task29_1_rlcp_verify.csv`

The Item 1 guardrail says to reuse Task 28's implementation if correct, and to
implement as published if it differs materially. Checked against
[Hore & Barber (arXiv:2310.07850)](https://arxiv.org/html/2310.07850v2), Task 28's
version is missing the two features that **define** RLCP:

| | published RLCP | Task 28's version |
|---|---|---|
| localization point | **sampled** `X̃ ~ H(X_{n+1}, ·)` | deterministic, centred at the test point |
| weight normalisation | over **n+1** points | over n calibration points only |
| test-point mass | `w̃_{n+1}·δ_{+∞}` in the quantile | **omitted** |

The randomization is what the paper's title refers to and what buys marginal
validity; the `+∞` atom is what makes the interval conservative rather than
anti-conservative. Without them the construction is **baseLCP**, which the paper
presents as the thing RLCP fixes.

So Task 28's reported "RLCP" numbers (climate Winkler 10.799) are baseLCP
numbers. This file implements the published algorithm and retains Task 28's
construction as `baselcp_intervals` so both can be reported side by side.

## The implemented algorithm

1. Sample `X̃ ~ N(z_test, 1/(2γ))` — the Gaussian localization kernel.
2. `w̃_i ∝ exp(−γ(z_i − X̃)²)` for **i = 1…n+1**, normalised over all n+1.
3. `q̂ = Quantile_{1−α}( Σ_{i≤n} w̃_i δ_{s_i} + w̃_{n+1} δ_{+∞} )`.
4. Interval = `pred(x) ± q̂`. If the finite atoms never reach 1−α, the quantile
   **is** +∞ and the interval is unbounded — handled explicitly, not clipped.

The kernel operates in the **same 1-D coordinate `z = β′x`** errordir uses. This
is deliberate: it isolates "kernel-weighted local quantile" vs "rank-based fixed
multiplier" on identical geometry rather than confounding mechanism with
representation.

## Correctness check

On a synthetic problem where score spread grows with z:

| method | coverage (target 0.90) | mean width |
|---|---|---|
| **RLCP (published)** | **0.900** | 6.323 |
| baseLCP (Task 28's) | 0.920 | 5.940 |

RLCP hits nominal exactly. baseLCP over-covers here, consistent with it being a
different estimator rather than a broken one.

## Bandwidth, fixed without test-period tuning

γ is set by the standard **median heuristic on FIT-split projections only**
(`γ = 1/(2·median pairwise squared distance)`). Not tuned against test results —
the discipline every parameter has followed since Task 22.

A sensitivity band (γ × {0.25, 0.5, 1, 2, 4}) is also run so RLCP is not
disadvantaged by one bandwidth, and the **best** RLCP over that band is reported
alongside the default. This is deliberately generous to RLCP: the burden of proof
here is on errordir.

Note on the sweep: on climate, `rlcp_best_gamma` equals the default γ — not
because the sweep failed, but because γ×2 and γ×4 produce unbounded intervals,
leaving Winkler mean undefined and disqualifying them from being "best."
