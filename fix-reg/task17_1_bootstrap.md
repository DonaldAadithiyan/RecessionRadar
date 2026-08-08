# Task 17 Item 1 — Bootstrap Uncertainty on the Headline R² Result

**Verdict: the six-month result is decisive; the three-month result is not. At
6M, ΔR² = 0.392 with a 95% interval of [0.167, 0.574] that excludes zero in
1000 of 1000 replicates. At 3M the interval is [−0.012, 0.396] — it crosses
zero, with P(Δ>0) = 0.960. The paper's most-quoted number now has uncertainty
attached, and one of the two horizons is weaker than the point estimates
implied.**

Script: `fix-reg/task17_135_boundary_bootstrap_qc.py`.
Data: `task17_1_bootstrap.csv` (full distributions), `task17_1_bootstrap_summary.csv`.

---

## Method

B = 1000 (the spec's preferred value; timing was checked first — ~11 minutes for
both horizons, so the B = 500 fallback was not needed).

**The pool is resampled, not the draws.** The 200 calibration draws are derived
from the 635-month pool, so bootstrapping them directly would double-count that
dependency. Each replicate instead:

1. block-resamples the 635-month pool (block = 12 months, matching the paper's
   autocorrelation-preserving choice elsewhere);
2. regenerates 200 fixed-size (N = 254) calibration draws from the resample;
3. recomputes R²(support width) and R²(rare count) against coverage.

Replicates whose coverage was constant across all 200 draws are dropped
(B_effective = 996 at 3M, 1000 at 6M).

## Result

| Horizon | B_eff | R²(support) median [95%] | R²(rare) median [95%] | **ΔR² median [95%]** | Excludes 0 |
|---|---|---|---|---|---|
| 3M | 996 | 0.348 [0.061, 0.519] | 0.120 [0.008, 0.329] | **0.205 [−0.012, 0.396]** | **No** |
| 6M | 1000 | 0.448 [0.203, 0.651] | 0.046 [0.001, 0.170] | **0.392 [0.167, 0.574]** | **Yes** |

P(ΔR² > 0) across replicates: **3M = 0.960, 6M = 1.000.**

## Reading this honestly

**6M is a strong result.** The interval on ΔR² is [0.167, 0.574] — wide, but
entirely above zero, and every one of 1000 replicates puts support width ahead.
The headline claim survives resampling that respects the pool's autocorrelation.

**3M is suggestive, not established.** Its lower bound sits at −0.012: a hair
below zero. 96% of replicates favour support width, which is real evidence, but
the 95% interval does not exclude the null and the paper should not describe
3M as demonstrating the separation.

**The intervals are wide, and that is the honest finding.** The spec anticipated
this: *"a wide interval around a still-clearly-separated R² pair is a different,
weaker finding than a tight one, and the paper should say which it got."* It got
the wide version. R²(support) at 6M ranges [0.203, 0.651] across replicates —
the point estimate of 0.448 is not precise, and the published 0.85 (from the
in-sample Phase 1 analysis) sits well outside this out-of-fold interval, which
is consistent with everything Task 3 established about in-sample optimism.

## What the paper should change

1. **Quote ΔR² with its interval**, not R² values alone. At 6M:
   *ΔR² = 0.392, 95% CI [0.167, 0.574]*.
2. **Do not claim the separation at 3M.** Report it as directionally consistent
   (P = 0.96) with an interval that includes zero.
3. **Stop quoting 0.85 vs 0.02 as the headline** unless it is explicitly labelled
   in-sample. The out-of-fold, resampled equivalent is 0.448 vs 0.046 — the same
   qualitative story, an order of magnitude less dramatic in the ratio.

## Honest caveats

- **Block bootstrap with a fixed block length of 12.** Autocorrelation in the
  score series is not exactly 12-month; a different block length would shift the
  intervals somewhat. 12 was chosen for consistency with the rest of the paper,
  not fitted.
- **The resample changes the rare-event count**, since blocks are drawn with
  replacement. That is intended — it propagates uncertainty in the pool's
  composition — but it means each replicate's pool is not a permutation of the
  original.
- **Dropped replicates are few but not zero** (4 at 3M). They are cases where
  coverage was constant across all 200 draws, leaving no correlation to compute;
  dropping them slightly favours replicates with more variance.
- **This bootstraps the diagnostic, not the selector.** It quantifies uncertainty
  in "support width predicts coverage better than rare count", not in any
  coverage number the selector achieves.
- **Only 3M and 6M were run**, per the spec. Current and 1M are saturated or
  near-saturated and would not have produced meaningful intervals.
