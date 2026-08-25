# Task 34, Item 3 — Full Retest: Check 2 vs Directional Specificity

> **⚠ SUPERSEDED IN PART — see `task36_3_correction.md` (Task 36).**
> The four **ridge** rows in this report are an instrument artifact. Task 36
> measured δᵢ directly on those fits and found **one unique value across 400 test
> points** (CV ≈ 2×10⁻¹⁵): the ridge prediction path is three composed affine
> maps, so δᵢ has no xᵢ dependence and Spearman(δ, e) is tie-degenerate. The
> conclusion drawn from them — *"Check 2 certifies magnitude alignment, not
> difficulty relevance"* — is **withdrawn**.
> The **tree** results (0/16 under DS), the synthetic verification, and the n=40
> power finding are **unaffected and stand**.


## Result: **the corrected test recovers ZERO tree fits (0/16 at both sample sizes). And the ridge fits that passed Check 2 do NOT pass DS — the guardrail's red flag, investigated below and resolved.**

Script: `task34_3_retest.py` · Data: `task34_3_retest.csv` (n=40),
`task34_3_retest_n400.csv` (n=400)

24 cells: 2 domains × 3 models × 4 mechanisms (β, local kNN, disagreement,
locally-varying NW). All construction copied unchanged from Tasks 32–33 — only
the test differs.

## The full table (n=400, the better-powered run)

| domain | model | class | mechanism | Check 2 pct | C2 pass | DS | DS pct | DS pass |
|---|---|---|---|---|---|---|---|---|
| insurance | xgboost | tree | beta | 0.665 | ✗ | −0.056 | 0.255 | ✗ |
| insurance | xgboost | tree | A_local_knn | 0.620 | ✗ | −0.069 | 0.180 | ✗ |
| insurance | xgboost | tree | B_disagreement | 0.625 | ✗ | −0.051 | 0.310 | ✗ |
| insurance | xgboost | tree | D_local_nw | 0.255 | ✗ | +0.024 | 0.810 | ✗ |
| insurance | lightgbm | tree | beta | 0.840 | ✗ | −0.028 | 0.265 | ✗ |
| insurance | lightgbm | tree | A_local_knn | 0.775 | ✗ | −0.039 | 0.195 | ✗ |
| insurance | lightgbm | tree | B_disagreement | 0.660 | ✗ | +0.010 | 0.550 | ✗ |
| insurance | lightgbm | tree | D_local_nw | 0.625 | ✗ | −0.012 | 0.380 | ✗ |
| insurance | **ridge** | linear | beta | **0.995** | **✓** | +0.035 | 0.515 | ✗ |
| insurance | **ridge** | linear | A_local_knn | **0.975** | **✓** | +0.039 | 0.793 | ✗ |
| insurance | ridge | linear | B_disagreement | 0.365 | ✗ | −0.032 | 0.337 | ✗ |
| insurance | ridge | linear | D_local_nw | 0.930 | ✗ | −0.037 | 0.318 | ✗ |
| energy | xgboost | tree | beta | 0.530 | ✗ | +0.106 | 0.315 | ✗ |
| energy | xgboost | tree | A_local_knn | 0.740 | ✗ | +0.074 | 0.195 | ✗ |
| energy | xgboost | tree | B_disagreement | 0.545 | ✗ | +0.104 | 0.295 | ✗ |
| energy | xgboost | tree | D_local_nw | 0.790 | ✗ | −0.049 | 0.000 | ✗ |
| energy | lightgbm | tree | beta | 0.545 | ✗ | −0.061 | 0.055 | ✗ |
| energy | lightgbm | tree | A_local_knn | 0.805 | ✗ | −0.076 | 0.080 | ✗ |
| energy | lightgbm | tree | B_disagreement | 0.660 | ✗ | −0.062 | 0.085 | ✗ |
| energy | lightgbm | tree | D_local_nw | 0.795 | ✗ | −0.124 | 0.010 | ✗ |
| energy | **ridge** | linear | beta | **0.995** | **✓** | −0.037 | 0.204 | ✗ |
| energy | **ridge** | linear | A_local_knn | **1.000** | **✓** | +0.038 | 0.673 | ✗ |
| energy | ridge | linear | B_disagreement | 0.225 | ✗ | −0.001 | 0.538 | ✗ |
| energy | ridge | linear | D_local_nw | 0.920 | ✗ | +0.173 | **0.994** | **✓** |

**Tree: 0/16 under both tests. Linear: 4 pass Check 2, 1 passes DS.**

## The guardrail's red flag — investigated, not accepted

The guardrail states: *"Ridge fits that previously passed must still pass under
the corrected test, or the correction itself needs revisiting."* Four ridge fits
pass Check 2; none of those four passes DS. That required investigation.

**First hypothesis: DS is underpowered at n=40.** DS is a Spearman correlation
over `n` test points, so its null has SE ≈ 1/√(n−3). At the n=40 inherited from
Check 2:

| n | Spearman SE | null 95th pct ≈ pass bar |
|---|---|---|
| **40** | 0.164 | **0.270** |
| 100 | 0.102 | 0.167 |
| 400 | 0.050 | **0.083** |

At n=40 the bar is **|DS| > 0.270** — larger than the biggest value observed on
*any* real fit (0.253). The test literally could not pass anything. That is a
real design flaw, inherited by copying Check 2's sample size without rechecking
it for a correlation-based statistic.

**So I re-ran the entire table at n=400**, where the bar drops to 0.083.

**The result settles it.** If the n=40 DS values had been real-but-underpowered
signal, they would persist at n=400 and clear the lower bar. Instead:

| | median &#124;DS&#124; | max &#124;DS&#124; | pass bar |
|---|---|---|---|
| n=40 | 0.135 | 0.253 | 0.270 |
| n=400 | **0.044** | 0.173 | 0.083 |

**DS magnitudes shrank by 67%** as sample size grew 10×. That is the signature of
sampling noise averaging out, not of signal emerging. Tree DS at n=400 has median
**−0.044**, range −0.124 to +0.106 — indistinguishable from zero.

**Resolution of the red flag:** the ridge/DS failures are not evidence the
correction is broken. They are evidence that DS, correctly powered, finds **no
directional specificity on these fits either** — including the ridge fits where
Check 2's magnitude criterion passed. Check 2 and DS measure genuinely different
things: a direction can dominate on induced *magnitude* (β on ridge, percentile
0.995) while carrying no cross-point *difficulty information* (DS −0.037).

That is coherent, and it is a finding about what Check 2's pass actually
certified — magnitude alignment, not difficulty relevance.

## What this rules out

Task 33's null-discrimination hypothesis was well-motivated: trees respond to
everything, so a magnitude percentile can't discriminate. Item 2 confirmed the
mechanism exists — DS detects a synthetic signal Check 2 ranks *below every
random direction* (percentile 0.000 on known-signal case A).

But on real fits, with the power problem fixed, **there is no directional
specificity to find**. The 0/20 tree record is not an artifact of the test being
magnitude-based. It survives a test explicitly designed to be magnitude-blind.
