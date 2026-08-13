# Task 18 Item 1 — Regime-Weighted Conformal Baseline

**Verdict: regime-weighted CP matches or slightly exceeds the selector's
coverage, and is the first standard-toolbox method in this paper to do so. But
it reaches that coverage with wider intervals, and it works for the same reason
the selector does — it raises the operative quantile. This is a meaningful
addition to the baseline table and it does not threaten the paper's mechanism.**

Script: `fix-reg/task18_13457_recession_items.py`.
Data: `task18_1_weighted_cp.csv`.

---

## Method, and how λ was fixed

Every pooled score is weighted by `w_i ∝ exp(−λ·d_i)`, where `d_i` is months
since the most recent regime transition, and ACI's calibration quantile is the
**weighted** empirical quantile of the full pool.

**λ was fixed before touching test data**, from the training data's own regime
structure: half-life = the mean duration of a rare-event episode in the training
period. That gives a half-life of ~6 months and **λ = 0.1155**. No test-set
tuning, per the guardrail.

## Result

| Horizon | Trailing | **Regime-weighted** | Selector | Weighted width | Fraction of the selector's gain |
|---|---|---|---|---|---|
| Current | 93.85 | 93.85 | 93.85 | 165.76 | — (no gap) |
| 1M | 89.06 | **98.44** | 96.88 | 173.28 | **1.20×** |
| 3M | 90.32 | **95.16** | 95.16 | 141.48 | **1.00×** |
| 6M | 84.75 | **96.61** | 96.61 | 141.56 | **1.00×** |

**It closes the gap completely at 3M and 6M, and overshoots at 1M.** This is the
first method from the standard conformal toolbox — across Mondrian, PID,
EVT-tail, DtACI, AcMCP, Bellman CI and CPTC — to match the selector on coverage.

## Why this is a strengthening result, not a threat

**It succeeds through the paper's own mechanism.** Exponential regime weighting
up-weights scores from recent and matching-regime months, which in this data are
disproportionately the rare-event months carrying the large scores. That raises
the weighted (1−α) quantile — exactly the quantity §4 identifies as operative.
It is a different route to the same tail-reach, which is precisely what Task 17
Item 2 showed with `tail_plus_recency`.

**It costs more width.** At 6M the weighted method uses 141.56 against the
selector's 123.86 (Task 7) — about 14% wider for the same 96.61% coverage. At 1M
it overshoots to 98.44%, i.e. overcoverage, at 173.28 width.

So the honest framing for the baseline table is: *regime-weighted CP is
competitive on coverage and the strongest standard-toolbox comparator, but the
selector reaches the same coverage more efficiently.*

## What the paper should do

1. **Add regime-weighted CP to the main baseline table.** It is the strongest
   comparator found and its absence would be a real gap — the reviewer was right
   to ask.
2. **Report it as matching, not losing.** Anything else would misrepresent
   3M and 6M.
3. **Use it as supporting evidence for the mechanism**, since it works by
   raising the operative quantile rather than by some unrelated route.

## Honest caveats

- **One λ, chosen structurally.** A different half-life would move these numbers;
  λ = 0.1155 was set from the training regime structure and not swept. A
  sensitivity sweep would strengthen the comparison but risks becoming the
  test-set tuning the guardrail forbids.
- **The weighting uses the regime label**, which at test time requires knowing
  whether the current month is a rare-event month. In deployment that label is
  not available contemporaneously — the paper's selector needs no test-time
  labels at all. This is a real practical asymmetry and should be stated when
  the baseline is reported.
- **Width is not directly comparable at 1M**, where the weighted method
  overshoots to 98.44%; part of its extra width buys coverage nobody asked for.
- **Recession testbed only**, per the spec. Not extended to healthcare or
  climate.
