# Task 31, Items 2–3 — Comparison on Epidemic, and the Pre-Registered Test

## Outcome: **2 — NOT CONFIRMED.** Frozen-estimate methods do rank near the bottom (8th and 10th of 11), but the class-level separation is not significant (Mann-Whitney p=0.109), one *online* method ranks below both of them, and errordir itself over-covers at 100% and places 5th, losing to four simpler online methods.

Script: `task31_23.py` · Data: `task31_2_baselines.csv`,
`task31_3_comparison.csv`, `task31_3_significance.csv`

## Full table (epidemic, n=400 test country-days, spread 1.091)

| rank | method | anchor | coverage | width | ratio | Winkler mean | Winkler median |
|---|---|---|---|---|---|---|---|
| 1 | pooled_trailing | online | 91.50 | 1.205 | 1.11 | **1.434** | 1.174 |
| 2 | pid_conformal | online | 90.25 | 1.159 | 1.06 | 1.438 | 1.104 |
| 3 | bellman_ci | online | 89.25 | 1.082 | 0.99 | 1.499 | 1.123 |
| 4 | dtaci | online | 92.75 | 1.452 | 1.33 | 1.594 | 1.162 |
| 5 | **errordir** | online | **100.00** | 2.016 | 1.85 | 2.016 | 1.938 |
| 6 | acmcp | online | 90.00 | 1.865 | 1.71 | 2.237 | 1.654 |
| 7 | evt_tail | online | 99.50 | 2.290 | 2.10 | 2.292 | 2.168 |
| 8 | **cqr** | **frozen** | 92.25 | 2.174 | 1.99 | **2.622** | 2.118 |
| 9 | mondrian | online | 99.25 | 2.637 | 2.42 | 2.644 | 1.712 |
| 10 | **rlcp** | **frozen** | 99.75 | 2.643 | 2.42 | **2.644** | 2.292 |
| 11 | diversity_optimal | online | 100.00 | 6.486 | 5.95 | 6.486 | 6.161 |

No method produced unbounded intervals; none is vacuous.

## The pre-registered test, evaluated

**Prediction:** online-anchored methods outperform frozen-estimate methods on
this concept-drift domain, mirroring energy.

| class | n | Winkler values | median | best |
|---|---|---|---|---|
| online | 9 | 1.434, 1.438, 1.499, 1.594, 2.016, 2.237, 2.292, 2.644, 6.486 | 2.016 | 1.434 |
| frozen | 2 | 2.622, 2.644 | 2.633 | 2.622 |

**Mann-Whitney (online < frozen): U=3.0, p=0.109 — not significant.**

The directional pattern is there: both frozen methods land in the bottom four,
and the frozen median (2.633) is worse than the online median (2.016). But with
only **two** frozen-estimate methods the test has almost no power, and the
separation fails at conventional significance.

Two facts specifically contradict a clean confirmation:

1. **An online method ranks below both frozen ones.** diversity_optimal (6.486)
   is by far the worst method in the table — 2.5× worse than either frozen
   method. If the online/frozen distinction were the operative variable, that
   should not happen.
2. **errordir over-covers badly.** 100.00% coverage at width 2.016 — it never
   misses once in 400 test days, which means its intervals are far wider than
   needed. It ranks 5th, losing to pooled_trailing, pid_conformal, bellman_ci
   and dtaci, all simpler online methods.

## Why errordir over-covers here — the direction asymmetry matters

Item 1 flagged this in advance: epidemic's drift runs **opposite** to energy's.
Errors **shrink** in the test period (ratio 0.443) rather than grow.

errordir's ACI anchor adapts by widening after misses. With errors shrinking,
there are almost no misses to trigger adaptation downward, and the multiplier's
floor of 0.75 bounds how far it can narrow. So the method inherits calibration-
period widths that are simply too large for an easier test period — the mirror
image of energy, where growing errors rewarded its widening behaviour.

This is exactly the vulnerability Item 1 predicted the direction-flip would
expose, and it is why an opposite-direction domain was the stricter test.

## errordir vs each baseline (BH across all cells)

| baseline | anchor | mean diff | win rate | BH q |
|---|---|---|---|---|
| diversity_optimal | online | −4.470 | 100.0% | 0.0000 |
| rlcp | **frozen** | −0.629 | 76.5% | 0.0000 |
| mondrian | online | −0.628 | 16.2% | 0.0000 |
| cqr | **frozen** | −0.607 | 60.2% | 0.0000 |
| evt_tail | online | −0.276 | 85.0% | 0.0000 |
| acmcp | online | −0.221 | 40.8% | 0.2517 |
| dtaci | online | **+0.421** | 12.0% | 1.0000 |
| bellman_ci | online | **+0.517** | 9.8% | 1.0000 |
| pid_conformal | online | **+0.578** | 6.5% | 1.0000 |
| pooled_trailing | online | **+0.581** | 4.8% | 1.0000 |

errordir does beat both frozen methods significantly (q=0.0000, win rates 76.5%
and 60.2%). But it **loses significantly to four online baselines**, with win
rates as low as 4.8% against pooled_trailing.

## Honest reading

The frozen-estimate methods did underperform, and errordir beat both. That is
consistent with the mechanism. But the evidence does not support calling this a
confirmation:

- the class-level test is underpowered (n=2) and non-significant;
- an online method is the single worst performer in the table;
- errordir's own result here is poor in absolute terms (5th of 11, 100%
  over-coverage), so its wins over RLCP and CQR come from a method that is
  itself badly calibrated for this domain.

The most defensible statement is that **the online/frozen distinction did not
cleanly separate performance on this domain**, and that errordir's advantage
under drift is direction-dependent — it helps when errors grow and hurts when
they shrink.
