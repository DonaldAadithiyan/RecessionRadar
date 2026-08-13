# Task 18 Item 3 — Winkler Interval Score: Selector vs DtACI

**Verdict: DtACI beats the selector on the Winkler score at all four horizons,
and so does the plain trailing baseline. On a proper scoring rule that prices
width and miscoverage together, the selector is the WORST of the three. This is
an unfavourable result and it is reported as the pre-specified rule produced
it.**

Script: `fix-reg/task18_13457_recession_items.py`.
Data: `task18_3_utility_scores.csv`.

---

## The scoring rule, pre-specified

Per the guardrail, the rule was fixed **before computing anything**. For a
central (1−α) interval [l, u] and outcome y:

```
W = (u − l)                        if l ≤ y ≤ u
  = (u − l) + (2/α)(l − y)         if y < l
  = (u − l) + (2/α)(y − u)         if y > u
```

Lower is better. Chosen because it is the standard proper scoring rule for
interval forecasts and prices width and miscoverage on a single scale — not
selected after seeing which method it favours.

## Result (mean Winkler score, lower is better)

| Horizon | Diversity-optimal | Pooled/trailing | **DtACI** | Best |
|---|---|---|---|---|
| Current | 147.20 | 118.72 | **111.05** | DtACI |
| 1M | 108.55 | 95.52 | **76.84** | DtACI |
| 3M | 139.40 | 118.64 | **109.74** | DtACI |
| 6M | 148.62 | 122.77 | **116.71** | DtACI |

**4 of 4 to DtACI.** And the selector loses to the plain trailing baseline as
well, at every horizon.

## What this means, stated plainly

The selector maximises coverage and pays for it in width. The Winkler score
charges for that width directly, and at α = 0.10 the miscoverage penalty
(2/α = 20× the shortfall) is not large enough to offset intervals that are
2–5× wider than the baseline's.

**This is consistent with everything already established, not a new
contradiction.** Task 9 found tuned BCI and CPTC beat the selector on sharpness;
Task 11 introduced pool augmentation specifically because the selector's width
cost is intrinsic; Task 16D showed the mechanism weakens as α rises. Item 3
quantifies the same tradeoff on a single number.

**What it does change** is that the paper cannot claim the selector is the best
choice by a utility criterion. It is the best choice *if the objective is
coverage*, which is the objective conformal prediction is normally deployed
against — but a reader who prices width will prefer DtACI.

## Recommended framing

> Under a Winkler interval score, which prices width and miscoverage jointly,
> DtACI outperforms the diversity-optimal selector at all four horizons
> (116.71 vs 148.62 at six months). The selector is designed to maximise
> coverage under rare-event scarcity, and does so; it is not designed to
> optimise a width-penalised utility, and does not. Practitioners whose loss
> function prices interval width should prefer DtACI or the pool-augmentation
> variant of §X, which attains comparable coverage at substantially lower width.

That is more useful to a reader than omitting the comparison, and the reviewer
asked for it directly.

## Honest caveats

- **The result depends on α through the 2/α penalty.** At α = 0.10 the
  miscoverage charge is 20× the shortfall; at α = 0.01 it would be 200×, which
  would favour wide intervals far more. The ranking here is specific to the
  paper's operating point.
- **Winkler is one proper rule among several.** Pinball loss integrated across
  quantiles, or a coverage-constrained width criterion, would weight the
  tradeoff differently. One rule was pre-specified to avoid selecting after the
  fact; that discipline also means alternatives were not explored.
- **DtACI's intervals were reconstructed from its returned widths**, centred on
  the point prediction. That is how the method builds them, but it is a
  reconstruction rather than bounds the runner returned directly.
- **Recession testbed, out-of-fold**, per the spec. Not extended to healthcare or
  climate.
