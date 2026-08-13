# Task 18 Item 5 — Recency/Tail Split Ablation

**Verdict: width is NOT constant across splits — the review's framing assumed it
was. And the split barely matters: any tail allocation from 25% upward gives
identical coverage, width and Q_C. Only the 0% (pure recency) point differs.
The binding quantity is whether the tail is represented at all, not how much
budget it receives.**

The spec asked for width to be reported explicitly rather than assumed constant.
That was the right instruction — it is not constant.

Script: `fix-reg/task18_13457_recession_items.py`.
Data: `task18_5_recency_tail_split.csv`.

---

## Result at fixed N = 254

Coverage / width by tail-budget fraction:

| Horizon | 0% tail | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Current | 93.8 / **21.7** | 93.8 / 108.7 | 93.8 / 108.7 | 93.8 / 108.7 | 93.8 / 108.7 |
| 1M | 89.1 / **21.5** | 96.9 / 92.1 | 96.9 / 92.1 | 96.9 / 92.1 | 96.9 / 92.1 |
| 3M | 90.3 / **31.9** | 95.2 / 109.1 | 95.2 / 109.1 | 95.2 / 109.1 | 95.2 / 109.1 |
| 6M | 84.8 / **53.3** | 96.6 / 123.9 | 96.6 / 123.9 | 96.6 / 123.9 | 96.6 / 123.9 |

**Width jumps 2–5× between 0% and 25% tail, then is exactly flat.** So the
comparison the review asked for — selector vs recency "holding width constant" —
cannot be made, because width is a step function of the split, not a constant.

## Why the plateau: the ceiling effect again

Q_C at 6M by split fraction:

| tail_frac | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| Q_C(0.90) | 20.05 | **68.72** | **68.72** | **68.72** | **68.72** |
| pool max in set | 99.92 | 99.92 | 99.92 | 99.92 | 99.92 |

Even a 25% tail allocation (≈64 scores) captures enough of the pool's upper
extreme that the selected set's 90th percentile lands on the same scores as a
100% allocation. Beyond 25%, extra tail budget adds scores *below* the operative
quantile and changes nothing ACI consumes.

This is the third independent appearance of the same phenomenon — Task 10's
ceiling result, Task 17 Item 2's identical-Q_C finding, and now this. The
consistency is itself evidence for the mechanism.

## What the paper should say

1. **Correct the premise.** Width is not constant across splits; it is
   approximately bimodal (low at 0% tail, high at ≥25%). Any "holding width
   constant" comparison must be constructed deliberately, not assumed.
2. **The practical finding is a cheap one:** a practitioner needs only ~25% of
   the calibration budget in the tails to get the full coverage benefit. The
   remaining 75% is free for recency, domain relevance, or any other
   consideration. That is a more useful deployment statement than "use the
   extreme-tail selector".
3. **This subsumes Task 17 Item 2's `tail_plus_recency` result** (50/50 matched
   the selector exactly) and shows 50% was not a special value — anything ≥25%
   works.

## Honest caveats

- **Five split points only** (0, 25, 50, 75, 100%). The transition between 0%
  and 25% is unmapped; the true minimum viable tail fraction is somewhere below
  25% and was not located. A finer sweep between 0 and 25% would give the
  practitioner-facing number precisely.
- **N = 254 throughout.** The plateau depends on 25% of N being enough scores to
  reach the pool's extreme; at much smaller N the minimum fraction would rise.
- **"Recency" here means the trailing months of the pool.** Any other
  non-tail allocation (random, stratified) would likely behave the same, since
  the plateau is driven by what the tail half captures, not by what the other
  half contains — Task 17 Item 2's stratified results support that reading.
- **Recession testbed only**, per the spec.
