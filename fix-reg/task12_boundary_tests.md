# Task 12 — Two Boundary Tests on the Selector's Optimality Claim

**Both close cleanly. The optimality claim can drop its hedge, and α-drift
re-selection is now falsified rather than merely doubted.**

Script: `fix-reg/task12_small_n_and_drift.py`.
Data: `task12a_small_n_ceiling.csv`, `task12b_alpha_drift.csv`.

---

## Study A — does optimality survive small N?

Task 10 proved the support-width selector attains exactly the maximum achievable
quantile reach, then hedged: *"the negative result is specific to this N/pool
regime… at much smaller N, selection could matter again. Not tested."* That
hedge sat directly beneath an otherwise unqualified optimality claim, which made
it the most exposed statement in the theory section.

**Result: 56 of 56 configurations at ratio exactly 1.000000.** N swept from 254
down to 20, across all 8 cells, at all three operative quantiles.

| N | 20 | 30 | 50 | 80 | 120 | 180 | 254 |
|---|---|---|---|---|---|---|---|
| min ratio (all 8 cells) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

### Why it holds — an analytic argument, not just 56 data points

The alternating-tail rule always supplies **~N/2** extreme-tail scores. But the
(1−α) quantile of an N-set depends only on its top **(1−q)·N** scores, and the
measured ACI window is q ∈ [0.868, 0.927] — the top 7–13%.

| N | Scores the operative quantile depends on | Scores the selector supplies |
|---|---|---|
| 20 | ~3 | 10 |
| 50 | ~7 | 25 |
| 254 | ~34 | 127 |

The selector is **oversupplied by roughly 4× at every N**. Optimality could only
break if N/2 < (1−q)·N — that is, if **q < 0.5**, an operating quantile far
outside anything ACI visits.

This converts the empirical finding into a structural one: *the selector attains
the ceiling for any N, at any operating quantile above the median.* That is a
materially stronger statement than "we checked 254 and it held", and it is
falsifiable — a calibration regime driving α above 0.5 would break it.

### Practical note: coverage rises as N shrinks

| Cell | N=20 | N=80 | N=254 |
|---|---|---|---|
| Climate ridge | 100.00 | 100.00 | 99.59 |
| Healthcare ridge | 99.32 | 98.63 | 92.47 |
| Recession 6M | 100.00 | 96.61 | 96.61 |
| Recession Current | 98.46 | 96.92 | 93.85 |

Smaller calibration sets give *higher* coverage here, because the selector packs
a small set almost entirely with extremes. That is not a recommendation — those
intervals are correspondingly wide, and small-N conformal quantiles are unstable
— but it is worth one sentence, since a reader may assume bigger is always
better.

## Study B — does re-selecting as α drifts help?

Task 10 called this "much less promising" given the narrow α window, but left it
untested. Untested is not falsified, so it was run: an ACI variant that
re-selects its calibration subset at every step to target the *current* α_t,
against the static selector on identical streams.

**Result: 0 of 8. Every cell identical to the static selector — coverage
unchanged, width ratio exactly 1.000.**

| Domain | Model | Horizon | Static | Re-select | Δ |
|---|---|---|---|---|---|
| Recession | stacking-chain | Current | 93.85 | 93.85 | 0.00 |
| Recession | stacking-chain | 1M | 96.88 | 96.88 | 0.00 |
| Recession | stacking-chain | 3M | 95.16 | 95.16 | 0.00 |
| Recession | stacking-chain | 6M | 96.61 | 96.61 | 0.00 |
| Healthcare | ridge | 30-day | 92.47 | 92.47 | 0.00 |
| Healthcare | gradboost | 30-day | 93.84 | 93.84 | 0.00 |
| Climate | ridge | region-month | 99.59 | 99.59 | 0.00 |
| Climate | gradboost | region-month | 98.77 | 98.77 | 0.00 |

### The identical results are diagnostic, and were checked

Byte-identical output usually means the treatment never applied. It did here:
re-selection genuinely picks **different sets** as α moves — only ~50% overlap
with the static selection (126–130 of 254 indices at α ∈ [0.076, 0.100]).

But those different sets produce **identical operative quantiles**:

| α | Re-selected set's q(1−α) | Static set's q(1−α) |
|---|---|---|
| 0.076 | 75.730 | 75.730 |
| 0.085 | 71.346 | 71.346 |
| 0.095 | 69.890 | 69.890 |
| 0.100 | 68.719 | 68.719 |

This is the Task 10 ceiling result reappearing in a new guise. The static
selector is already at the maximum reach at *every* α in the visited window, so
re-targeting within that window cannot find anything better. **Re-selection is
not merely unhelpful — it is provably incapable of helping given Study A.**

## What both results change

1. **The optimality claim loses its hedge.** Replace "specific to this N/pool
   regime, not tested at smaller N" with the tested-and-explained version: it
   holds for all N ≥ 20 across 8 cells, and structurally for any operating
   quantile above the median.
2. **α-drift re-selection moves from "probably not worth it" to "tested and
   ruled out"**, with the mechanism explained. One less open thread, and it
   closes off another class of future work.
3. **Both reinforce the same underlying point:** for a fixed pool, selection is
   exhausted. Everything that remains must change what is *in* the pool — which
   is exactly what Task 11's augmentation does.

## Honest caveats

- **Study A tests the ceiling, not coverage optimality.** The selector attains
  the maximum achievable *quantile reach*; that is the quantity Phase 5 proved
  governs the coverage deficit, but coverage also depends on the test-time tail,
  which no calibration choice controls.
- **The N sweep bottoms out at 20.** Below that, empirical quantiles become so
  coarse that the ceiling comparison stops being meaningful rather than starting
  to fail. Not a tested boundary, a measurement floor.
- **The q < 0.5 boundary is derived, not observed.** No configuration in this
  paper drives α anywhere near 0.5, so the predicted failure regime is
  untested — it is a falsifiable prediction, not a demonstrated limit.
- **Study B tests one re-selection rule** (quantile-targeted within ±20% of the
  current α). A fundamentally different adaptive scheme is not ruled out, though
  Study A makes it hard to see where the headroom would come from.
