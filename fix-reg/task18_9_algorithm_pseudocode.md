# Task 18 Item 9 — Algorithm Box: The Alternating-Tail Selector

**Transcribed from the implementation, not the prose.** Two edge cases the
paper's prose description does not cover were resolved by reading and probing
the code: **odd N gives the extra point to the LOW tail**, and **ties are broken
by NumPy's default (non-stable) argsort**, which is deterministic per
NumPy version but not guaranteed portable. Neither affects any published
number (verified below), but both belong in the box.

Source: `fix-reg/selector_lib.py::support_width_selector`, which is byte-for-byte
the rule used in `task_phase2.py` and `task7_baseline_horse_race.py`.

---

## Algorithm box (paper-ready)

```
Algorithm 1: Alternating-tail calibration selection

Input:  S = (s_1, ..., s_M)   nonconformity scores over the calibration pool
        N                      target calibration-set size
Output: C ⊆ {1, ..., M}        indices of the selected calibration set, |C| = min(N, |V|)

 1  V ← { i : s_i is finite }                    ▷ drop unscored pool members
 2  π ← argsort( (s_i)_{i∈V} )                   ▷ ascending; ties by pool index
 3  C ← ∅ ;  lo ← 1 ;  hi ← |V| ;  takeLow ← true
 4  while |C| < min(N, |V|) and lo ≤ hi do
 5      if takeLow then
 6          C ← C ∪ { π[lo] } ;  lo ← lo + 1     ▷ smallest remaining score
 7      else
 8          C ← C ∪ { π[hi] } ;  hi ← hi − 1     ▷ largest remaining score
 9      takeLow ← ¬takeLow
10  return sort(C)
```

**Selection order is low-first and strictly alternating.** For a sorted pool
(10, 20, 30, 40, 50, 60, 70):

| N | Selection sequence | Returned set |
|---|---|---|
| 2 | low:10 → high:70 | {10, 70} |
| 3 | low:10 → high:70 → **low:20** | {10, 20, 70} |
| 5 | low:10 → high:70 → low:20 → high:60 → **low:30** | {10, 20, 30, 60, 70} |

## The two edge cases, stated explicitly

**Odd N → the extra point goes to the LOW tail.** Because `takeLow` initialises
to `true` and alternates, an odd budget always ends on a low-tail pick. At
N = 3 the set is {10, 20, 70}, not {10, 60, 70}. This is a consequence of the
initialisation, not a deliberate design choice, and it is worth stating because
a reimplementation starting `takeLow ← false` would produce a different
(mirror-image) set at every odd N.

**Ties are broken by pool index, via NumPy's default argsort.** The default is
introsort/quicksort, which is *not* guaranteed stable. On a small tied example
default and stable agree; on 5000 tie-heavy points **they diverge**. So the
selection is deterministic for a given NumPy version but not guaranteed portable
across implementations.

**Neither edge case affects any published result.** All four recession score
pools have **zero duplicate values** (545 finite scores each, 545 distinct), so
the tie path never executes:

| Horizon | n scored | duplicate values | default == stable |
|---|---|---|---|
| Current | 545 | 0 | yes |
| 1M | 545 | 0 | yes |
| 3M | 545 | 0 | yes |
| 6M | 545 | 0 | yes |

And N = 254 is even throughout the paper, so the odd-N branch never fires
either. Both are specified for reimplementation correctness, not because they
change anything reported.

## Termination and size guarantees

- The loop terminates when either the budget is met or the two pointers cross,
  so `|C| = min(N, |V|)` exactly — the selector degrades gracefully when the
  pool is smaller than the requested budget rather than erroring.
- `lo ≤ hi` prevents the same index being taken twice when N ≥ |V|.
- Output is returned in ascending index order (`sort(C)`), which matters only
  for downstream code that assumes temporal ordering; the ACI runner consumes
  the scores as a set, so ordering is immaterial to coverage.

## A recommended addition to the box's caption

The optimality property proved in §4.3 is a property of *which scores end up in
the set*, not of the order they are picked in. Any rule returning the N/2
largest and N/2 smallest scores attains the same (1−α) quantile for
α < 0.5 — which is exactly what Task 17 Item 2 confirmed empirically, where a
different rule sharing only 65% of indices produced an identical Q_C. The box
should therefore be presented as *one* implementation of extreme-tail selection,
not as the unique procedure the theory requires.

## Honest caveats

- **This documents the current implementation.** If the paper's earlier phases
  used a different variant (e.g. the seeded `greedy_select` in
  `task_oof_and_probit.py`, which was removed as dead code), results from those
  phases would need re-checking against this box. Phase 2 and Task 7 both call
  this exact function.
- **The `min(N, |V|)` degradation is silent.** A caller requesting N = 254 from
  a 200-score pool receives 200 scores with no warning. Not a bug in any
  reported run — every pool exceeds N — but a reimplementer should know.
