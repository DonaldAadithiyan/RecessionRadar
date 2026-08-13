# Task 18 Item 2 — Pooling Clarification

**Answer: no pooling occurs. Selection is performed per-horizon and per-model
throughout, on that slice's scores only. Both comparisons the spec conditioned
on pooling — the coverage comparison and the α_t-stability comparison — are
therefore not applicable and were not run. The reviewer's concern about pooled
selection interacting with ACI's per-horizon updates does not arise.**

Answered from the code, as the spec required, then confirmed empirically.

---

## The code path

**`fix-reg/task7_baseline_horse_race.py`** (every baseline comparison in the
paper) loops over horizons and slices the score matrix *before* selecting:

```python
for h_idx, h in enumerate(LABELS):
    s = np.abs(preds_pool[:, h_idx] - y_train[:, h_idx])   # one horizon only
    rows += run_all_strategies(..., h, scoring, s, ...)
```

and inside, the selector receives that single-horizon array:

```python
sel = greedy_extreme(s)          # s is already horizon-sliced
```

**`fix-reg/task_phase2.py`** (the original selector result) is the same:

```python
def scores_all(h):
    y = y_train_full[:, h]; p = preds_train_full[:, h]
    return np.abs(p - y)         # column h only

def greedy_select(h, N=N_FIX):   # takes a horizon index
    s_all = scores_all(h)
```

Models are likewise handled one at a time — `run_all_strategies` is called once
per `(domain, model, horizon)` triple, and `domain_healthcare.py` /
`domain_climate.py` loop `for name in ["ridge", "gradboost"]` with
`SCORES[name]` selected independently.

## Empirical confirmation

If selection were pooled across horizons, all four horizons would receive the
*identical* calibration set. They do not — pairwise index overlap:

| Pair | Overlap |
|---|---|
| Current vs 1M | 176/254 (69.3%) |
| Current vs 3M | 147/254 (57.9%) |
| Current vs 6M | 132/254 (52.0%) |
| 1M vs 3M | 145/254 (57.1%) |
| 1M vs 6M | 136/254 (53.5%) |
| 3M vs 6M | 157/254 (61.8%) |

Overlap of 52–69% is what independent selection on correlated score columns
produces. Pooling would give 100%.

## What this means for the review's question

The reviewer asked whether pooled selection "interacts with ACI's per-horizon
updates" — specifically whether a shared calibration set, changing at every
horizon simultaneously, would destabilise the online α_t trajectory.

**The premise does not hold.** Each horizon runs an independent ACI instance
against its own independently-selected calibration set, so there is no shared
state through which one horizon's updates could perturb another's. The α_t
trajectories are independent by construction, not by empirical accident.

This should be stated in one sentence in the paper's method section, because the
prose does not currently make it explicit and a careful reader can reasonably
wonder — as this one did.

## Why the α_t-stability comparison was not run anyway

The spec conditioned it on pooling being present ("If no pooling is happening,
say so plainly and skip both comparisons"). Running a pooled-vs-horizon-specific
comparison would require *building* a pooled variant that the paper does not use,
then comparing it to the real one — which answers a question about a method
nobody proposed rather than a property of the paper's method.

If the reviewer's underlying interest is whether α_t is well-behaved at all
(independent of pooling), that is already measured: Task 10 recorded the α
trajectories across every domain, model and horizon and found ACI stays within
**[0.073, 0.132]** throughout, i.e. a narrow band around nominal with no
divergence or oscillation. That result can be cited directly in the response.

## Honest caveats

- **One thing IS shared across horizons: the underlying model.** The
  RegressorChain predicts all four horizons jointly, so the *scores* are
  correlated across horizons even though the *selection* is independent. That is
  a property of the forecaster, not of the calibration procedure, and it is why
  the overlap is 52–69% rather than near-chance.
- **This documents the current code.** Any earlier phase using a different
  selection call would need separate checking; Phase 2 and Task 7 are the two
  that produce every selector number in the paper, and both are per-horizon.
