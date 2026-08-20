# Task 20, Item 2 — Signal B: Recent Realized-Error Drift

**Status: built. (Not validated as a miss predictor — see Item 3.)**

Script: `task20_2_signalB.py`

## Definition

    drift_ratio_t = spread(last K resolved scores, strictly before t) / Q_C(q_t)

| Value | Reading |
|---|---|
| `<< 1` | recent errors small relative to what the calibration set is provisioned for |
| `-> 1` | recent errors brushing the set's own (1-alpha)-quantile — set may be falling behind |
| `> 1`  | recent reality has already exceeded it |

Both spread definitions named in the spec are computed and reported:
the window's own **max** and its own **p90**. Neither was selected by
looking at Item 3's outcome — both appear in every results table.

## K = 6, fixed in advance

**Rationale, stated before any validation run:** the primary testbed is monthly
macroeconomic data and the headline horizon is 6M, so K=6 is exactly one
forecast horizon of resolved history — the shortest window covering a full
horizon's realized error without reaching back into a prior regime. This is also
the value the spec itself suggests.

**No K-sensitivity sweep was run anywhere in this task.** This is the item the
spec flagged as most at risk of becoming a parameter search, and the guardrail
was held: one value, fixed from rationale, never tuned against Item 3. Sweeping
K now — after seeing Item 3's null — would be exactly the failure mode Tasks
17/18/19 each guarded against. If a sweep is ever wanted it must be a separate,
clearly-labelled robustness study reported alongside this null, not a
replacement for it.

## No-leakage design

At step `t` the window uses resolved scores from steps **strictly < t**. The
score at `t` is excluded — at prediction time it has not happened yet. The
signal returns `None` during warm-up (first K steps) rather than guessing, which
is why Signal B's tables show `n=53-59` against Signal A's `n=59-65`.

Enforced by construction in `recent_window()` and independently audited in
`task20_3_leakcheck.py` (check 2): corrupting all scores at steps `>= t` leaves
the signal bit-identical (`1.726358` both ways), proving no forward reach.
