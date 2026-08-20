# Task 20 — A Per-Prediction Tail-Reach Confidence Signal

## Verdict

**Neither signal is validated. Signal A is NOT refuted — see the climate
follow-up (`task20_5_climate.md`), which revised this: on recession its lift is
*negative* in 5 of 8 cells, but that testbed never moves the margin out of
4.10-6.89. On climate, where the margin reaches 2.31, the sign flips POSITIVE on
both models (+5.06, +6.01 pp) and survives a confound check — still not
significant after correction (perm_p 0.062, 0.149), but no longer evidence
against the signal. Signal B is directionally consistent and
strengthens with horizon, but nothing survives multiplicity correction (minimum
BH q-value 0.264 across 24 tests; zero cells at q < 0.10), and its one
nominally-significant cell (p=0.036) collapses under both BH and a permutation
null. Item 4 does not run.**

The task asked whether the paper's aggregate, retrospective honesty about 6M
coverage can be made live and per-prediction. On this testbed, with this data:
**not yet demonstrated.** That is a real answer, not a failure to finish — and
it was reached without ever using the outcome being scored, which was the
constraint the whole task existed to respect.

## Item-by-item

| Item | Status | Finding |
|---|---|---|
| 1 — Signal A (structural margin) | **Built, exactly verified** | Supply matches `selector_lib` for all 7 N values; 0/84 inconsistent cells vs. Section 4.4's ceiling sweep; margin hits exactly 1.000 at q=0.5, the proof's own boundary. |
| 2 — Signal B (error drift) | **Built** | K=6 fixed in advance from stated rationale; both spread definitions reported; no K sweep run anywhere. |
| 3 — Falsification | **Run, NULL** | Min BH q=0.264, 0 cells at q<0.10. Signal A negative-lift in 5/8 cells *on recession*. Signal B underpowered, not refuted. |
| 5 — Climate follow-up | **Run, revises Item 3** | Climate's alpha reaches 0.216 (8x recession's excursion), margin down to 2.31 but still short of the q=0.5 boundary. Signal A's sign flips POSITIVE (+5.06, +6.01 pp, n=243); not significant after correction. |
| 4 — Packaging | **Not run** | Gated on Item 3 validating a signal. |

## What was actually learned

**Signal A is a correct instrument aimed at a condition that doesn't occur *in
this project's domains*.** (Revised by the climate follow-up — see below.)
Item 1's verification is airtight — the margin reproduces the proof's boundary
behaviour to the decimal, reaching exactly 1.000 at q=0.5 without any fitting.
But in live operation ACI's alpha stays in a narrow band, so the observed margin
never leaves **4.10-6.89**, always deep in the regime where the guarantee is
exact. The signal can only discriminate near margin→1. This is worth
distinguishing carefully: Signal A is uninformative *as a live diagnostic on the
recession testbed*, not shown to be wrong as mathematics. On a system whose alpha does
wander toward 0.5 it might well be informative — that was a hypothesis, and the
climate follow-up partially tested it: climate pushes the margin to 2.31 (vs.
recession's floor of 4.10) and Signal A's lift flips positive on both models,
consistent with the signal being real but detectable only once alpha has room to
move. Still not significant after correction, so the honest status is **not
validated, not refuted — untested at the condition it was designed for.**

**Signal B is the one worth revisiting, and precisely why it can't be claimed
now.** Every cell where misses exist shows positive lift, growing monotonically
with horizon (+4.94 → +10.71 → +17.25 pp at 1M/3M/6M pooled). That is what a
real effect looks like. It is also what noise looks like at n=53 with ~6 misses
in the flagged quartile. The paper's own 6M sample-size problem (n≈60) binds
here exactly as the guardrail anticipated, and the honest statement is that this
data cannot separate the two.

**A structural obstacle worth naming for any follow-up.** The diversity-optimal
strategy misses 2-3 times per horizon (zero at Current and 1M). Several
validation cells are therefore exactly 0.00 with p=1.0 — there is nothing to
predict. The better-calibrated the strategy, the less signal exists to validate
a diagnostic against. This is the same low-headroom problem Task 19 Item 3
flagged for LLM settings, arriving from a different direction.

## On the discipline

Two things that would have produced a false positive, avoided deliberately:

- **Not tuning K.** K=6 was fixed from the horizon length before any validation.
  Sweeping K after seeing the null would almost certainly have produced a
  "significant" cell somewhere in the 24-test family — the exact failure mode
  Tasks 17/18/19 each guarded against.
- **Not reporting the nominal p.** 6M pooled Signal B at p=0.0358 is publishable-
  looking in isolation. It is one test among 24. Both corrections were computed
  *because* that number was tempting, not after being asked for.

The no-leakage constraint was independently audited (`task20_3_leakcheck.py`):
an oracle positive control registers +44.75 pp, confirming the machinery detects
real signal when present; corrupting all data at steps ≥ t leaves Signal B
bit-identical; Q_G appears nowhere in executable code. One caveat on that audit:
the positive control initially read +0.00 — a flaw in the control itself (a
binary oracle makes `quantile(0.75)=0`, flagging every month), not in the
pipeline. Worth recording because a broken control would have made the null
uninterpretable, and it nearly passed unnoticed as confirmation.

## If this is picked up again

The productive path is Signal B with more test months, not a new signal design
and not a K sweep. Two concrete options: pool the four horizons into a single
test with a horizon fixed effect (roughly 4× the misses, at the cost of assuming
a shared effect), or extend the test window using `task9a_extend_window.py`'s
extended recession data, which already exists. Either would give the power this
testbed lacks. Signal A should not be carried forward for the *recession* testbed, but the
climate result argues against writing it off: settling it needs a testbed whose
ACI genuinely operates at low q (margin near 1), which no domain here provides.

## Files

- `task20_1_signalA.py` / `.md` — exact margin + both hand-verifications
- `task20_1_signalA_verify.csv` — 84-cell check against Section 4.4
- `task20_2_signalB.py` / `.md` — drift signal, K rationale
- `task20_3_validation.py` / `.md` — falsification, both signals separately
- `task20_3_validation.csv`, `task20_3_permutation.csv`, `task20_3_traces.csv`
- `task20_3_leakcheck.py` — independent no-leakage audit
- `task20_5_climate.py` / `.md` — climate follow-up, revises Item 3's Signal A verdict
- `task20_5_climate.csv`, `task20_5_climate_alpha.csv`, `task20_5_climate_permutation.csv`, `task20_5_climate_traces.csv`
- `task20_4_packaging.md` — gated off
