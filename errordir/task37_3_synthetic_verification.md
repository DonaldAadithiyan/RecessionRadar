# Task 37, Item 3 — Synthetic Verification of HOS

## Result: **both cases correct. HOS detects known signal (percentile 1.000) and rejects known null (0.825).**

Data: `task37_3_synthetic.csv`

| case | requirement | HOS | null mean | percentile | verdict | correct |
|---|---|---|---|---|---|---|
| known signal | DETECT | **0.5938** | 0.1577 | **1.000** | DETECT | **Yes** |
| known null | REJECT | 0.0432 | 0.0251 | 0.825 | REJECT | **Yes** |

**Known-signal construction:** linear data where the error scale is genuinely
driven by a known direction (`sd = 0.3 + 2·max(X·w, 0)`), with the conditional
mean driven by *different* features. HOS returns 0.594 against a null mean of
0.158 — a 3.8× separation.

**Known-null construction:** homoscedastic error, independent of every feature.
HOS returns 0.043 against a null mean of 0.025, percentile 0.825 — elevated but
comfortably below the 0.95 bar.

Both cases were required before use on real fits, per the guardrail. A test
verified only on "can it find signal" would hide the worse failure — passing
everything.

## A note on the known-null margin

The null case's percentile of 0.825 is not near-zero. That is expected and
benign: β fit to a random target on finite data will, by chance, have *some*
positive held-out correlation, and permutation directions have the same property.
The percentile compares one draw against 200, and 0.825 means β's draw happened
to land in the upper portion of that distribution without reaching the tail.

Worth recording as a sensitivity note: at a 0.95 bar, a known-null case landing
at 0.825 leaves less headroom than one landing at 0.5. Had it landed at 0.94 the
instrument would still have "passed" verification while being uncomfortably close
to a false positive. The real-fit results (1.000 and 0.990, HOS 2.7–3.0× the null
mean) sit far enough above this that the margin is not the binding concern there.
