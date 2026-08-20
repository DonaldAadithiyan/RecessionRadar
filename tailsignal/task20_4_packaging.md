# Task 20, Item 4 — Packaging: NOT RUN

**Status: gated off, as specified.**

Item 4 runs "only if Item 3 shows at least one signal is real." Item 3 returned
a null: nothing survives multiplicity correction (min BH q = 0.264, zero cells
at q < 0.10), Signal A is negative-lift in 5 of 8 cells, and Signal B's single
nominally-significant cell (p=0.0358) does not survive either BH (q=0.264) or a
permutation null (p=0.108).

Item 4's own instruction is explicit: "Do not attempt this before Item 3 has an
answer — designing the interface for a signal that turns out not to be
predictive would be wasted effort."

Designing an output format now would also create a subtler problem: a shipped
warning level implies a validated threshold. There is no validated threshold
here. A "margin < 2 = warning" band would look principled while resting on a
signal whose measured lift is *negative* on this testbed.
