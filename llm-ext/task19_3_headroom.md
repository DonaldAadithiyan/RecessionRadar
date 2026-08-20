# Task 19, Item 3 — Low-Headroom Check: NOT RUN

**Status: not applicable — Item 3 is defined on the data Item 2 uses, and
Item 2 did not run.**

Item 3's instruction is to "check where **the eval set's** baseline coverage
already sits" and to "report the headroom honestly **for whatever data Item 2
used**." No eval set was identified (see `task19_2_real_signal.md`), so there is
no baseline coverage to measure. Computing headroom for a tabular domain instead
would answer a question this item did not ask.

## The one thing worth recording in advance

Item 3 exists because headroom determines how much effect to expect. The paper's
own spread is the reference: recession 6M baseline sat at 67.8% in-sample /
84.8% out-of-fold — wide headroom, clean effect — while healthcare sat at 88-91%
with "about ten points of headroom, most spent on over-coverage," and showed the
weakest, least clean effect of any domain tested.

This is a live concern for LLM settings specifically, and should be checked
first whenever Item 2 becomes runnable. Well-calibrated LLM confidence signals on
a standard benchmark often already sit near or above 90% coverage at
alpha=0.10 — that is the *healthcare* regime, not the recession regime. If so,
Item 3's own guardrail applies directly: report it as a reason to expect a
smaller, harder-to-detect effect, not as a pipeline failure, and do not go
hunting for a lower-coverage eval set to make the result look better.
