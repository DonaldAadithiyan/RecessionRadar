# Task 19, Item 4 — Score-Design Sensitivity: NOT RUN

**Status: gated off, as specified.**

Item 4 runs "only if Items 1-3 all show a real, usable effect." Items 2 and 3 did
not run for lack of any LLM eval data (see `task19_2_real_signal.md`), so no real
effect has been demonstrated and there is nothing to robustness-check.

Item 4's own guardrail is explicit that this is the correct outcome: "This item
only runs if the earlier items already show a real effect. Don't spend time here
first — score-design sensitivity is a robustness check on a positive result, not
a way to go looking for one."

Running two score designs against synthetic data alone would test only whether a
generator built with a chosen severity coupling still shows that coupling under a
second scalarization. That is a property of the generator, not evidence about
LLM scores, and it would create a positive-looking artefact with no empirical
content.
