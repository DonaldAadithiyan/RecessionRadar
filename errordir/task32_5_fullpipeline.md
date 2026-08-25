# Task 32, Item 5 — Full Pipeline: NOT RUN

**No candidate passed Item 4 on any tree-model domain, so there is nothing to run the pipeline on.**

Item 5's condition is "for any candidate that passes Item 4 on at least one tree-model domain." The tree record is **0 of 8** across both candidates:

| candidate | tree fits validated | ridge control validated |
|---|---|---|
| A_local_knn | **0 / 4** | 2 / 2 |
| B_disagreement | **0 / 4** | 0 / 2 |

Candidate A validates on ridge, but ridge is the control arm, not a tree-model domain — and errordir's ridge result is already established across Tasks 24–31. Running the full pipeline on Candidate A/ridge would re-test a settled configuration with a different difficulty estimator, which is not what Item 5 exists to measure and would not speak to model-agnosticism at all.

Per the guardrail that a null result at the gate is a complete answer, Item 5 is reported as not run rather than repurposed.
