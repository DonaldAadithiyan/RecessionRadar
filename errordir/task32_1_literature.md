# Task 32, Item 1 — Literature Check Before Building

## Result: **both candidates are established prior work.** Candidate B especially — "normalize the nonconformity score by ensemble prediction variance" is the *standard* difficulty estimator for tree ensembles, named explicitly as such in the literature. This reframes Items 2–5 from "build a new mechanism" to "adapt and verify an existing one."

Reading and search only, per the guardrail. No implementation in this item.

## Candidate A — local/kNN difficulty: occupied

**Normalized (locally adaptive) conformal prediction** is a standard family.
The literature describes it directly: *"locally adaptive conformal methods modify
the nonconformity score so that interval width can vary with input-dependent
predictive difficulty, with a common strategy being to normalize residuals by an
estimated scale or difficulty function."* That is Candidate A's construction.

Directly relevant work:

- **Conformal and kNN Predictive Uncertainty Quantification Algorithms in Metric
  Spaces** (arXiv:2507.15741) — the closest match. Local kNN difficulty in metric
  spaces with finite-sample coverage guarantees, explicitly adaptive to
  heteroscedastic geometry.
- **A Framework for Uncertainty Quantification Based on Nearest Neighbors Across
  Layers** (arXiv:2506.19895) — post-hoc, model-agnostic, no retraining; the same
  design goal this task states.
- **Density-based nonconformity scores** via a kNN density surrogate, listed as
  one of three generic score families.

## Candidate B — ensemble disagreement as the score: occupied, and more directly

This is the more consequential finding. The literature states it plainly:
*"normalized conformal prediction normalizes nonconformity scores based on an
estimated degree of difficulty for each sample, with common methods including
**measuring the variance of ensemble predictions**."*

And specifically for tree models: *"a nonconformity score can be scaled by a
normalization factor defined as the standard deviation of ensemble predictions,
estimated using a Random Forest regressor composed of decision trees."*

That is Candidate B, exactly — including the model class this task wants it for.

- **Efficient Conformal Predictor Ensembles** (Neurocomputing, 2019) — finds
  variance-based nonconformity *"significantly outperforms standard
  non-normalized measures."*
- **Efficient Normalized Conformal Prediction ... with Deep Regression Forests**
  (arXiv:2402.14080) — ensemble-variance normalization on tree ensembles.
- **Nested conformal prediction and quantile out-of-bag ensemble methods**
  (Pattern Recognition, 2021) — OOB ensemble machinery for the same purpose.

The literature also lists the three standard difficulty estimators as: *ensemble
prediction variance, training an additional model to predict sample error, and
MC-dropout.* Note the middle one — **"training an additional model to predict
sample error" is errordir's β.** So this project's own mechanism is also a named
member of that family; what was novel was the validated 1-D geometry and the
data-efficiency result, not the idea of learning an error predictor.

## What this changes for Items 2–5

1. **Neither candidate can be framed as a new mechanism.** Both are named,
   published difficulty estimators. Items 2–3 are adaptation-and-verification.
2. **The interesting question narrows,** and it is still worth running: the
   literature reports variance-normalization *works* on tree ensembles, while
   this project found β fails Task 21's Check 2 there 0/6. Either (a) Candidate B
   passes Check 2 where β could not, which would mean the failure was β's linear
   form rather than tree models per se — a real, specific finding; or (b) it also
   fails Check 2, which would mean **Check 2 is stricter than what the published
   normalized-conformal literature requires**, since these methods are reported to
   work while being unable to clear a causal-adjacency bar.
3. Outcome (b) would be a finding about **the validation gate**, not about
   model-agnosticism — and per this project's history (Task 25 Item 1) that is
   worth knowing rather than papering over.
4. **No claim of novelty is available from this task** regardless of outcome. The
   honest framing is scope determination: does errordir's *validation standard*
   extend to tree models under any known difficulty estimator?

## Sources

- [Conformal and kNN Predictive Uncertainty Quantification Algorithms in Metric Spaces](https://arxiv.org/abs/2507.15741)
- [A Framework for Uncertainty Quantification Based on Nearest Neighbors Across Layers](https://arxiv.org/html/2506.19895v1)
- [Efficient Conformal Predictor Ensembles (Neurocomputing)](https://www.sciencedirect.com/science/article/abs/pii/S0925231219316108)
- [Efficient Normalized Conformal Prediction with Deep Regression Forests](https://arxiv.org/html/2402.14080)
- [Nested conformal prediction and quantile out-of-bag ensemble methods](https://www.sciencedirect.com/science/article/abs/pii/S0031320321006725)
- [Conformal Prediction: a Unified Review of Theory and New Challenges](https://arxiv.org/pdf/2005.07972)
