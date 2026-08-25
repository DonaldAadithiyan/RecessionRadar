# Task 36, Item 3 — Correction to Task 34's Report

## What Task 34 claimed, and why it is wrong

Task 34's Item 3 report and summary both drew a substantive conclusion from four
ridge rows:

> *"Four ridge fits passed Check 2; none passes DS... a direction can dominate on
> induced magnitude while carrying no cross-point difficulty information. That
> reframes what a Check 2 pass ever certified: **magnitude alignment, not
> difficulty relevance.** The four ridge passes across Tasks 24–33 were weaker
> evidence than they appeared."*

**That conclusion is withdrawn.** It rests on DS values computed where DS cannot
function.

Task 36 Item 1 measured δᵢ directly on those exact four fits: **one unique value
across 400 test points, CV ≈ 2×10⁻¹⁵.** The ridge prediction path in
`task34_3_retest.py` is three composed affine maps with no clipping or bound, so
δᵢ = |2h·(a·v)| carries no xᵢ dependence — the constancy is forced by the code,
not a property of the data. With δ constant, Spearman(δ, e) is tie-degenerate and
the reported percentile reflects tie-breaking noise, not the fit.

## Specific corrections, superseding the originals

| Task 34 statement | Status |
|---|---|
| "Four ridge fits passed Check 2; none passes DS" — presented as an informative contrast | **Superseded.** The DS half measured nothing. |
| "Check 2 certifies magnitude alignment, not difficulty relevance" | **Withdrawn.** No evidence supports this; it was inferred solely from the void ridge rows. |
| "The four ridge passes across Tasks 24–33 were weaker evidence than they appeared" | **Withdrawn.** Those Check 2 passes are unaffected by this task. |
| The guardrail's red flag "investigated and resolved" | **Resolution was wrong.** The correct resolution is that DS is inapplicable to linear models, not that Check 2 and DS measure different things on them. |

## What in Task 34 still stands

The correction is narrow and does **not** touch Task 34's main result:

- **Item 2's synthetic verification stands in full.** DS detects a known signal
  (percentile 1.000) and rejects known null in both normal and hyper-responsive
  regimes (0.639, 0.463). Those cases used *tree* models, where δᵢ genuinely
  varies — confirmed here at CV 0.49–0.82.
- **Check 2 scoring 0.000 on the known-signal case stands.** That remains a
  real demonstration that Check 2 fails to detect a difficulty direction present
  by construction, on tree-like data.
- **The tree retest stands: 0/16 under DS, at both n=40 and n=400.** Trees are
  the regime where DS is valid, and the n=400 rerun (median |DS| shrinking 67%
  as n grew 10×) remains the evidence that those values are noise around zero.
- **The n=40 power finding stands.** DS at n=40 has a pass bar of |DS| > 0.270,
  above any observed value; n ≥ 400 remains the requirement for future use.

So Task 34's headline — *the corrected test recovers zero tree validations, and
that is not an artifact of magnitude-blindness* — is unaffected. Only its
subsidiary claim about what Check 2 certifies on linear models is withdrawn.

## A note on how this error arose

DS was designed for the tree problem, verified on tree synthetic cases, and then
applied uniformly across the whole 24-cell table including linear fits. The
verification in Task 34 Item 2 was thorough **within its intended regime** and
said nothing about validity outside it. The gap was applying a
tree-motivated instrument to linear models without checking that its core
quantity varies there — a check that costs one line (`len(np.unique(δ))`) and
would have caught it immediately.

Recorded as a reusable lesson: **an instrument verified on one model class is not
thereby verified on another.** The same discipline this project applies to
mechanisms (ridge control arms since Task 25) applies to test statistics.
