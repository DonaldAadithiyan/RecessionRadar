# Task 17 Item 4 — Correlated-Generator Synthetic Construction

**Verdict: the corrected generator SUCCEEDS. It predicts the right sign at
4 of 4 real domains, against 0 of 4 for Task 16's independent generator. The
diagnosed cause — independence between rare-event status and score magnitude —
was correct, and correlation through a shared latent severity is the missing
ingredient.**

This item was explicitly allowed to fail. It did not, and that outcome is
reported with the same scrutiny a failure would have received.

Script: `fix-reg/task17_4_correlated_synthetic.py`.
Data: `task17_4_correlated_synthetic.csv`, `task17_4_sign_check.csv`.
Figure: `figures/task17_correlated_phase_diagram.{pdf,png}`.

---

## The generator, specified from the diagnosed cause

Task 16C found the independent-draw generator made rare-count artificially
informative: knowing how many rare draws you took nearly determined the upper
tail, which is untrue of real scores. The fix couples both through a latent
severity:

```
Z_t = 0.6·Z_{t-1} + η_t                 persistent latent severity
P(R_t = 1 | Z_t) = sigmoid(2.0·Z_t + b) event probability driven by Z
S_t = |N(0,1) + Δ_S·Z_t|                score magnitude driven by the SAME Z
```

`b` is solved per cell to hit the target prevalence. **Coupling strength
(a = 2.0) and persistence (ρ = 0.6) were fixed in advance and not tuned**, per
the spec's guardrail: one design, derived from the mechanism, run once.

## Result — the sign check

| Domain | Frequency | Δ_S | **Corrected generator** | Task 16 independent | Actual |
|---|---|---|---|---|---|
| Recession 3M | 8.4% | 2.19 | **+0.389** ✓ | −0.16 ✗ | +0.402 |
| Recession 6M | 8.4% | 1.26 | **+0.269** ✓ | −0.02 ✗ | +0.442 |
| Healthcare | 10.3% | 2.41 | **+0.421** ✓ | −0.24 ✗ | +0.122 |
| Climate | 10.7% | 2.97 | **+0.439** ✓ | −0.36 ✗ | +0.062 |

**4 of 4 correct, versus 0 of 4.** And recession 3M is close in magnitude too
(+0.389 predicted vs +0.402 measured), not merely correct in sign.

The whole plane is now positive: **60 of 60 cells** show diversity dominating,
against 50 of 66 (with a large negative region) under the independent generator.

## What this establishes

**The independence artefact was the problem, and it is now identified
precisely.** This is a genuinely strong addition to §5.2: the synthetic section
can be restored as corroboration, provided it uses the correlated construction
and states why the earlier one failed.

**The mechanism explanation is now complete.** Under independence, rare-count is
a sufficient statistic for the upper tail by construction, so it wins. Under
correlated severity — as in real data, where a severe month is both more likely
to be labelled rare *and* to produce a large score — the rare-count label is a
noisy proxy for severity while the score distribution's spread measures it
directly. That is why diversity dominates in reality.

## Two honesty notes

**The generator now over-predicts the gap in two domains.** Healthcare
(+0.421 predicted vs +0.122 actual) and climate (+0.439 vs +0.062) are correct
in sign but roughly 3–7× too large in magnitude. The generator captures the
direction of the mechanism, not its strength. It should be cited as
qualitative corroboration only — a claim that it reproduces the real effect
*sizes* would not survive inspection.

**One grid column is a seed artifact and is disclosed rather than smoothed.**
At Δ_S = 0.75 the gap collapses to ≈ +0.02 at every frequency, breaking an
otherwise smooth pattern. Diagnosed: `rho_supp` is identical (0.165) and mean
coverage identical (95.38) across all six frequencies in that column, because
the test stream uses a fixed seed (999) and that one Δ_S drew an unlucky test
set. It is a single-seed artifact in one column, not a property of the
generator. No real domain sits near Δ_S = 0.75 (all are ≥ 1.26), so the sign
check is unaffected — but the column should not be read as a genuine dip.

## Honest caveats

- **One seed per cell**, as with Task 16C. The Δ_S = 0.75 column shows exactly
  what that costs. A multi-seed version would smooth it; it was not run because
  the spec asked for one design run once, and re-running with different seeds
  until the artifact disappeared would edge toward the tuning the guardrail
  forbids.
- **Magnitudes are wrong even where signs are right** (see above). Corroboration
  is qualitative.
- **Two free parameters were still chosen**, a = 2.0 and ρ = 0.6. They were set
  in advance rather than tuned, but they were not derived from data either, and
  a different choice could change the magnitudes.
- **Δ_S means something different here.** Under the independent generator it was
  a direct mean shift; here it scales the latent coupling. The axes are
  comparable in role but not identical in units, so the two phase diagrams
  should be read as answering the same question, not as the same measurement.
- **This does not re-validate Task 16C's negative result as wrong.** That
  generator genuinely fails; this one genuinely succeeds. The paper should
  report both, because the contrast is the finding.
