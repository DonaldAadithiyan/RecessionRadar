"""
TASK 8 figures/tables — literature baselines vs internal models.

Produces:
  figures/task8_model_independence.{pdf,png}
      Diversity-optimal's coverage gain over pooled/trailing, plotted against
      the underlying model that produced the nonconformity scores. The point of
      the panel is that the gain is flat across model families: a 2010 clinical
      index and a gradient-boosting regressor give the same calibration story.
  fix-reg/task8_paper_tables.md
      Paper-ready tables: point-prediction comparison per domain (mirroring
      Table 6's recession format) and the extended calibration table
      (mirroring Tables 8/9, now with 4 underlying models per domain).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_FIG = "figures"
OUT = "fix-reg"
os.makedirs(OUT_FIG, exist_ok=True)

SRC = [
    ("Healthcare", "fix-reg/task8_healthcare_literature_baselines.csv", "literature"),
    ("Healthcare", "fix-reg/task7_baselines_healthcare.csv", "internal"),
    ("Climate", "fix-reg/task8_climate_literature_baselines.csv", "literature"),
    ("Climate", "fix-reg/task7_baselines_climate.csv", "internal"),
]

rows = []
for dom, path, kind in SRC:
    if not os.path.exists(path):
        continue
    d = pd.read_csv(path)
    # Filter by the file's own domain column rather than trusting the loop
    # variable: the Task 7 CSVs carry their domain explicitly, and keying off
    # `dom` alone would silently mix models across panels.
    d = d[d["domain"] == dom]
    for m, g in d.groupby("model"):
        base = g[g.strategy == "pooled_trailing"].iloc[0]
        div = g[g.strategy == "diversity_optimal"].iloc[0]
        rows.append(dict(domain=dom, kind=kind, model=m,
                         base_cov=base.coverage, div_cov=div.coverage,
                         gain=round(div.coverage - base.coverage, 2),
                         div_lo=div.wilson_lo, div_hi=div.wilson_hi,
                         wmult=round(div.mean_width / base.mean_width, 2),
                         sep=bool(div.wilson_lo > base.coverage)))
t = pd.DataFrame(rows)

# ── Figure: coverage gain by underlying model family ─────────────────────────
# NOT sharey: each panel has its own model names on the y-axis, and a shared
# axis would render only one panel's labels for both.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, dom in zip(axes, ["Healthcare", "Climate"]):
    d = t[t.domain == dom].sort_values(["kind", "model"])
    if d.empty:
        ax.set_visible(False); continue
    y = np.arange(len(d))[::-1]
    colors = ["#1a1a1a" if k == "literature" else "#5a7fa8" for k in d.kind]
    for i, (yy, (_, r)) in enumerate(zip(y, d.iterrows())):
        ax.plot([r.base_cov, r.div_cov], [yy, yy], color=colors[i], lw=2.0,
                alpha=.5, zorder=2)
        ax.plot(r.base_cov, yy, "o", ms=6, mfc="white", mec=colors[i],
                mew=1.8, zorder=3)
        ax.plot(r.div_cov, yy, "o", ms=9, color=colors[i], zorder=3)
        ax.annotate(f"+{r.gain:.1f}pp  ({r.wmult:.1f}x width)",
                    (max(r.div_cov, r.base_cov) + 0.6, yy), va="center",
                    fontsize=8.5, color="#444")
    ax.axvline(90, color="crimson", lw=1.3, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.model}\n({r.kind})" for _, r in d.iterrows()],
                       fontsize=9)
    ax.set_title(f"{dom}", fontsize=12, pad=8)
    ax.set_xlabel("Coverage %  (hollow = pooled/trailing, solid = diversity-optimal)",
                  fontsize=9)
    ax.grid(axis="x", alpha=.25, lw=.7)
    ax.set_axisbelow(True)
    ax.set_xlim(min(80, d.base_cov.min() - 4), 108)

fig.suptitle("The calibration-diversity effect does not depend on which model "
             "produced the scores\n"
             "(black = published literature baseline, blue = internal ML model; "
             "red line = 90% nominal)",
             fontsize=11.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT_FIG}/task8_model_independence.{ext}", dpi=170,
                bbox_inches="tight")
print(f"Saved {OUT_FIG}/task8_model_independence.pdf / .png")

# ── Paper-ready tables ───────────────────────────────────────────────────────
md = ["# Task 8 — paper-ready tables", "",
      "## Point-prediction comparison (mirrors Table 6's recession format)", ""]

hp = "fix-reg/task8_healthcare_point_prediction.csv"
if os.path.exists(hp):
    h = pd.read_csv(hp)
    md += ["### Healthcare (§6.3)", "",
           "| Model | Origin | AUC-ROC (binary) | OOF MAE (cohort rate, pts) |",
           "|---|---|---|---|"]
    for _, r in h.iterrows():
        md.append(f"| {r.baseline} | van Walraven 2010 / Donzé 2013 | "
                  f"{r.auc_binary:.4f} | {r.oof_mae_rate:.3f} |")
    for m, mae in [("ridge", None), ("gradboost", None)]:
        md.append(f"| {m} | internal sensitivity check | — | see §6.3 |")
    md += ["", "Published reference points on this dataset (Emi-Johnson et al. "
           "2026): LACE-based logistic ≈0.608, logistic ≈0.642, RF ≈0.630, "
           "XGBoost ≈0.667.", ""]

cp = "fix-reg/task8_climate_point_prediction.csv"
if os.path.exists(cp):
    c = pd.read_csv(cp)
    md += ["### Climate (§6.4)", "",
           "| Model | Origin | OOF MAE | Test MAE | Score support (p95−p5) |",
           "|---|---|---|---|---|"]
    for _, r in c.iterrows():
        md.append(f"| {r.baseline} | WeatherBench baseline tier "
                  f"(Rasp et al. 2020) | {r.oof_mae:.3f} | {r.test_mae:.3f} | "
                  f"{r.score_support:.3f} |")
    md += [""]

md += ["## Calibration comparison, extended to 4 underlying models per domain",
       "", "(mirrors Tables 8/9; coverage % with 95% Wilson interval)", ""]
for dom, paths in [("Healthcare",
                    ["fix-reg/task8_healthcare_literature_baselines.csv",
                     "fix-reg/task7_baselines_healthcare.csv"]),
                   ("Climate",
                    ["fix-reg/task8_climate_literature_baselines.csv",
                     "fix-reg/task7_baselines_climate.csv"])]:
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        continue
    d = pd.concat(frames)
    d = d[d["domain"] == dom]     # keep panels domain-pure (see note above)
    piv = d.pivot_table(index="strategy", columns="model", values="coverage")
    md += [f"### {dom}", "", "| Strategy | " + " | ".join(piv.columns) + " |",
           "|" + "---|" * (len(piv.columns) + 1)]
    order = ["pooled_trailing", "mondrian", "pid_conformal", "evt_tail",
             "dtaci", "acmcp", "bellman_ci", "diversity_optimal"]
    for s in order:
        if s not in piv.index:
            continue
        cells = " | ".join(f"{piv.loc[s, c]:.2f}" for c in piv.columns)
        star = " **(ours)**" if s == "diversity_optimal" else ""
        md.append(f"| {s}{star} | {cells} |")
    md += [""]

with open(f"{OUT}/task8_paper_tables.md", "w") as f:
    f.write("\n".join(md))
print(f"Saved {OUT}/task8_paper_tables.md")
