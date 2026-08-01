"""
TASK 7 figure — coverage vs width across all three domains.

With 8 strategies x 3 domains a single grouped bar chart is unreadable, so the
figure is a 1x3 panel of coverage-with-CI dot plots (one panel per domain),
each annotated with the width multiple relative to the pooled/trailing
baseline. Width is shown because a coverage "win" bought with vacuous
intervals is not a win — the same convention used throughout this repo.

Output: figures/methods_coverage_comparison_alldomains.{pdf,png}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

ORDER = ["pooled_trailing", "mondrian", "pid_conformal", "evt_tail",
         "dtaci", "acmcp", "bellman_ci", "diversity_optimal"]
NICE = {"pooled_trailing": "Pooled/trailing ACI", "mondrian": "Mondrian",
        "pid_conformal": "PID-conformal", "evt_tail": "EVT-tail",
        "dtaci": "DtACI (2024)", "acmcp": "AcMCP (2024)",
        "bellman_ci": "Bellman CI (2024)",
        "diversity_optimal": "Diversity-optimal (ours)"}

PANELS = [
    ("Recession (6M, out-of-fold)", "fix-reg/task7_baselines_recession.csv",
     lambda d: d[(d.scoring == "out-of-fold") & (d.horizon == "6M")]),
    ("Healthcare (30-day readmission)", "fix-reg/task7_baselines_healthcare.csv",
     lambda d: d[d.model == "gradboost"]),
    ("Climate (storm intensity)", "fix-reg/task7_baselines_climate.csv",
     lambda d: d[d.model == "gradboost"]),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.0), sharey=True)

for ax, (title, path, sel) in zip(axes, PANELS):
    if not os.path.exists(path):
        ax.set_visible(False)
        continue
    d = sel(pd.read_csv(path))
    base_w = float(d[d.strategy == "pooled_trailing"]["mean_width"].iloc[0])

    rows = [d[d.strategy == s] for s in ORDER]
    rows = [r for r in rows if not r.empty]
    labels = [NICE[r["strategy"].iloc[0]] for r in rows]
    cov = np.array([float(r["coverage"].iloc[0]) for r in rows])
    lo = np.array([float(r["wilson_lo"].iloc[0]) for r in rows])
    hi = np.array([float(r["wilson_hi"].iloc[0]) for r in rows])
    wmult = np.array([float(r["mean_width"].iloc[0]) / base_w for r in rows])
    is_ours = np.array([r["strategy"].iloc[0] == "diversity_optimal" for r in rows])

    y = np.arange(len(rows))[::-1]
    ax.axvline(90, color="crimson", lw=1.4, ls="--", zorder=1,
               label="90% nominal")
    for i, yy in enumerate(y):
        c = "#1a1a1a" if is_ours[i] else "#5a7fa8"
        ax.plot([lo[i], hi[i]], [yy, yy], color=c, lw=2.2, alpha=.75, zorder=2)
        ax.plot(cov[i], yy, "o", ms=9 if is_ours[i] else 7, color=c, zorder=3)
        ax.annotate(f"{wmult[i]:.1f}x", (hi[i] + 0.8, yy), va="center",
                    fontsize=8.5, color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_title(title, fontsize=11.5, pad=9)
    ax.set_xlabel("Coverage % (dot) with 95% Wilson interval (bar)", fontsize=9.5)
    ax.grid(axis="x", alpha=.25, lw=.7)
    ax.set_axisbelow(True)
    lo_x = min(50, float(np.min(lo)) - 3)
    ax.set_xlim(lo_x, 103)

axes[0].legend(loc="lower left", fontsize=8.5, framealpha=.9)
fig.suptitle("Calibration strategies across three rare-event domains "
             "(annotation = mean interval width relative to pooled/trailing)",
             fontsize=12.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/methods_coverage_comparison_alldomains.{ext}",
                dpi=170, bbox_inches="tight")
print(f"Saved {OUT}/methods_coverage_comparison_alldomains.pdf / .png")
