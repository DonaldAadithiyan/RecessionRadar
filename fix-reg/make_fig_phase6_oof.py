"""Out-of-fold analogue of F_methods (figures/methods_coverage_comparison.pdf).

Same house style as make_fig_phase6.py (serif 8pt, validated palette,
grayscale-safe hatching), but every number is out-of-fold rather than
in-sample, and each bar carries its 95% Wilson interval.

Source: fix-reg/task7_baselines_recession.csv, rows with scoring=="out-of-fold".
Those rows come from make_fig_task7.py's upstream script
(task7_baseline_horse_race.py), which runs the identical strategy code over the
rolling-origin `oof_pred` array imported from task_oof_and_probit.py — the same
OOF predictions behind task3_insample_vs_oof.csv.

No number here is computed by this script; it only reads and plots.

Output: figures/methods_coverage_comparison_oof.{pdf,png}
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
    "font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.3,
    "axes.linewidth":0.6,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"ps.fonttype":42})
BLUE,YEL,GREEN,RED,ORANGE,VIOLET="#2a78d6","#eda100","#008300","#e34948","#eb6834","#4a3aa7"
GRID="#d9d9d6"; H=["Current","1M","3M","6M"]

df = pd.read_csv(os.path.join(HERE, "task7_baselines_recession.csv"))
oof = df[(df.scoring == "out-of-fold") & (df.model == "stacking-chain")]

# (label, strategy key in the CSV, colour, hatch) — order matches the in-sample
# figure, with EVT-tail added as the fifth series.
series = [("Pooled ACI (trailing)", "pooled_trailing", BLUE,   ""),
          ("Mondrian ACI",          "mondrian",        RED,    "xx"),
          ("PID-conformal",         "pid_conformal",   GREEN,  ".."),
          ("EVT tail fit",          "evt_tail",        VIOLET, "--"),
          ("Diversity-optimal",     "diversity_optimal", ORANGE, "//")]


def pull(key, col):
    """Exact value for (strategy, horizon); raises if a cell is missing."""
    out = []
    for h in H:
        r = oof[(oof.strategy == key) & (oof.horizon == h)]
        if len(r) != 1:
            raise SystemExit(f"expected exactly 1 OOF row for {key}/{h}, got {len(r)}")
        out.append(float(r[col].iloc[0]))
    return np.array(out)


fig, ax = plt.subplots(figsize=(3.3, 2.5))
x = np.arange(len(H)); w = 0.16
for k, (name, key, c, ht) in enumerate(series):
    cov = pull(key, "coverage")
    lo, hi = pull(key, "wilson_lo"), pull(key, "wilson_hi")
    # asymmetric Wilson interval, expressed as distance from the bar top
    err = np.vstack([cov - lo, hi - cov])
    ax.bar(x + (k - 2) * w, cov, w, label=name, color=c, alpha=0.85,
           edgecolor="white", linewidth=0.4, hatch=ht, zorder=2)
    ax.errorbar(x + (k - 2) * w, cov, yerr=err, fmt="none", ecolor="#333",
                elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=4)

ax.axhline(90, color="#444", ls=(0, (4, 3)), lw=0.8, zorder=5)
ax.text(-0.48, 90.4, "90%", fontsize=6, color="#444", va="bottom", ha="left")
ax.set_xticks(x); ax.set_xticklabels(H); ax.set_ylabel("ACI coverage (%), out-of-fold")
ax.set_ylim(55, 103); ax.set_yticks([60, 70, 80, 90, 100])
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.4, zorder=0); ax.tick_params(length=2.5, width=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False,
          handlelength=1.5, columnspacing=1.0)
fig.savefig(os.path.join(OUT, "methods_coverage_comparison_oof.pdf"), format="pdf")
fig.savefig(os.path.join(OUT, "methods_coverage_comparison_oof.png"), format="png")
plt.close(fig); print("wrote figures/methods_coverage_comparison_oof.pdf + .png")
