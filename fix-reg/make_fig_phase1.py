"""Phase 1 figure: support-width (diversity) vs ACI coverage, 4 horizon facets.
From fix-reg/phase1_random_sweep.csv. Paper style: serif 8pt, blue points."""
import os,numpy as np,pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"..","figures"); os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","Times","DejaVu Serif"],
    "font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5,
    "axes.linewidth":0.6,"figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"ps.fonttype":42})
BLUE,GRID="#2a78d6","#d9d9d6"
df=pd.read_csv(os.path.join(HERE,"phase1_random_sweep.csv"))
horizons=["Current","1M","3M","6M"]
fig,axes=plt.subplots(1,4,figsize=(7.0,2.1),sharey=True)
for ax,h in zip(axes,horizons):
    x=df[f"{h}_supp"].values; y=df[f"{h}_cov"].values
    ax.scatter(x,y,s=7,color=BLUE,alpha=0.5,edgecolor="none")
    if np.std(y)>1e-9:
        rho=stats.spearmanr(x,y).correlation; r=stats.pearsonr(x,y).statistic
        ax.text(0.04,0.04,f"ρ={rho:.2f}\nR²={r**2:.2f}",transform=ax.transAxes,fontsize=6.5,va="bottom")
    ax.axhline(90,color="#6a6a6a",ls=(0,(4,3)),lw=0.7,zorder=1)
    ax.set_title(h,fontsize=8); ax.set_xlabel("Score support (p95−p5, pp)")
    ax.grid(color=GRID,lw=0.4); 
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5,width=0.6)
axes[0].set_ylabel("ACI coverage (%)")
fig.savefig(os.path.join(OUT,"phase1_diversity_vs_coverage.pdf"),format="pdf")
fig.savefig(os.path.join(OUT,"phase1_diversity_vs_coverage.png"),format="png")
plt.close(fig); print("wrote figures/phase1_diversity_vs_coverage.pdf + .png")
