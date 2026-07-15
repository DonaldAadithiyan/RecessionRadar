"""
PHASE 4a — Synthetic rare-event injection.
Fully synthetic (no model): we build a base series of nonconformity scores for a
calibration POOL and a TEST set, where scores are drawn from an "expansion"
distribution in normal months and an inflated "rare-event" distribution during
simulated recession episodes. We then run the SAME Phase-1 analysis (fixed-size
random subsets -> diversity stat vs ACI coverage + matched pairs) on each variant.

ACI here is applied to residual magnitudes directly: an interval covers a test
point if that point's true nonconformity score <= q (the calibration quantile).
This isolates exactly the calibration-diversity -> coverage mechanism, free of any
particular forecasting model. Varies 3 axes: frequency, magnitude, clustering.
No model retraining.
"""
import numpy as np, pandas as pd
from scipy import stats
rng_global=np.random.default_rng(2026)

POOL=635; TEST=65; N_FIX=254; N_DRAWS=150; GAMMA=0.005; ALPHA=0.10

def make_series(n, rare_frac, rare_mag, cluster_len, base_scale=1.0, seed=0):
    """Return (scores, is_rare) for n months. Expansion scores ~ HalfNormal(base_scale);
    rare-event scores ~ HalfNormal(base_scale*rare_mag). Rare months arranged in
    clusters of length cluster_len at rare_frac overall frequency."""
    rng=np.random.default_rng(seed)
    is_rare=np.zeros(n,bool)
    n_rare=int(round(rare_frac*n))
    if n_rare>0:
        n_clusters=max(1,int(round(n_rare/cluster_len)))
        starts=rng.choice(np.arange(0,n-cluster_len), size=n_clusters, replace=False)
        for s in starts: is_rare[s:s+cluster_len]=True
        # trim/pad to hit ~n_rare
    scores=np.abs(rng.normal(0,base_scale,n))
    scores[is_rare]=np.abs(rng.normal(0,base_scale*rare_mag,int(is_rare.sum())))
    return scores, is_rare

def aci_coverage_scores(test_scores, cal_scores, gamma=GAMMA, alpha_t=ALPHA):
    """ACI on nonconformity magnitudes: covered_t = 1 if test_score_t <= q_t."""
    covered=[]; a=alpha_t; widths=[]
    for t in range(len(test_scores)):
        q=np.quantile(cal_scores,np.clip(1-a,0,1)); widths.append(2*q)
        miss=1 if test_scores[t]>q else 0; covered.append(1-miss)
        a=np.clip(a+gamma*(alpha_t-miss),0.01,0.99)
    return np.mean(covered)*100, np.mean(widths)

def support(s): return float(np.percentile(s,95)-np.percentile(s,5))

def run_variant(name, rare_frac, rare_mag, cluster_len):
    # fixed synthetic test set for this variant
    test_scores,_=make_series(TEST, rare_frac, rare_mag, cluster_len, seed=999)
    pool_scores, pool_rare = make_series(POOL, rare_frac, rare_mag, cluster_len, seed=1)
    xs=[]; ys=[]; rares=[]
    for d in range(N_DRAWS):
        idx=rng_global.choice(POOL,size=N_FIX,replace=False)
        cs=pool_scores[idx]
        cov,_=aci_coverage_scores(test_scores,cs)
        xs.append(support(cs)); ys.append(cov); rares.append(int(pool_rare[idx].sum()))
    xs=np.array(xs); ys=np.array(ys); rares=np.array(rares)
    rho_supp = stats.spearmanr(xs,ys).correlation if np.std(ys)>1e-9 else float("nan")
    rho_rare = stats.spearmanr(rares,ys).correlation if (np.std(ys)>1e-9 and np.std(rares)>0) else float("nan")
    r_supp = stats.pearsonr(xs,ys).statistic if np.std(ys)>1e-9 else float("nan")
    # matched-pairs: within support tertiles, corr(rare,cov)
    within=[]
    if np.std(ys)>1e-9:
        terts=np.quantile(xs,[0,1/3,2/3,1.0])
        for ti in range(3):
            m=(xs>=terts[ti])&(xs<=terts[ti+1] if ti==2 else xs<terts[ti+1])
            if m.sum()>=6 and np.std(rares[m])>0 and np.std(ys[m])>0:
                within.append(stats.spearmanr(rares[m],ys[m]).correlation)
        within_mean=float(np.nanmean(within)) if within else float("nan")
    else: within_mean=float("nan")
    return dict(variant=name, rare_frac=rare_frac, rare_mag=rare_mag, cluster_len=cluster_len,
                rho_supp=round(rho_supp,3), R2_supp=round(r_supp**2,3) if not np.isnan(r_supp) else None,
                rho_rare=round(rho_rare,3) if not np.isnan(rho_rare) else None,
                within_tertile_rho_rare=round(within_mean,3) if not np.isnan(within_mean) else None,
                cov_mean=round(float(np.mean(ys)),2), cov_std=round(float(np.std(ys)),2))

variants=[
    # baseline anchored to real data: ~50/635=7.9% freq, ~3x magnitude, cluster ~ 6mo
    ("baseline(real-like)",     0.079, 3.0, 6),
    # frequency axis
    ("freq_low(2%)",            0.02,  3.0, 6),
    ("freq_high(20%)",          0.20,  3.0, 6),
    # magnitude axis
    ("mag_low(1.5x)",           0.079, 1.5, 6),
    ("mag_high(6x)",            0.079, 6.0, 6),
    # clustering axis
    ("cluster_isolated(1mo)",   0.079, 3.0, 1),
    ("cluster_long(12mo)",      0.079, 3.0, 12),
]
rows=[run_variant(*v) for v in variants]
df=pd.DataFrame(rows)
df.to_csv("phase4a_synthetic_generalization.csv",index=False)
print(df.to_string(index=False))
print("\nSaved phase4a_synthetic_generalization.csv")
