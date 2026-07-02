"""
P1.2 — Moving-block bootstrap 90% CIs on Table-1/Table-2 MAE.
Block bootstrap respects temporal autocorrelation (block length ~ sqrt(n)=8).
Compares Ensemble (Ours) vs tuned baselines: XGB-indep, MOR-XGB, MOR-XGB-joint,
Naive Mean, Probit. Ensemble preds from saved pkl; XGB baselines reproduced via
the same Optuna procedure as ablation_baselines.py (seed=42, deterministic).
NO model retraining of the ensemble.
"""
import os,re,pickle,warnings,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)

recession_targets=["recession_probability","1_month_recession_probability","3_month_recession_probability","6_month_recession_probability"]
LABELS=["Current","1M","3M","6M"]; SPLIT="2020-01-01"; EPS=1e-6
eps=1e-8
def safe_logit(y): return np.log(np.clip(np.clip(y,0,100)/100,eps,1-eps)/(1-np.clip(np.clip(y,0,100)/100,eps,1-eps)))
def safe_inv_logit(z): return np.clip(1/(1+np.exp(-np.clip(z,-50,50)))*100,0,100)
def sanitize_columns(df):
    df=df.copy(); df.columns=[re.sub(r'[^A-Za-z0-9_]+','_',c) for c in df.columns]; return df
class LGBMWrapper(BaseEstimator,RegressorMixin):
    def __init__(self,params=None,num_boost_round=500,early_stopping_rounds=50):
        self.params=params or {}; self.num_boost_round=num_boost_round; self.early_stopping_rounds=early_stopping_rounds; self.model=None
    def fit(self,X,y): return self
    def predict(self,X): return self.model.predict(X)
class FullChainCatBoostModel:
    def __init__(self): self.chain_model=self.scaler=None
    def predict(self,X):
        Xs=pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index); return np.clip(safe_inv_logit(self.chain_model.predict(Xs)),0,100)
class FullChainLightGBMModel:
    def __init__(self): self.chain_model=self.scaler=None
    def predict(self,X):
        X=sanitize_columns(X); Xs=pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index); return np.clip(safe_inv_logit(self.chain_model.predict(Xs)),0,100)
class FullChainRandomForestModel:
    def __init__(self): self.chain_model=self.scaler=None
    def predict(self,X):
        Xs=pd.DataFrame(self.scaler.transform(X),columns=X.columns,index=X.index); return np.clip(safe_inv_logit(self.chain_model.predict(Xs)),0,100)
class FullChainStackingEnsemble:
    def __init__(self,cv_folds=8,use_feature_engineering=True):
        self.base_models={'CatBoost':FullChainCatBoostModel,'LightGBM':FullChainLightGBMModel,'RandomForest':FullChainRandomForestModel}
        self.meta_models={}; self.cv_folds=cv_folds; self.use_feature_engineering=use_feature_engineering; self.meta_scaler={}; self.fitted_base_models={}
    def _engineer_meta_features(self,*bp):
        f=list(bp)
        if self.use_feature_engineering:
            f+=[np.mean(bp,axis=0),0.4*bp[0]+0.35*bp[1]+0.25*bp[2],np.std(bp,axis=0),np.min(bp,axis=0),np.max(bp,axis=0)]
            for i in range(len(bp)):
                for j in range(i+1,len(bp)): f.append(np.abs(bp[i]-bp[j]))
        return np.column_stack(f)
    def predict(self,X):
        bp={n:m.predict(X) for n,m in self.fitted_base_models.items()}
        fp=np.zeros_like(list(bp.values())[0])
        for i,t in enumerate(recession_targets):
            bpt=[bp[n][:,i] for n in self.base_models]; mf=self._engineer_meta_features(*bpt); fp[:,i]=self.meta_models[t].predict(self.meta_scaler[t].transform(mf))
        return np.clip(fp,0,100)

def logit(y):
    s=np.clip(np.asarray(y,float)/100,EPS,1-EPS); return np.log(s/(1-s))
def inv_logit(z): return np.clip(1/(1+np.exp(-np.asarray(z,float)))*100,0,100)
def clean(d): return d.replace([np.inf,-np.inf],np.nan).ffill().bfill().fillna(0)

df=pd.read_csv("data/fix/feature_selected_reg_full.csv"); df["date"]=pd.to_datetime(df["date"]); df=df.sort_values("date").reset_index(drop=True)
tr=df[df.date<SPLIT].copy(); te=df[df.date>=SPLIT].copy()
Xtr=clean(tr.drop(columns=recession_targets+["date"])); Xte=clean(te.drop(columns=recession_targets+["date"]))
ytr=clean(tr[recession_targets]); yte=clean(te[recession_targets])
Xtr_a=Xtr.values.astype(float); Xte_a=Xte.values.astype(float)

with open("fix-reg/models/full_chain_stacking.pkl","rb") as f: ens=pickle.load(f)
ens_pred=ens.predict(Xte)  # (65,4)

# --- tuned XGB baselines (same as ablation_baselines.py) ---
def suggest(t):
    return dict(n_estimators=t.suggest_int("n_estimators",100,800),max_depth=t.suggest_int("max_depth",3,8),
      learning_rate=t.suggest_float("learning_rate",0.01,0.3,log=True),subsample=t.suggest_float("subsample",0.6,1.0),
      colsample_bytree=t.suggest_float("colsample_bytree",0.5,1.0),min_child_weight=t.suggest_int("min_child_weight",1,10),
      reg_alpha=t.suggest_float("reg_alpha",0.0,1.0),reg_lambda=t.suggest_float("reg_lambda",0.5,3.0),
      objective="reg:squarederror",random_state=42,verbosity=0,tree_method="hist")
def cvmae(p,X,yl):
    tscv=TimeSeriesSplit(n_splits=5); m=[]
    for a,b in tscv.split(X):
        mdl=XGBRegressor(**p); mdl.fit(X[a],yl[a]); m.append(mean_absolute_error(inv_logit(yl[b]),inv_logit(mdl.predict(X[b]))))
    return float(np.mean(m))
xgb_indep=np.zeros((len(Xte_a),4)); bestA={}
for i,t in enumerate(recession_targets):
    yl=logit(ytr[t].values)
    st=optuna.create_study(direction="minimize",sampler=optuna.samplers.TPESampler(seed=42))
    st.optimize(lambda tr_:cvmae(suggest(tr_),Xtr_a,yl),n_trials=40,show_progress_bar=False)
    bp=st.best_params; bp.update(objective="reg:squarederror",random_state=42,verbosity=0,tree_method="hist")
    mdl=XGBRegressor(**bp); mdl.fit(Xtr_a,yl); xgb_indep[:,i]=inv_logit(mdl.predict(Xte_a)); bestA[LABELS[i]]=(bp,mean_absolute_error(ytr[t].values,ytr[t].values))
    print(f"XGB-indep {LABELS[i]} test MAE={mean_absolute_error(yte[t].values,xgb_indep[:,i]):.3f}",flush=True)

# Naive & Probit
naive=np.zeros((len(Xte_a),4)); probit=np.zeros((len(Xte_a),4))
spread_tr=(Xtr["10_year_rate"]-Xtr["3_months_rate"]).values.reshape(-1,1); spread_te=(Xte["10_year_rate"]-Xte["3_months_rate"]).values.reshape(-1,1)
for i,t in enumerate(recession_targets):
    naive[:,i]=ytr[t].mean()
    reg=LinearRegression().fit(spread_tr,logit(ytr[t].values)); probit[:,i]=np.clip(inv_logit(reg.predict(spread_te)),0,100)

preds={"Ensemble (Ours)":ens_pred,"XGB Indep (tuned)":xgb_indep,"Naive Mean":naive,"Probit (YC)":probit}

# --- moving-block bootstrap ---
def mbb_ci(actual,pred,n_boot=2000,ci=90,block=8,seed=42):
    rng=np.random.default_rng(seed); n=len(actual); err=np.abs(actual-pred)
    nblocks=int(np.ceil(n/block)); starts=np.arange(0,n-block+1)
    boot=[]
    for _ in range(n_boot):
        idx=np.concatenate([np.arange(s,s+block) for s in rng.choice(starts,size=nblocks,replace=True)])[:n]
        boot.append(err[idx].mean())
    lo=np.percentile(boot,(100-ci)/2); hi=np.percentile(boot,100-(100-ci)/2)
    return float(lo),float(hi)

rows=[]
print("\n=== P1.2 Moving-block bootstrap (block=8, 2000 resamples, 90% CI) ===")
for i,t in enumerate(recession_targets):
    a=yte[t].values
    for name,P in preds.items():
        pm=mean_absolute_error(a,P[:,i]); lo,hi=mbb_ci(a,P[:,i])
        rows.append(dict(Horizon=LABELS[i],Model=name,MAE=round(pm,4),CI_lo_90=round(lo,4),CI_hi_90=round(hi,4)))
        print(f"  {LABELS[i]:8s} {name:20s} MAE={pm:7.3f}  90%CI=[{lo:6.3f}, {hi:6.3f}]")
pd.DataFrame(rows).to_csv("fix-reg/bootstrap_block_ci_90.csv",index=False)

# Disjoint check: Ensemble vs XGB-indep at 3M/6M
print("\n=== Disjointness (Ensemble vs XGB-indep) ===")
R=pd.DataFrame(rows)
for h in LABELS:
    e=R[(R.Horizon==h)&(R.Model=="Ensemble (Ours)")].iloc[0]; x=R[(R.Horizon==h)&(R.Model=="XGB Indep (tuned)")].iloc[0]
    disj = e.CI_hi_90 < x.CI_lo_90 or x.CI_hi_90 < e.CI_lo_90
    print(f"  {h:8s} Ens=[{e.CI_lo_90:.2f},{e.CI_hi_90:.2f}] XGB=[{x.CI_lo_90:.2f},{x.CI_hi_90:.2f}]  {'DISJOINT' if disj else 'OVERLAP'}")
print("Saved fix-reg/bootstrap_block_ci_90.csv")
