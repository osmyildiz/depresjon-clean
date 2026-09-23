"""Two controls the novelty-gate methodology seat said would decide the verdict.

1. Permuted-covariate residualisation. Within each LOSO fold, residualise every
   feature on a label-uncorrelated PERMUTATION of log(day count) (same variance,
   no relation to the label) and on a Gaussian covariate. If bagged trees still
   fall to ~0.75, the ~10-point drop under real residualisation is an artefact
   of residualising 1000+ features on 54 points, not a confound measurement.
   Also: residualise on real log D, then append log D itself as a feature — if
   accuracy returns, the removed signal was recording length; if not, it was
   nonlinear structure the trees used.
2. Profile ablation. 24-h profile ONLY (48 columns) vs 1092 TSFEL only vs both,
   same three fixed learners, same five seeds.
Thread-pinned; single process.
"""
import sys, json, os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ.setdefault(k,"1")
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject

SEEDS=[0,1,2,3,4]
X_raw,y_day,g=load_depresjon("data/depresjon"); g=np.asarray(g); y_day=np.asarray(y_day)
fd=drop_unstable(extract_tsfel(X_raw,g,cache_dir="cache",tag="depresjon",n_jobs=-1))
feats,y,ids=aggregate_per_subject(fd,g,y_day,raw_activity=None); y=np.asarray(y); n=len(y)
X1092=np.nan_to_num(feats.to_numpy(np.float32))
hour=X_raw.reshape(len(X_raw),24,60).mean(2)
prof=np.array([hour[g==s].mean(0) for s in ids],np.float32); shape=prof/(prof.sum(1,keepdims=True)+1e-9)
Xprof=np.hstack([prof,shape]); Xboth=np.hstack([X1092,Xprof])
logD=np.log(np.array([int((g==s).sum()) for s in ids],float)).reshape(-1,1)

def learners(seed): return {
 "rf": lambda: RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=seed),
 "bagging": lambda: BaggingClassifier(DecisionTreeClassifier(random_state=seed),n_estimators=200,n_jobs=1,random_state=seed),
 "logreg": lambda: LogisticRegression(max_iter=3000,C=0.1)}
def fit(c,X,yy):
    try: return c.fit(X,yy,sample_weight=compute_sample_weight("balanced",yy))
    except TypeError: return c.fit(X,yy)

def loso(X,seed,cov=None,append_cov=False):
    """cov: (n,1) covariate to residualise on within fold (None = no residualisation)."""
    pred={k:np.zeros(n,int) for k in learners(seed)}
    for tr,te in LeaveOneOut().split(X):
        Xtr,Xte=X[tr].copy(),X[te].copy()
        if cov is not None:
            lr=LinearRegression().fit(cov[tr],Xtr); Xtr-=lr.predict(cov[tr]); Xte-=lr.predict(cov[te])
            if append_cov: Xtr=np.hstack([Xtr,logD[tr]]); Xte=np.hstack([Xte,logD[te]])
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        for k,f in learners(seed).items(): pred[k][te[0]]=fit(f(),Xtr,y[tr]).predict(Xte)[0]
    return {k:int((v==y).sum()) for k,v in pred.items()}

def block(tag,fn):
    rs=[fn(s) for s in SEEDS]; res={}
    for k in rs[0]:
        ks=[r[k] for r in rs]; res[k]={"correct":ks,"mean_acc":round(float(np.mean(ks))/n,4),"min_acc":round(min(ks)/n,4),"max_acc":round(max(ks)/n,4)}
        print(f"{tag:36s} {k:8s} mean={np.mean(ks)/n:.3f} range=[{min(ks)/n:.3f},{max(ks)/n:.3f}] {ks}",flush=True)
    return res

out={}
# --- 2. profile ablation
out["profile_only"]=block("profile_only(48)",lambda s: loso(Xprof,s))
out["tsfel1092_plus_profile"]=block("1092+profile",lambda s: loso(Xboth,s))
# --- 1. residualisation controls on the 1092 representation
rng=np.random.RandomState(123)
perm=logD[rng.permutation(n)]                       # same values, label-unrelated order
gauss=rng.normal(logD.mean(),logD.std(),size=logD.shape)
out["resid_real_logD"]=block("1092 resid real logD",lambda s: loso(X1092,s,cov=logD))
out["resid_permuted_logD"]=block("1092 resid PERMUTED logD",lambda s: loso(X1092,s,cov=perm))
out["resid_gaussian"]=block("1092 resid gaussian cov",lambda s: loso(X1092,s,cov=gauss))
out["resid_real_plus_logD_feature"]=block("1092 resid real + logD appended",lambda s: loso(X1092,s,cov=logD,append_cov=True))
Path("results/decisive_controls.json").write_text(json.dumps(out,indent=2)); print("written")
