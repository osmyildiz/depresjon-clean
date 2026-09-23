"""Pre-specified, selection-free learners on the paper's own 1092-feature representation,
5 seeds, LOSO; plus the within-fold log(day-count) residualised variant. Completes the
comparison with new_formulations.py (which adds the 24-h profile)."""
import sys, json, os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
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
feats,y,ids=aggregate_per_subject(fd,g,y_day,raw_activity=None)
X=np.nan_to_num(feats.to_numpy(np.float32)); y=np.asarray(y); n=len(y)
logD=np.log(np.array([int((g==s).sum()) for s in ids])).reshape(-1,1)

def learners(seed): return {
 "rf": lambda: RandomForestClassifier(n_estimators=500,n_jobs=1,random_state=seed),
 "bagging": lambda: BaggingClassifier(DecisionTreeClassifier(random_state=seed),n_estimators=200,n_jobs=1,random_state=seed),
 "logreg": lambda: LogisticRegression(max_iter=3000,C=0.1)}
def fit(c,X,yy):
    try: return c.fit(X,yy,sample_weight=compute_sample_weight("balanced",yy))
    except TypeError: return c.fit(X,yy)
def run(seed,resid):
    pred={k:np.zeros(n,int) for k in learners(seed)}
    for tr,te in LeaveOneOut().split(X):
        Xtr,Xte=X[tr].copy(),X[te].copy()
        if resid:
            lr=LinearRegression().fit(logD[tr],Xtr); Xtr-=lr.predict(logD[tr]); Xte-=lr.predict(logD[te])
        sc=StandardScaler().fit(Xtr); Xtr,Xte=sc.transform(Xtr),sc.transform(Xte)
        for k,f in learners(seed).items(): pred[k][te[0]]=fit(f(),Xtr,y[tr]).predict(Xte)[0]
    return {k:int((v==y).sum()) for k,v in pred.items()}
out={}
for tag,resid in [("noselect_1092",False),("noselect_1092_resid",True)]:
    rs=[run(s,resid) for s in SEEDS]; out[tag]={}
    for k in rs[0]:
        ks=[r[k] for r in rs]; out[tag][k]={"correct":ks,"mean_acc":round(float(np.mean(ks))/n,4),"min_acc":round(min(ks)/n,4),"max_acc":round(max(ks)/n,4)}
        print(f"{tag:22s} {k:8s} mean={np.mean(ks)/n:.3f} range=[{min(ks)/n:.3f},{max(ks)/n:.3f}] {ks}",flush=True)
Path("results/noselect_1092.json").write_text(json.dumps(out,indent=2)); print("written")
