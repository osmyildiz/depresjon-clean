"""Paired comparison of selection-free 1092 Bagging against the strongest nuisance
baseline (depth-2 tree on day count + zero-fraction + mean activity), plus the
nuisance tree's own accuracy inside the 13-14-day stratum."""
import sys, json, os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ.setdefault(k,"1")
from pathlib import Path
import numpy as np
from scipy import stats as ss
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject
X_raw,y_day,g=load_depresjon("data/depresjon"); g=np.asarray(g); y_day=np.asarray(y_day)
fd=drop_unstable(extract_tsfel(X_raw,g,cache_dir="cache",tag="depresjon",n_jobs=-1))
feats,y,ids=aggregate_per_subject(fd,g,y_day,raw_activity=None); y=np.asarray(y); ids=list(ids); n=len(y)
X=np.nan_to_num(feats.to_numpy(np.float32))
D=np.array([int((g==s).sum()) for s in ids],float)
X3=np.column_stack([D,[(X_raw[g==s]==0).mean() for s in ids],[X_raw[g==s].mean() for s in ids]])
def loso(Xm,make):
    p=np.zeros(n,int)
    for tr,te in LeaveOneOut().split(Xm):
        sc=StandardScaler().fit(Xm[tr]); m=make().fit(sc.transform(Xm[tr]),y[tr],sample_weight=compute_sample_weight("balanced",y[tr]))
        p[te[0]]=m.predict(sc.transform(Xm[te]))[0]
    return p
def mcn(pa,pb):
    a=int(((pa==y)&(pb!=y)).sum()); b=int(((pb==y)&(pa!=y)).sum())
    return {"A_right_B_wrong":a,"B_right_A_wrong":b,"exact_p":round(float(ss.binomtest(a,a+b,0.5).pvalue),4) if a+b else 1.0}
nuis=loso(X3,lambda: DecisionTreeClassifier(max_depth=2,random_state=0))
bag =loso(X,lambda: BaggingClassifier(DecisionTreeClassifier(random_state=0),n_estimators=200,n_jobs=1,random_state=0))
strat=np.flatnonzero((D>=13)&(D<=14))
def wilson(k,m,z=1.959963985):
    p=k/m; d=1+z*z/m; c=(p+z*z/(2*m))/d; h=z*np.sqrt(p*(1-p)/m+z*z/(4*m*m))/d; return [round(float(c-h),3),round(float(c+h),3)]
out={"bagging_1092_correct":int((bag==y).sum()),"nuisance_depth2_correct":int((nuis==y).sum()),
     "mcnemar_bagging_vs_nuisance":mcn(bag,nuis),
     "stratum_13_14":{"n":int(len(strat)),"majority":round(float(max(y[strat].mean(),1-y[strat].mean())),3),
        "bagging_correct":int((bag[strat]==y[strat]).sum()),"bagging_wilson":wilson(int((bag[strat]==y[strat]).sum()),len(strat)),
        "nuisance_correct":int((nuis[strat]==y[strat]).sum()),
        "mcnemar_in_stratum":None},   # filled in below with stratum-restricted labels
     "nuisance_errors":[ids[i] for i in np.flatnonzero(nuis!=y)]}
# in-stratum McNemar needs stratum-restricted y; recompute properly
ys=y[strat]; a=int(((bag[strat]==ys)&(nuis[strat]!=ys)).sum()); b=int(((nuis[strat]==ys)&(bag[strat]!=ys)).sum())
out["stratum_13_14"]["mcnemar_in_stratum"]={"A_right_B_wrong":a,"B_right_A_wrong":b,"exact_p":round(float(ss.binomtest(a,a+b,0.5).pvalue),4) if a+b else 1.0}
Path("results/mcnemar_nuisance.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2)); print("written")
