"""Does the selection-free 1092 Bagging result separate from recording length?

1. Re-run 1092-only Bagging (5 seeds) KEEPING per-subject predictions; exact McNemar
   against the pre-specified day-count stump; overlap of error sets.
2. Three-feature nuisance baseline: day count + mean zero-fraction + mean activity,
   LOSO stump and L2 logistic regression.
3. Accuracy of the same predictions inside the stratum of subjects with 13-14
   recorded days, where recording length is uninformative by construction.
"""
import sys, json, os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(k,"1")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as ss
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject

SEEDS=[0,1,2,3,4]
X_raw,y_day,g=load_depresjon("data/depresjon"); g=np.asarray(g); y_day=np.asarray(y_day)
fd=drop_unstable(extract_tsfel(X_raw,g,cache_dir="cache",tag="depresjon",n_jobs=-1))
feats,y,ids=aggregate_per_subject(fd,g,y_day,raw_activity=None); y=np.asarray(y); ids=list(ids); n=len(y)
X=np.nan_to_num(feats.to_numpy(np.float32))
D=np.array([int((g==s).sum()) for s in ids],float)
zf=np.array([(X_raw[g==s]==0).mean() for s in ids]); ma=np.array([X_raw[g==s].mean() for s in ids])
sw=lambda yy: compute_sample_weight("balanced",yy)

def loso_pred(Xm, make):
    p=np.zeros(n,int)
    for tr,te in LeaveOneOut().split(Xm):
        sc=StandardScaler().fit(Xm[tr]); m=make()
        try: m.fit(sc.transform(Xm[tr]),y[tr],sample_weight=sw(y[tr]))
        except TypeError: m.fit(sc.transform(Xm[tr]),y[tr])
        p[te[0]]=m.predict(sc.transform(Xm[te]))[0]
    return p

def mcnemar(pa,pb):
    a=(pa==y)&(pb!=y); b=(pb==y)&(pa!=y); k=int(a.sum()); m=int(b.sum())
    p=float(ss.binomtest(k,k+m,0.5).pvalue) if k+m>0 else 1.0
    return {"A_right_B_wrong":k,"B_right_A_wrong":m,"exact_p":round(p,4)}

out={"n":n}
# stump on day count (pre-specified nuisance baseline)
stump=loso_pred(D.reshape(-1,1), lambda: DecisionTreeClassifier(max_depth=1,random_state=0))
out["stump_daycount"]={"correct":int((stump==y).sum()),"errors":[ids[i] for i in np.flatnonzero(stump!=y)]}
# 3-feature nuisance baseline
X3=np.column_stack([D,zf,ma])
for name,make in [("stump",lambda: DecisionTreeClassifier(max_depth=1,random_state=0)),
                  ("tree_depth2",lambda: DecisionTreeClassifier(max_depth=2,random_state=0)),
                  ("logreg",lambda: LogisticRegression(max_iter=3000))]:
    p=loso_pred(X3,make); out[f"nuisance3_{name}"]={"correct":int((p==y).sum()),"accuracy":round(float((p==y).mean()),4)}
    print(f"nuisance3 {name:12s} {int((p==y).sum())}/{n}",flush=True)
# 1092 bagging, 5 seeds, with predictions
strat=np.flatnonzero((D>=13)&(D<=14)); out["stratum_13_14"]={"n":int(len(strat)),"depressed":int(y[strat].sum()),"control":int(len(strat)-y[strat].sum())}
res=[]
for s in SEEDS:
    p=loso_pred(X, lambda: BaggingClassifier(DecisionTreeClassifier(random_state=s),n_estimators=200,n_jobs=1,random_state=s))
    r={"seed":s,"correct":int((p==y).sum()),"errors":[ids[i] for i in np.flatnonzero(p!=y)],
       "mcnemar_vs_stump":mcnemar(p,stump),
       "error_overlap_with_stump":int(((p!=y)&(stump!=y)).sum()),
       "stratum_13_14_correct":int((p[strat]==y[strat]).sum()),
       "stratum_13_14_acc":round(float((p[strat]==y[strat]).mean()),4),
       "stratum_13_14_sens":round(float((p[strat][y[strat]==1]==1).mean()),4),
       "stratum_13_14_spec":round(float((p[strat][y[strat]==0]==0).mean()),4)}
    res.append(r); print(f"seed {s}: {r['correct']}/{n}  McNemar vs stump {r['mcnemar_vs_stump']}  stratum13-14 {r['stratum_13_14_correct']}/{len(strat)}={r['stratum_13_14_acc']:.3f} (sens {r['stratum_13_14_sens']:.2f} spec {r['stratum_13_14_spec']:.2f})  errors={r['errors']}",flush=True)
out["bagging_1092_seeds"]=res
out["stratum_majority"]=round(float(max(y[strat].mean(),1-y[strat].mean())),4)
Path("results/stratum_mcnemar.json").write_text(json.dumps(out,indent=2)); print("written")
