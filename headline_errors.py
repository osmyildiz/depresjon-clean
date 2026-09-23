"""Which subjects the headline configuration gets wrong.

Runs Info Gain + Bagging on the TSFEL-only representation under LOSO and prints
the misclassified subjects with their diagnosis, MADRS score and recording
length. Reproduces the error list in Section 5.4.

    python headline_errors.py
"""
import json
import numpy as np, pandas as pd
from scipy import stats as ss
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject
from nested_loso import _score_selector,_indices_for_k,fresh_classifiers,_fit_predict
from confound_checks import inner_pick_k, SELECTOR, CLASSIFIER, SEED

X_raw,y_day,g = load_depresjon('data/depresjon')
fd = drop_unstable(extract_tsfel(X_raw,g,cache_dir='cache',tag='depresjon',n_jobs=-1))
feats,y,ids = aggregate_per_subject(fd,g,y_day,raw_activity=None)
X=feats.to_numpy(dtype=np.float32); y=np.asarray(y); ids=list(ids)
g=np.asarray(g); days=pd.Series(g).value_counts().reindex(ids).to_numpy()

preds=np.zeros(len(y),int)
for tr,te in LeaveOneOut().split(X):
    k=inner_pick_k(X[tr],y[tr],SEED)
    sc=StandardScaler().fit(X[tr]); A,B=sc.transform(X[tr]),sc.transform(X[te])
    s,extra=_score_selector(SELECTOR,A,y[tr],SEED)
    Ai=extra(A) if extra else A; Bi=extra(B) if extra else B
    idx=_indices_for_k(SELECTOR,s,k)
    lab,_=_fit_predict(fresh_classifiers(SEED)[CLASSIFIER],Ai[:,idx],y[tr],Bi[:,idx],
                       compute_sample_weight('balanced',y[tr]))
    preds[te[0]]=lab[0]
print(f'headline accuracy {(preds==y).mean():.4f} ({(preds==y).sum()}/{len(y)})')

sc_df=pd.read_csv('data/depresjon/scores.csv').set_index('number')
AFF={1:'bipolar II',2:'unipolar',3:'bipolar I'}
err=[]
for i,sid in enumerate(ids):
    if preds[i]==y[i]: continue
    r=sc_df.loc[sid] if sid in sc_df.index else None
    err.append({'subject':sid,'true':int(y[i]),'pred':int(preds[i]),'valid_days':int(days[i]),
        'afftype':AFF.get(int(r.afftype)) if r is not None and pd.notna(r.afftype) else None,
        'madrs1':None if r is None or pd.isna(r.madrs1) else float(r.madrs1)})
print(json.dumps(err,indent=2))

# recording-length signature of the errors
ok=np.array([preds[i]==y[i] for i in range(len(y))])
for cls,name in [(1,'depressed'),(0,'control')]:
    m=y==cls
    d_ok=days[m&ok]; d_err=days[m&~ok]
    print(f'{name}: correct n={len(d_ok)} median days={np.median(d_ok):.0f} | '
          f'errors n={len(d_err)} median days={np.median(d_err):.0f}')
u,p=ss.mannwhitneyu(days[ok],days[~ok])
print(f'day count, correct vs misclassified: p={p:.4f}')
from pathlib import Path
Path('results').mkdir(exist_ok=True)
json.dump(err, open('results/headline_errors.json','w'), indent=2)
