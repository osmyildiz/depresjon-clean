"""Are recording length (D) and non-wear one confound?

The day-count correlation of per-subject features has a quantile gradient
(min |rho| 0.76, p25 0.64, p75 0.06, max 0.25). Extreme-value scaling would
load min AND max symmetrically; a gradient loading the LOW tail is what you get
when long recordings accumulate low-activity (non-wear-like) days. Test: count
per subject the days with a >6 h contiguous zero block (N_nw); relate N_nw to D
and to group; then compute, per order-statistic family, the Spearman correlation
of each feature with D, and the partial Spearman controlling N_nw.
"""
import sys, json, os
os.environ.setdefault("OMP_NUM_THREADS","1")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as ss
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject

X_raw, y_day, g = load_depresjon("data/depresjon"); g=np.asarray(g); y_day=np.asarray(y_day)
fd = drop_unstable(extract_tsfel(X_raw, g, cache_dir="cache", tag="depresjon", n_jobs=-1))
feats, y, ids = aggregate_per_subject(fd, g, y_day, raw_activity=None); y=np.asarray(y); ids=list(ids)

def longest_zero(day):
    best=cur=0
    for v in (day==0):
        cur=cur+1 if v else 0; best=max(best,cur)
    return best
lz = np.array([longest_zero(d) for d in X_raw])
nw_day = lz > 360
D    = np.array([int((g==s).sum()) for s in ids])
Nnw  = np.array([int(nw_day[g==s].sum()) for s in ids])
Fnw  = Nnw / D
zero_frac = np.array([(X_raw[g==s]==0).mean() for s in ids])

def partial_spearman(x, yv, z):
    rx = ss.rankdata(x); ry = ss.rankdata(yv); rz = ss.rankdata(z)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b]); return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0,1])

out = {"n": len(y)}
out["N_nw_by_group"] = {name: {"subjects": int((y==c).sum()),
        "days_with_gt6h_zero": int(Nnw[y==c].sum()), "days_total": int(D[y==c].sum()),
        "frac_days": float(Nnw[y==c].sum()/D[y==c].sum()),
        "subjects_with_ge1": int((Nnw[y==c]>0).sum()),
        "median_N_nw": float(np.median(Nnw[y==c])), "median_frac": float(np.median(Fnw[y==c]))}
        for c,name in [(0,"control"),(1,"depressed")]}
out["N_nw_vs_D"] = {"spearman": float(ss.spearmanr(Nnw, D).statistic), "p": float(ss.spearmanr(Nnw, D).pvalue)}
out["frac_nw_vs_group_mannwhitney_p"] = float(ss.mannwhitneyu(Fnw[y==1], Fnw[y==0]).pvalue)
out["N_nw_vs_group_mannwhitney_p"] = float(ss.mannwhitneyu(Nnw[y==1], Nnw[y==0]).pvalue)
out["D_vs_group_mannwhitney_p"] = float(ss.mannwhitneyu(D[y==1], D[y==0]).pvalue)
# partial: does D still separate groups once N_nw is controlled? (rank-based logistic-ish: partial spearman with label)
out["D_vs_label_spearman"] = float(ss.spearmanr(D, y).statistic)
out["D_vs_label_partial_given_Nnw"] = partial_spearman(D, y, Nnw)
out["Nnw_vs_label_partial_given_D"] = partial_spearman(Nnw, y, D)
out["zero_frac_vs_label_spearman"] = float(ss.spearmanr(zero_frac, y).statistic)

M = feats.to_numpy(np.float64); cols = list(feats.columns)
fam = {}
for stat in ["min","p25","p50","p75","max","mean","std"]:
    idx = [i for i,c in enumerate(cols) if c.startswith(stat+"_")]
    r_D, r_D_given_nw, r_nw, r_nw_given_D = [], [], [], []
    for i in idx:
        v = M[:,i]
        if not np.all(np.isfinite(v)) or np.std(v)==0: continue
        r_D.append(abs(ss.spearmanr(v, D).statistic))
        r_D_given_nw.append(abs(partial_spearman(v, D, Nnw)))
        r_nw.append(abs(ss.spearmanr(v, Nnw).statistic))
        r_nw_given_D.append(abs(partial_spearman(v, Nnw, D)))
    fam[stat] = {"n_features": len(r_D),
                 "median_abs_rho_with_D": round(float(np.median(r_D)),3),
                 "median_abs_partial_rho_with_D_given_Nnw": round(float(np.median(r_D_given_nw)),3),
                 "median_abs_rho_with_Nnw": round(float(np.median(r_nw)),3),
                 "median_abs_partial_rho_with_Nnw_given_D": round(float(np.median(r_nw_given_D)),3)}
out["quantile_gradient"] = fam
Path("results/one_confound.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
