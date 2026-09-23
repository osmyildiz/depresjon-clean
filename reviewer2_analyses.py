"""Analyses requested by Reviewer 2 (review report PDF).

Major 3  unipolar vs bipolar subgroup sensitivity
Major 4  redundancy / collinearity among the top selected features
Major 5  empirical characterisation of the Zero Crossing Rate on non-negative counts
Major 6  zero-minute structure within valid days (non-wear cannot be distinguished)
Minor 2  identity, MADRS score and recording length of the misclassified subjects
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as ss
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject

X_raw, y_day, g = load_depresjon("data/depresjon")
feats_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir="cache", tag="depresjon", n_jobs=-1))
feats, y, ids = aggregate_per_subject(feats_day, g, y_day, raw_activity=None)
g = np.asarray(g); y = np.asarray(y); ids = list(ids)
days = pd.Series(g).value_counts().reindex(ids).to_numpy()

sc = pd.read_csv("data/depresjon/scores.csv").set_index("number")
AFF = {1: "bipolar II", 2: "unipolar", 3: "bipolar I"}
out = {}

def wilson(k, n, z=1.959963985):
    if n == 0: return (float('nan'),)*2
    p = k/n; d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return float(c-h), float(c+h)

# ---------------- Major 3 + Minor 2 : needs per-subject predictions ----------
pf = pd.read_csv("results/nested/tsfel_only/per_fold_predictions.csv")
pf = pf[pf.metric == "balanced_accuracy"].sort_values("subject_index")
assert len(pf) == len(ids)
pred = pf.y_pred.to_numpy()

rows = []
for i, sid in enumerate(ids):
    rec = {"subject": sid, "true": int(y[i]), "pred": int(pred[i]),
           "correct": bool(y[i] == pred[i]), "valid_days": int(days[i])}
    if sid in sc.index:
        r = sc.loc[sid]
        rec["afftype"] = AFF.get(int(r.afftype), None) if pd.notna(r.afftype) else None
        rec["madrs1"] = None if pd.isna(r.madrs1) else float(r.madrs1)
        rec["madrs2"] = None if pd.isna(r.madrs2) else float(r.madrs2)
        rec["melancholic"] = None if pd.isna(r.melanch) else int(r.melanch) == 1
    rows.append(rec)
res = pd.DataFrame(rows)
res.to_csv("results/reviewer2_per_subject.csv", index=False)

dep = res[res.true == 1]
sub = {}
for grp, name in [(dep.afftype == "unipolar", "unipolar"),
                  (dep.afftype.isin(["bipolar I", "bipolar II"]), "bipolar")]:
    d = dep[grp]
    k, n = int(d.correct.sum()), len(d)
    lo, hi = wilson(k, n)
    sub[name] = {"n": n, "correctly_detected": k, "sensitivity": k/n,
                 "wilson95": [lo, hi],
                 "madrs1_median": float(d.madrs1.median()),
                 "valid_days_median": float(d.valid_days.median())}
tab = [[sub["unipolar"]["correctly_detected"], sub["unipolar"]["n"]-sub["unipolar"]["correctly_detected"]],
       [sub["bipolar"]["correctly_detected"],  sub["bipolar"]["n"] -sub["bipolar"]["correctly_detected"]]]
sub["fisher_p"] = float(ss.fisher_exact(tab)[1])
out["major3_subgroup_sensitivity"] = sub

miss = res[~res.correct]
out["minor2_misclassified"] = miss.to_dict("records")

# ---------------- Major 4 : redundancy among top features --------------------
top = ["p25_0_Zero crossing rate", "p50_0_Zero crossing rate",
       "p75_0_Zero crossing rate", "mean_0_Zero crossing rate",
       "min_0_Zero crossing rate", "circ_bout_count_mean"]
avail = [c for c in top if c in feats.columns]
M = feats[avail].to_numpy(dtype=float)
C = np.corrcoef(M.T)
out["major4_collinearity"] = {
    "features": avail,
    "pearson_matrix": [[round(float(v), 3) for v in row] for row in C],
    "max_offdiag": round(float(np.max(np.abs(C - np.eye(len(avail))))), 3),
    "mean_abs_offdiag": round(float(np.abs(C[~np.eye(len(avail), dtype=bool)]).mean()), 3),
}

# ---------------- Major 5 + 6 : zero-minute structure ------------------------
zero_frac, longest_zero, day_zero_runs = [], [], []
for sid in ids:
    m = g == sid
    days_x = X_raw[m]
    zf, lz = [], []
    for d in days_x:
        z = d == 0
        zf.append(z.mean())
        # longest run of zeros
        best = cur = 0
        for v in z:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        lz.append(best)
        day_zero_runs.append(best)
    zero_frac.append(float(np.mean(zf)))
    longest_zero.append(float(np.mean(lz)))
zero_frac = np.array(zero_frac); longest_zero = np.array(longest_zero)

def cmp(v, name):
    u, p = ss.mannwhitneyu(v[y == 1], v[y == 0])
    return {"control_median": float(np.median(v[y == 0])),
            "depressed_median": float(np.median(v[y == 1])),
            "mannwhitney_p": float(p)}
out["major5_major6_zero_structure"] = {
    "zero_crossing_definition": ("TSFEL zero_cross = len(np.where(np.diff(np.sign(signal)))[0]); "
        "on non-negative activity counts np.sign takes only the values 0 and 1, so the feature counts "
        "transitions between zero-count minutes and non-zero-count minutes. The implicit threshold is "
        "activity count > 0, with no epsilon and no smoothing."),
    "mean_fraction_of_zero_minutes_per_day": cmp(zero_frac, "zero_frac"),
    "mean_longest_daily_zero_run_minutes": cmp(longest_zero, "longest_zero"),
    "days_with_zero_run_over_360min": int(np.sum(np.array(day_zero_runs) > 360)),
    "days_total": int(len(day_zero_runs)),
    "non_wear_handling": ("Days with fewer than 1440 timestamped minutes are discarded. Within a retained "
        "day no non-wear detection is applied: the Depresjon release carries no wear sensor or non-wear flag, "
        "and no Choi/Troiano-style algorithm was used. Zero-count minutes are therefore treated as inactivity "
        "regardless of whether the device was worn."),
}

Path("results/reviewer2_analyses.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
