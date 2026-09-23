"""Baselines and checks requested by the independent panel (revision/PANEL_REVIEW.md).

1c  Pre-specified recording-length baseline, one method fixed in advance, in every
    cohort the paper reports a model on: model-free AUC of day count, and a single
    decision stump under LOSO. No learner is chosen after seeing results.
1d  ZCR <-> bout-count relationship (ZCR on non-negative counts should be ~2 x bouts).
1e  Jakobsen et al. (2020) three-feature baseline under their own protocol
    (per-day features, leave-one-user-out, majority vote per subject), and a
    per-subject variant of the same three features; plus Bagging on all 1092
    features with no selection, under plain LOSO.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as ss
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject

X_raw, y_day, g = load_depresjon("data/depresjon")
g = np.asarray(g); y_day = np.asarray(y_day)
feats_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir="cache", tag="depresjon", n_jobs=-1))
out = {}

def wilson(k, n, z=1.959963985):
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h), float(c+h)]

def subject_table(mask):
    ids = np.unique(g[mask]); y = np.array([int(y_day[g == s][0]) for s in ids])
    days = np.array([int(((g == s) & mask).sum()) for s in ids])
    return ids, y, days

def stump_loso(x, y):
    x = x.reshape(-1, 1); pred = np.zeros(len(y), int)
    for tr, te in LeaveOneOut().split(x):
        m = DecisionTreeClassifier(max_depth=1, random_state=0).fit(x[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        pred[te[0]] = m.predict(x[te])[0]
    return pred

# ---- 1c: recording-length baseline per cohort -------------------------------
cohorts = {}
cohorts["full"] = np.ones(len(g), bool)
c = pd.Series(g).value_counts()
cohorts["overlap_13_28"] = np.isin(g, [s for s in c.index if 13 <= c[s] <= 28])
out["daycount_baseline"] = {}
for name, mask in cohorts.items():
    ids, y, days = subject_table(mask)
    auc = float(roc_auc_score(y, -days))            # fewer days -> depressed
    pred = stump_loso(days.astype(float), y)
    k = int((pred == y).sum()); n = len(y)
    out["daycount_baseline"][name] = {
        "n": n, "majority": float(max(np.bincount(y)) / n),
        "auc_model_free_fewer_days_is_depressed": auc,
        "stump_loso_accuracy": k / n, "stump_correct": k, "wilson95": wilson(k, n)}
    print(f"[1c] {name:14s} n={n} AUC(day count)={auc:.3f}  stump LOSO={k}/{n}={k/n:.3f}")

# ---- 1d: ZCR vs bout count --------------------------------------------------
zcr_day = feats_day["0_Zero crossing rate"].to_numpy()
bouts_day = np.array([int(np.sum(np.diff(np.concatenate([[0], (d > 0).astype(int), [0]])) == 1)) for d in X_raw])
r_day = float(np.corrcoef(zcr_day, bouts_day)[0, 1])
ratio = float(np.median(zcr_day / np.maximum(bouts_day, 1)))
ids, y, _ = subject_table(cohorts["full"])
zcr_mean = np.array([zcr_day[g == s].mean() for s in ids])
bout_mean = np.array([bouts_day[g == s].mean() for s in ids])
r_subj = float(np.corrcoef(zcr_mean, bout_mean)[0, 1])
out["zcr_vs_bout_count"] = {"pearson_day_level": r_day, "median_zcr_over_bouts_day_level": ratio,
                            "pearson_subject_mean": r_subj,
                            "spearman_subject_mean": float(ss.spearmanr(zcr_mean, bout_mean).statistic)}
print(f"[1d] ZCR vs bout count: day-level r={r_day:.4f}, median ZCR/bouts={ratio:.3f}, subject-mean r={r_subj:.4f}")

# ---- 1e: Jakobsen 3-feature baseline, their protocol ------------------------
F3 = np.column_stack([X_raw.mean(1), X_raw.std(1), (X_raw == 0).mean(1)])
def jakobsen_loso(F, clf_factory):
    pred = np.zeros(len(ids), int)
    for i, s in enumerate(ids):
        tr = g != s; te = g == s
        sc = StandardScaler().fit(F[tr])
        m = clf_factory().fit(sc.transform(F[tr]), y_day[tr], sample_weight=compute_sample_weight("balanced", y_day[tr]))
        pred[i] = int(np.round(m.predict(sc.transform(F[te])).mean()))   # majority vote over days
    return pred
res = {}
for name, fac in [("logreg", lambda: LogisticRegression(max_iter=2000)),
                  ("bagging", lambda: BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=100, random_state=0, n_jobs=1))]:
    p = jakobsen_loso(F3, fac); k = int((p == y).sum())
    res[f"per_day_majority_vote_{name}"] = {"correct": k, "n": len(y), "accuracy": k/len(y), "wilson95": wilson(k, len(y))}
    print(f"[1e] Jakobsen-3 per-day+vote {name:8s}: {k}/{len(y)}={k/len(y):.3f}")
# per-subject mean of the 3 features
S3 = np.array([F3[g == s].mean(0) for s in ids])
pred = np.zeros(len(y), int)
for tr, te in LeaveOneOut().split(S3):
    sc = StandardScaler().fit(S3[tr])
    m = LogisticRegression(max_iter=2000).fit(sc.transform(S3[tr]), y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    pred[te[0]] = m.predict(sc.transform(S3[te]))[0]
k = int((pred == y).sum()); res["per_subject_mean_logreg"] = {"correct": k, "n": len(y), "accuracy": k/len(y), "wilson95": wilson(k, len(y))}
print(f"[1e] Jakobsen-3 per-subject mean logreg: {k}/{len(y)}={k/len(y):.3f}")
out["jakobsen_three_feature_baseline"] = res

# ---- 1e: Bagging on all 1092, no selection ----------------------------------
feats, y2, ids2 = aggregate_per_subject(feats_day, g, y_day, raw_activity=None)
assert list(ids2) == list(ids)
Xs = feats.to_numpy(np.float32); pred = np.zeros(len(y), int)
for tr, te in LeaveOneOut().split(Xs):
    sc = StandardScaler().fit(Xs[tr])
    m = BaggingClassifier(DecisionTreeClassifier(random_state=0), n_estimators=100, random_state=0, n_jobs=1)
    m.fit(sc.transform(Xs[tr]), y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    pred[te[0]] = m.predict(sc.transform(Xs[te]))[0]
k = int((pred == y).sum()); out["bagging_no_selection_1092"] = {"correct": k, "n": len(y), "accuracy": k/len(y), "wilson95": wilson(k, len(y))}
print(f"[1e] Bagging, all 1092, no selection: {k}/{len(y)}={k/len(y):.3f}")

Path("results/panel_baselines.json").write_text(json.dumps(out, indent=2))
print("written results/panel_baselines.json")
