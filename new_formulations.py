"""Three alternative formulations that address the two causes of the accuracy drop
(selection-search overfitting on 54 subjects; recording-length confound handled by
discarding data) by design rather than by trimming.

A  Day-level training, subject-level decision. Every complete day is a training
   sample (1029 instead of 55); LOSO is grouped by subject; the held-out subject's
   label is the mean predicted probability over its days, thresholded at 0.5.
   Mean-probability aggregation is invariant to the number of days by construction.
B  Length-invariant per-subject vector. Order statistics whose expectation does not
   depend on D (p10, p25, p50, p75, p90, IQR, MAD, mean) instead of min/max, plus the
   24-hour mean activity profile and its normalised shape. No feature selection;
   a single pre-specified learner family (bagged trees) under LOSO.
C  Within-fold confound adjustment. Each feature is regressed on log(day count) on
   the training subjects only; residuals are used for train and test. Day count is
   then uninformative by construction; nothing is discarded.

Every variant is run over 5 seeds and reported as mean/min/max of LOSO accuracy.
All learners are fixed in advance (no post-hoc choice among them is used as a
headline): RandomForest(500), Bagging(200 trees), LogisticRegression(L2).
"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable

SEEDS = [0, 1, 2, 3, 4]
X_raw, y_day, g = load_depresjon("data/depresjon")
g = np.asarray(g); y_day = np.asarray(y_day)
F_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir="cache", tag="depresjon", n_jobs=-1)).to_numpy(np.float32)
ids = np.unique(g); y = np.array([int(y_day[g == s][0]) for s in ids]); n = len(ids)
D = np.array([int((g == s).sum()) for s in ids]); logD = np.log(D)
hour_day = X_raw.reshape(len(X_raw), 24, 60).mean(2)          # (1029, 24)

def learners(seed):
    return {
        "rf":      lambda: RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=seed),
        "bagging": lambda: BaggingClassifier(DecisionTreeClassifier(random_state=seed), n_estimators=200, n_jobs=1, random_state=seed),
        "logreg":  lambda: LogisticRegression(max_iter=3000, C=0.1),
    }

def wilson(k, m, z=1.959963985):
    p = k/m; d = 1+z*z/m; c = (p+z*z/(2*m))/d; h = z*np.sqrt(p*(1-p)/m+z*z/(4*m*m))/d
    return [round(float(c-h),3), round(float(c+h),3)]

def fit(clf, X, yy):
    try: return clf.fit(X, yy, sample_weight=compute_sample_weight("balanced", yy))
    except TypeError: return clf.fit(X, yy)

# ---------------------------------------------------------------- A -------
def run_A(feature_block, seed):
    preds = {k: np.zeros(n, int) for k in learners(seed)}
    for i, s in enumerate(ids):
        tr, te = g != s, g == s
        sc = StandardScaler().fit(feature_block[tr])
        Xtr, Xte = sc.transform(feature_block[tr]), sc.transform(feature_block[te])
        for name, fac in learners(seed).items():
            m = fit(fac(), Xtr, y_day[tr])
            preds[name][i] = int(m.predict_proba(Xte)[:, 1].mean() >= 0.5)
    return {k: int((v == y).sum()) for k, v in preds.items()}

# ---------------------------------------------------------------- B -------
def subject_vector(day_feats, day_hours, invariant=True):
    rows = []
    for s in ids:
        m = g == s; F = day_feats[m]; H = day_hours[m]
        if invariant:
            q = np.percentile(F, [10, 25, 50, 75, 90], axis=0)
            stats = [q[0], q[1], q[2], q[3], q[4], q[3]-q[1],
                     np.median(np.abs(F - q[2]), axis=0), F.mean(0)]
        else:  # the paper's original seven
            q = np.percentile(F, [25, 50, 75], axis=0)
            stats = [F.min(0), q[0], q[1], q[2], F.max(0), F.mean(0), F.std(0)]
        prof = H.mean(0)                                   # 24-h mean profile
        shape = prof / (prof.sum() + 1e-9)                 # normalised shape
        rows.append(np.concatenate(stats + [prof, shape]))
    return np.nan_to_num(np.array(rows, np.float32))

def run_subject(Xs, seed, residualize=False):
    preds = {k: np.zeros(n, int) for k in learners(seed)}
    for tr, te in LeaveOneOut().split(Xs):
        Xtr, Xte = Xs[tr].copy(), Xs[te].copy()
        if residualize:
            lr = LinearRegression().fit(logD[tr].reshape(-1, 1), Xtr)
            Xtr = Xtr - lr.predict(logD[tr].reshape(-1, 1))
            Xte = Xte - lr.predict(logD[te].reshape(-1, 1))
        sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        for name, fac in learners(seed).items():
            preds[name][te[0]] = fit(fac(), Xtr, y[tr]).predict(Xte)[0]
    return {k: int((v == y).sum()) for k, v in preds.items()}

def summarise(tag, per_seed):
    out = {}
    for name in per_seed[0]:
        ks = [r[name] for r in per_seed]
        out[name] = {"correct_per_seed": ks, "mean_acc": round(float(np.mean(ks))/n, 4),
                     "min_acc": round(min(ks)/n, 4), "max_acc": round(max(ks)/n, 4),
                     "wilson95_at_mean": wilson(float(np.mean(ks)), n)}
        print(f"  {tag:34s} {name:8s} mean={np.mean(ks)/n:.3f}  range=[{min(ks)/n:.3f},{max(ks)/n:.3f}]  correct={ks}", flush=True)
    return out

results = {"n": n, "seeds": SEEDS}
t0 = time.time()
print("A. day-level training, mean-probability decision")
results["A_daylevel_tsfel"]        = summarise("A tsfel156",          [run_A(F_day, s) for s in SEEDS])
results["A_daylevel_tsfel_hours"]  = summarise("A tsfel156+24h",      [run_A(np.hstack([F_day, hour_day]), s) for s in SEEDS])
print(f"  [{time.time()-t0:.0f}s]")

print("B. length-invariant subject vector, no selection")
XB = subject_vector(F_day, hour_day, invariant=True)
X7 = subject_vector(F_day, hour_day, invariant=False)
results["B_invariant_plus_profile"] = summarise("B invariant+24h", [run_subject(XB, s) for s in SEEDS])
results["B_original7_plus_profile"] = summarise("B original7+24h", [run_subject(X7, s) for s in SEEDS])
print(f"  [{time.time()-t0:.0f}s]")

print("C. within-fold residualisation on log(day count)")
results["C_invariant_resid"] = summarise("C invariant+24h resid", [run_subject(XB, s, residualize=True) for s in SEEDS])
results["C_original7_resid"] = summarise("C original7+24h resid", [run_subject(X7, s, residualize=True) for s in SEEDS])
print(f"  [{time.time()-t0:.0f}s]")

# reference: day-count stump and majority
results["reference"] = {"majority": round(float(max(np.bincount(y))/n), 4), "n_features_B": int(XB.shape[1]), "n_features_7": int(X7.shape[1])}
Path("results/new_formulations.json").write_text(json.dumps(results, indent=2))
print("written results/new_formulations.json")
