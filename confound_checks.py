"""Recording-length confound checks for the headline configuration.

Section 5.5 of the submitted manuscript raises two concerns and defers both
to future work:

  * per-subject recording length ranges from 12 to 45 days, and the min/max
    order statistics scale with it by construction;
  * no formal test of the day-count effect was reported.

daycount_confound.py shows the concern is real: depressed subjects were
recorded for significantly fewer days (Mann-Whitney p = 0.0021), and
recording length alone classifies the two groups at 0.78 LOSO accuracy.
This script quantifies how much of the headline result survives once the
confound is removed, under three interventions:

  full          all 1092 TSFEL distributional features (headline setting)
  no_minmax     drop the min and max order statistics (5 stats x 156 = 780)
  first_N       truncate every subject to their first N chronological days
                before aggregation, equalising recording length across
                subjects (N = 12, the shortest recording in the cohort)
  first_N_nomm  both interventions together

Everything else -- selector, classifier, inner k tuning, sample weighting,
LOSO -- is held at the headline pipeline.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject
from nested_loso import (_score_selector, _indices_for_k, fresh_classifiers,
                         _fit_predict, wilson_interval, K_GRID, INNER_FOLDS)

SELECTOR, CLASSIFIER, SEED = "info_gain", "Bagging", 0


def inner_pick_k(X_tr, y_tr, seed):
    """Reproduces src.experiment_subject._inner_select_k, scores cached per fold.

    The proxy learner (balanced ExtraTrees) and the plain-accuracy criterion
    are those of the submitted pipeline; only the redundant re-scoring of the
    selector at every k is removed.
    """
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    acc = {k: [] for k in K_GRID}
    for a, b in inner.split(X_tr, y_tr):
        Xa, ya, Xb, yb = X_tr[a], y_tr[a], X_tr[b], y_tr[b]
        sc = StandardScaler().fit(Xa)
        Xa_s, Xb_s = sc.transform(Xa), sc.transform(Xb)
        scores, extra = _score_selector(SELECTOR, Xa_s, ya, seed)
        Xa_in = extra(Xa_s) if extra else Xa_s
        Xb_in = extra(Xb_s) if extra else Xb_s
        for k in K_GRID:
            if k > X_tr.shape[1]:
                continue
            idx = _indices_for_k(SELECTOR, scores, k)
            cheap = ExtraTreesClassifier(n_estimators=100, class_weight="balanced",
                                         n_jobs=1, random_state=seed)
            cheap.fit(Xa_in[:, idx], ya)
            acc[k].append(accuracy_score(yb, cheap.predict(Xb_in[:, idx])))
    best_k, best = K_GRID[0], -np.inf
    for k in K_GRID:
        if not acc[k]:
            continue
        m = float(np.mean(acc[k]))
        if m > best:
            best, best_k = m, k
    return best_k


def loso(X, y, seed=SEED):
    preds = np.zeros(len(y), int)
    ks = []
    for tr, te in LeaveOneOut().split(X):
        X_tr, y_tr = X[tr], y[tr]
        k = inner_pick_k(X_tr, y_tr, seed)
        ks.append(k)
        sc = StandardScaler().fit(X_tr)
        Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X[te])
        scores, extra = _score_selector(SELECTOR, Xtr_s, y_tr, seed)
        Xtr_in = extra(Xtr_s) if extra else Xtr_s
        Xte_in = extra(Xte_s) if extra else Xte_s
        idx = _indices_for_k(SELECTOR, scores, k)
        sw = compute_sample_weight("balanced", y_tr)
        lab, _ = _fit_predict(fresh_classifiers(seed)[CLASSIFIER],
                              Xtr_in[:, idx], y_tr, Xte_in[:, idx], sw)
        preds[te[0]] = lab[0]
    corr = int((preds == y).sum())
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    lo, hi = wilson_interval(corr, len(y))
    return {
        "accuracy": corr / len(y), "correct": corr, "n": len(y),
        "wilson95": [lo, hi],
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "kappa": float(cohen_kappa_score(y, preds)),
        "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "k_median": float(np.median(ks)),
        "k_iqr": [float(np.percentile(ks, 25)), float(np.percentile(ks, 75))],
        "n_features": int(X.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/depresjon")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--first-n", type=int, default=12)
    ap.add_argument("--out", default="results/confound_checks.json")
    args = ap.parse_args()

    X_raw, y_day, g = load_depresjon(args.data_dir)
    feats_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                                            tag="depresjon", n_jobs=-1))
    g = np.asarray(g)

    # groupby('date') in the loader yields days in chronological order, so
    # "first N days" is well defined per subject.
    keep = np.zeros(len(g), bool)
    for sid in np.unique(g):
        pos = np.flatnonzero(g == sid)[:args.first_n]
        keep[pos] = True

    variants = {}
    full, y, ids = aggregate_per_subject(feats_day, g, y_day, raw_activity=None)
    trunc, y_t, ids_t = aggregate_per_subject(
        feats_day.iloc[keep].reset_index(drop=True), g[keep],
        np.asarray(y_day)[keep], raw_activity=None)
    assert list(ids) == list(ids_t)
    y = np.asarray(y)

    mm = [c for c in full.columns if c.startswith(("min_", "max_"))]
    variants["full"] = full
    variants["no_minmax"] = full.drop(columns=mm)
    variants[f"first{args.first_n}"] = trunc
    variants[f"first{args.first_n}_no_minmax"] = trunc.drop(columns=mm)

    out = {"selector": SELECTOR, "classifier": CLASSIFIER,
           "first_n_days": args.first_n, "variants": {}}
    for name, F in variants.items():
        t = time.time()
        r = loso(F.to_numpy(dtype=np.float32), y)
        r["seconds"] = round(time.time() - t, 1)
        out["variants"][name] = r
        print(f"{name:22s} acc={r['accuracy']:.4f} ({r['correct']}/{r['n']})  "
              f"bal={r['balanced_accuracy']:.4f}  kappa={r['kappa']:.3f}  "
              f"k~{r['k_median']:.0f}  p={r['n_features']}  [{r['seconds']}s]",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
