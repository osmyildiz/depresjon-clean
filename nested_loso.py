"""Fully nested leave-one-subject-out evaluation.

Reviewer comment (Major 4): the headline configuration in the submitted
manuscript was chosen by ranking all 105 (selector, classifier) pairs on
their LOSO test performance, with the inner cross-validation tuning only
the number of retained features k. That reporting scheme is optimistic:
the outer test subjects participated, through the ranking step, in the
choice of selector and classifier.

This script removes that optimism. In every outer LOSO fold the *entire*
configuration -- feature selector, classifier, and k -- is chosen by inner
cross-validation on the 54 training subjects alone. The held-out subject
is touched exactly once, by the single pipeline that the inner search
selected. No quantity derived from the held-out subject enters model
construction, including the choice of configuration.

Search space per outer fold:
    7 selectors x 15 classifiers x 8 values of k = 840 configurations.

Inner protocol:
    Stratified 5-fold CV on the 54 training subjects (matching
    SubjectExperimentConfig.inner_folds=5 used for the submitted results).
    Predictions from the 5 inner folds are pooled and the selection metric
    is computed once on the pooled predictions.

Selection metric:
    Reported for three pre-specified choices, run independently:
      balanced_accuracy (primary)  -- robust to the 23:32 class ratio
      accuracy                     -- the metric used by the submitted
                                      inner loop, for continuity
      roc_auc                      -- threshold-free alternative
    Ties are broken deterministically: smaller k first, then selector
    order as listed in SELECTOR_ORDER, then classifier order as listed in
    CLASSIFIER_ORDER.

Usage:
    python nested_loso.py --data-dir data/depresjon --cache-dir cache \
        --feature-set tsfel_only --out-dir results/nested
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))

from src.aggregation import aggregate_per_subject
from src.data import load_depresjon
from src.experiment_subject import build_classifiers
from src.features import drop_unstable, extract_tsfel
from src.selectors import SELECTORS

SELECTOR_ORDER = (
    "lasso", "ridge", "elastic_net", "kbest_chi2",
    "fisher", "info_gain", "rf_importance",
)
CLASSIFIER_ORDER = (
    "RandomForest", "ExtraTrees", "Bagging", "SVM", "XGBoost", "AdaBoost",
    "KNN", "MLP", "QDA", "GaussianNB", "LogReg_L1", "LogReg_L2",
    "LogReg_EN", "HistGB", "Stacking",
)
K_GRID = (10, 20, 30, 50, 70, 100, 150, 200)
INNER_FOLDS = 5
METRICS = ("balanced_accuracy", "accuracy", "roc_auc")


# --------------------------------------------------------------------------
# Selector scoring, computed once per (fold, selector) instead of once per k.
#
# Every scorer below reproduces the corresponding class in src/selectors.py
# exactly; `verify_selector_equivalence` asserts index-for-index agreement
# before any experiment is run.
# --------------------------------------------------------------------------

def _score_selector(name: str, X: np.ndarray, y: np.ndarray, random_state: int):
    """Return (scores, transform) for a selector fitted on (X, y).

    `transform` maps a raw standardized matrix to the selector's own input
    space (identity for all selectors except chi2, which MinMax-scales).
    """
    if name == "lasso":
        m = Lasso(alpha=0.01, max_iter=5000, random_state=random_state).fit(X, y)
        return np.abs(m.coef_), None
    if name == "ridge":
        m = Ridge(alpha=1.0, random_state=random_state).fit(X, y)
        return np.abs(m.coef_), None
    if name == "elastic_net":
        m = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000,
                       random_state=random_state).fit(X, y)
        return np.abs(m.coef_), None
    if name == "kbest_chi2":
        scaler = MinMaxScaler().fit(X)
        scores = chi2(scaler.transform(X), y)[0]
        return np.nan_to_num(scores, nan=-np.inf), scaler.transform
    if name == "fisher":
        classes = np.unique(y)
        mu = X.mean(axis=0)
        num = np.zeros(X.shape[1])
        den = np.zeros(X.shape[1])
        for c in classes:
            mask = y == c
            num += mask.sum() * (X[mask].mean(axis=0) - mu) ** 2
            den += mask.sum() * (X[mask].var(axis=0) + 1e-12)
        return num / (den + 1e-12), None
    if name == "info_gain":
        return mutual_info_classif(X, y, random_state=random_state), None
    if name == "rf_importance":
        rf = RandomForestClassifier(n_estimators=200, n_jobs=1,
                                    random_state=random_state).fit(X, y)
        return rf.feature_importances_, None
    raise KeyError(name)


def _indices_for_k(name: str, scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices the original selector would retain at this k.

    src/selectors.py uses `np.argsort(scores)[-k:]` for every selector
    except kbest_chi2, which goes through SelectKBest and therefore returns
    indices in ascending column order. Column order is reproduced exactly
    because it perturbs the RNG stream of the tree ensembles.
    """
    k = min(k, len(scores))
    if name == "kbest_chi2":
        # SelectKBest ranks with a stable sort; 67 chi2 scores tie at 0/-inf,
        # so the sort kind changes which tied columns survive at the k boundary.
        idx = np.sort(np.argsort(scores, kind="mergesort")[-k:])
    else:
        idx = np.argsort(scores)[-k:]
    return idx


def verify_selector_equivalence(X: np.ndarray, y: np.ndarray, random_state: int = 0,
                                n_train: int = 43) -> None:
    """Assert the cached scorers select exactly what src/selectors.py selects."""
    Xa, ya = X[:n_train], y[:n_train]
    Xa_s = StandardScaler().fit_transform(Xa)
    for name in SELECTOR_ORDER:
        scores, _ = _score_selector(name, Xa_s, ya, random_state)
        for k in K_GRID:
            mine = _indices_for_k(name, scores, k)
            ref = SELECTORS[name](k=k, random_state=random_state).fit(Xa_s, ya).selected_
            if not np.array_equal(np.asarray(mine), np.asarray(ref)):
                raise AssertionError(
                    f"selector {name} k={k}: cached ranking does not reproduce "
                    f"src/selectors.py (|mine|={len(mine)}, |ref|={len(ref)}, "
                    f"set-equal={set(mine) == set(ref)})"
                )
    print("[verify] cached selector ranking reproduces src/selectors.py exactly")


# --------------------------------------------------------------------------
# Classifiers
# --------------------------------------------------------------------------

def _single_threaded(est):
    """Pin every estimator to one thread.

    Outer folds are parallelised, so per-estimator threading would
    oversubscribe the machine. On 43x10 training matrices the thread pool
    costs far more than it saves (Bagging: 4.9 s -> 0.09 s).
    """
    if hasattr(est, "n_jobs"):
        try:
            est.set_params(n_jobs=1)
        except Exception:
            pass
    for sub in getattr(est, "estimators", None) or []:
        _single_threaded(sub[1] if isinstance(sub, tuple) else sub)
    if getattr(est, "final_estimator", None) is not None:
        _single_threaded(est.final_estimator)
    if getattr(est, "estimator", None) is not None:
        _single_threaded(est.estimator)
    return est


def fresh_classifiers(random_state: int):
    return {n: _single_threaded(c)
            for n, c in build_classifiers(random_state=random_state).items()}


def _fit_predict(clf, Z_tr, y_tr, Z_te, sample_weight):
    """Fit with balanced sample weights where supported; return (label, score).

    Mirrors src/experiment_subject.run_loso: estimators that reject
    sample_weight (KNN, MLP, QDA) fall back to an unweighted fit.
    """
    try:
        clf.fit(Z_tr, y_tr, sample_weight=sample_weight)
    except (TypeError, ValueError):
        clf.fit(Z_tr, y_tr)
    label = clf.predict(Z_te)
    try:
        p = clf.predict_proba(Z_te)
        score = p[:, 1] if p.shape[1] == 2 else np.max(p, axis=1)
    except Exception:
        score = label.astype(float)
    return label.astype(int), score.astype(float)


def _pooled_metric(metric: str, y_true, y_pred, y_score) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "roc_auc":
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            return float("nan")
    raise KeyError(metric)


# --------------------------------------------------------------------------
# One outer fold
# --------------------------------------------------------------------------

def run_outer_fold(fold_idx, X, y, tr, te, random_state, verbose=True):
    """Inner search over all 840 configurations, then one held-out prediction.

    Returns one record per selection metric. The three metrics share the same
    inner predictions, so the search is run once and scored three ways.
    """
    t0 = time.time()
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                            random_state=random_state)
    n_tr = len(y_tr)
    # pooled inner out-of-fold predictions, per configuration
    pred = {}
    score = {}
    for s in SELECTOR_ORDER:
        for k in K_GRID:
            for c in CLASSIFIER_ORDER:
                pred[(s, k, c)] = np.full(n_tr, -1, dtype=int)
                score[(s, k, c)] = np.full(n_tr, np.nan, dtype=float)

    for in_tr, in_va in inner.split(X_tr, y_tr):
        Xa, ya = X_tr[in_tr], y_tr[in_tr]
        Xb = X_tr[in_va]
        sc = StandardScaler().fit(Xa)
        Xa_s, Xb_s = sc.transform(Xa), sc.transform(Xb)
        sw = compute_sample_weight("balanced", ya)
        for s in SELECTOR_ORDER:
            scores_vec, extra = _score_selector(s, Xa_s, ya, random_state)
            Xa_in = extra(Xa_s) if extra is not None else Xa_s
            Xb_in = extra(Xb_s) if extra is not None else Xb_s
            for k in K_GRID:
                idx = _indices_for_k(s, scores_vec, k)
                Za, Zb = Xa_in[:, idx], Xb_in[:, idx]
                clfs = fresh_classifiers(random_state)
                for c in CLASSIFIER_ORDER:
                    try:
                        lab, sco = _fit_predict(clfs[c], Za, ya, Zb, sw)
                    except Exception:
                        continue
                    pred[(s, k, c)][in_va] = lab
                    score[(s, k, c)][in_va] = sco

    # rank configurations; ties broken by (k, selector order, classifier order)
    rows = []
    for si, s in enumerate(SELECTOR_ORDER):
        for k in K_GRID:
            for ci, c in enumerate(CLASSIFIER_ORDER):
                p, v = pred[(s, k, c)], score[(s, k, c)]
                ok = p >= 0
                if ok.sum() < n_tr:
                    continue
                rec = {"selector": s, "k": k, "classifier": c,
                       "_tie": (k, si, ci)}
                for m in METRICS:
                    rec[m] = _pooled_metric(m, y_tr[ok], p[ok], v[ok])
                rows.append(rec)
    inner_table = pd.DataFrame(rows)

    # refit the winning configuration on all 54 training subjects
    sc = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = sc.transform(X_tr), sc.transform(X_te)
    sw = compute_sample_weight("balanced", y_tr)
    cached = {}
    out = []
    for m in METRICS:
        ranked = inner_table.sort_values(
            [m, "_tie"], ascending=[False, True], kind="mergesort"
        )
        best = ranked.iloc[0]
        n_tied_top = int((ranked[m] == best[m]).sum())
        s, k, c = best["selector"], int(best["k"]), best["classifier"]
        if s not in cached:
            cached[s] = _score_selector(s, Xtr_s, y_tr, random_state)
        scores_vec, extra = cached[s]
        idx = _indices_for_k(s, scores_vec, k)
        Xtr_in = extra(Xtr_s) if extra is not None else Xtr_s
        Xte_in = extra(Xte_s) if extra is not None else Xte_s
        lab, sco = _fit_predict(fresh_classifiers(random_state)[c],
                                Xtr_in[:, idx], y_tr, Xte_in[:, idx], sw)
        out.append({
            "metric": m, "fold": fold_idx,
            "subject_index": int(te[0]), "y_true": int(y_te[0]),
            "y_pred": int(lab[0]), "y_score": float(sco[0]),
            "selector": s, "k": k, "classifier": c,
            "inner_score": float(best[m]),
            "inner_n_configs": int(len(inner_table)),
            "inner_n_tied_top": n_tied_top,
        })
    if verbose:
        picks = " | ".join(f"{r['metric'][:3]}:{r['selector']}/{r['classifier']}/k={r['k']}"
                           f"{'ok' if r['y_pred'] == r['y_true'] else 'MISS'}"
                           for r in out)
        print(f"[fold {fold_idx + 1:2d}] {time.time() - t0:6.1f}s  {picks}", flush=True)
    return out


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def summarise(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    lo, hi = wilson_interval(correct, n)
    try:
        auc = float(roc_auc_score(y_true, df["y_score"].to_numpy()))
    except ValueError:
        auc = float("nan")
    return {
        "n": n, "correct": correct,
        "accuracy": correct / n,
        "wilson_lo": lo, "wilson_hi": hi,
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "auc_pooled": auc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="results/nested")
    ap.add_argument("--feature-set", choices=["tsfel_only", "tsfel_circadian"],
                    default="tsfel_only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cohort", choices=["full", "overlap_13_28", "first12", "no_nonwear"],
                    default="full",
                    help="full: all 55 subjects, all days | overlap_13_28: subjects with "
                         "13-28 complete days, all days | first12: every subject truncated to "
                         "first 12 complete days | no_nonwear: drop days containing a zero-count "
                         "run longer than 360 minutes")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--limit-folds", type=int, default=None,
                    help="debug: run only the first N outer folds")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / f"{args.feature_set}_{args.cohort}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading dataset")
    X_raw, y_day, g = load_depresjon(args.data_dir)
    print("[2/4] TSFEL features (cached)")
    feats_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                                            tag="depresjon", n_jobs=-1))
    print(f"[3/4] cohort selection: {args.cohort}")
    g = np.asarray(g); y_day = np.asarray(y_day)
    keep = np.ones(len(g), bool)
    if args.cohort == "first12":
        keep[:] = False
        for sid in np.unique(g):
            keep[np.flatnonzero(g == sid)[:12]] = True
    elif args.cohort == "overlap_13_28":
        counts = pd.Series(g).value_counts()
        keep = np.isin(g, [s for s in counts.index if 13 <= counts[s] <= 28])
    elif args.cohort == "no_nonwear":
        for i, day in enumerate(X_raw):
            z = (day == 0).astype(int)
            best = cur = 0
            for v in z:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            if best > 360:
                keep[i] = False
        # a subject must still have at least one complete day
        left = pd.Series(g[keep]).value_counts()
        keep &= np.isin(g, left.index)
    print(f"      days kept {int(keep.sum())}/{len(keep)}, subjects {len(np.unique(g[keep]))}")
    feats_day = feats_day.iloc[keep].reset_index(drop=True)
    X_raw = X_raw[keep]; g = g[keep]; y_day = y_day[keep]
    print("      per-subject aggregation")
    raw = None if args.feature_set == "tsfel_only" else X_raw
    feats, y_subj, subj_ids = aggregate_per_subject(feats_day, g, y_day,
                                                    raw_activity=raw)
    X = feats.to_numpy(dtype=np.float32)
    y = np.asarray(y_subj)
    print(f"      X={X.shape}  classes[control, depressed]={np.bincount(y).tolist()}")

    verify_selector_equivalence(X, y, random_state=args.seed)

    folds = list(LeaveOneOut().split(X))
    if args.limit_folds:
        folds = folds[:args.limit_folds]
    n_cfg = len(SELECTOR_ORDER) * len(K_GRID) * len(CLASSIFIER_ORDER)
    print(f"[4/4] nested LOSO: {len(folds)} outer folds x {n_cfg} inner "
          f"configurations x {INNER_FOLDS} inner folds")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
        delayed(run_outer_fold)(i, X, y, tr, te, args.seed)
        for i, (tr, te) in enumerate(folds)
    )
    per_fold = pd.DataFrame([r for chunk in results for r in chunk])
    per_fold = per_fold.sort_values(["metric", "fold"]).reset_index(drop=True)
    per_fold.to_csv(out_dir / "per_fold_predictions.csv", index=False)
    print(f"\ntotal wall clock: {time.time() - t0:.0f}s")

    summaries = {}
    for m in METRICS:
        sub = per_fold[per_fold.metric == m]
        s = summarise(sub)
        s["selection_metric"] = m
        chosen = (sub.groupby(["selector", "classifier"]).size()
                  .sort_values(ascending=False))
        s["config_stability"] = {f"{a}+{b}": int(v) for (a, b), v in chosen.items()}
        s["k_median"] = float(sub["k"].median())
        s["k_iqr"] = [float(sub["k"].quantile(.25)), float(sub["k"].quantile(.75))]
        summaries[m] = s

    (out_dir / "summary.json").write_text(json.dumps(
        {"feature_set": args.feature_set, "cohort": args.cohort, "n_subjects": int(len(y)),
         "subject_ids": [str(s) for s in subj_ids], "n_features": int(X.shape[1]),
         "seed": args.seed, "inner_folds": INNER_FOLDS,
         "n_inner_configurations": n_cfg, "results": summaries},
        indent=2))

    print(f"\n=== NESTED LOSO — {args.feature_set} ({X.shape[1]} features) ===")
    for m in METRICS:
        s = summaries[m]
        print(f"\nselection metric: {m}")
        print(f"  accuracy {s['accuracy']:.4f} ({s['correct']}/{s['n']})  "
              f"Wilson95 [{s['wilson_lo']:.3f}, {s['wilson_hi']:.3f}]")
        print(f"  kappa {s['kappa']:.4f}  bal.acc {s['balanced_accuracy']:.4f}  "
              f"sens {s['sensitivity']:.3f}  spec {s['specificity']:.3f}")
        print(f"  k median {s['k_median']:.0f} IQR {s['k_iqr']}")
        top = list(s["config_stability"].items())[:5]
        print("  chosen configurations: " +
              ", ".join(f"{a} x{b}" for a, b in top))
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
