"""Confusion matrix and per-class metrics for the best subject-level
configuration (Information Gain + Bagging) under LOSO.

Run from the depresjon-clean project root:
    python confusion_best.py --data-dir data --cache-dir cache

Reuses the cached TSFEL features from the main run, so this is fast
(only the one selector x one classifier is re-fit across 55 folds).
"""
import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject
from src.experiment_subject import _inner_select_k, build_classifiers
from src.selectors import SELECTORS

SELECTOR = "info_gain"
CLASSIFIER = "Bagging"
SEED = 0
GRID = (10, 20, 30, 50, 70, 100, 150, 200)
INNER_FOLDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--selector", default=SELECTOR)
    ap.add_argument("--classifier", default=CLASSIFIER)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    X_raw, y_day, g = load_depresjon(args.data_dir)
    feats_day = extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                              tag="depresjon", n_jobs=-1)
    feats_day = drop_unstable(feats_day)
    feats_subj, y_subj, subj_ids = aggregate_per_subject(
        feats_day, g, y_day, raw_activity=X_raw
    )

    X = feats_subj.to_numpy(dtype=np.float32)
    y = np.asarray(y_subj)
    counts = np.bincount(y)
    print(f"Subjects: {len(y)}  class counts [control=0, depressed=1]: {counts.tolist()}")

    loo = LeaveOneOut()
    preds = np.full(len(y), -1, dtype=int)
    ks = []
    for fold, (tr, te) in enumerate(loo.split(X)):
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        best_k, _ = _inner_select_k(X_tr, y_tr, args.selector, GRID,
                                    INNER_FOLDS, args.seed)
        ks.append(best_k)
        scaler = StandardScaler().fit(X_tr)
        Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
        sel = SELECTORS[args.selector](k=best_k, random_state=args.seed).fit(Xs_tr, y_tr)
        Xtr_sel, Xte_sel = sel.transform(Xs_tr), sel.transform(Xs_te)
        clf = build_classifiers(random_state=args.seed)[args.classifier]
        sw = compute_sample_weight("balanced", y_tr)
        try:
            clf.fit(Xtr_sel, y_tr, sample_weight=sw)
        except (TypeError, ValueError):
            clf.fit(Xtr_sel, y_tr)
        preds[te[0]] = int(clf.predict(Xte_sel)[0])

    acc = accuracy_score(y, preds)
    kappa = cohen_kappa_score(y, preds)
    cm = confusion_matrix(y, preds, labels=[0, 1])  # rows=true, cols=pred
    TN, FP = int(cm[0, 0]), int(cm[0, 1])
    FN, TP = int(cm[1, 0]), int(cm[1, 1])

    sens = TP / (TP + FN) if (TP + FN) else float("nan")   # recall depressed
    spec = TN / (TN + FP) if (TN + FP) else float("nan")   # recall control
    prec_dep = TP / (TP + FP) if (TP + FP) else float("nan")
    prec_ctrl = TN / (TN + FN) if (TN + FN) else float("nan")

    print(f"\n=== {args.selector} + {args.classifier} (LOSO, n={len(y)}) ===")
    print(f"median k = {np.median(ks):.0f}  (IQR {np.percentile(ks,25):.0f}-{np.percentile(ks,75):.0f})")
    print(f"accuracy = {acc:.4f}  ({TP+TN}/{len(y)})")
    print(f"kappa    = {kappa:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred; 0=control, 1=depressed):")
    print(f"                 pred=control   pred=depressed")
    print(f"true=control        {TN:3d}            {FP:3d}")
    print(f"true=depressed      {FN:3d}            {TP:3d}")
    print(f"\nTP={TP}  TN={TN}  FP={FP}  FN={FN}")
    print(f"sensitivity (recall, depressed) = {sens:.3f}  ({TP}/{TP+FN})")
    print(f"specificity (recall, control)   = {spec:.3f}  ({TN}/{TN+FP})")
    print(f"precision depressed = {prec_dep:.3f}")
    print(f"precision control   = {prec_ctrl:.3f}")

    print("\n--- Paste-ready values for paper Section 4.1 ---")
    print(f"N_D_correct = {TP}   SENS = {sens:.3f}")
    print(f"N_C_correct = {TN}   SPEC = {spec:.3f}")
    print(f"N_D_miss (FN, depressed->control) = {FN}")
    print(f"N_C_miss (FP, control->depressed) = {FP}")


if __name__ == "__main__":
    main()
