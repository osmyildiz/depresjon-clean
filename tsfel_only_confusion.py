"""TSFEL-only baseline confusion matrix with live progress output.

Run from depresjon-clean project root:
    python tsfel_only_confusion.py --data-dir data/depresjon --cache-dir cache
"""
import argparse
import sys
import time
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


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    log("[1/4] Loading dataset...")
    X_raw, y_day, g = load_depresjon(args.data_dir)
    log(f"      {len(y_day)} days across {len(set(g))} subjects")

    log("[2/4] Extracting / loading TSFEL features (cache reuse if available)...")
    feats_day = extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                              tag="depresjon", n_jobs=-1)
    feats_day = drop_unstable(feats_day)
    log(f"      per-day features: {feats_day.shape}")

    log("[3/4] Aggregating to per-subject (TSFEL-only, no circadian)...")
    feats_subj, y_subj, subj_ids = aggregate_per_subject(
        feats_day, g, y_day, raw_activity=None
    )
    X = feats_subj.to_numpy(dtype=np.float32)
    y = np.asarray(y_subj)
    counts = np.bincount(y)
    log(f"      X shape: {X.shape}  classes [control=0, depressed=1]: {counts.tolist()}")

    log(f"[4/4] LOSO ({len(y)} folds) with {SELECTOR} + {CLASSIFIER}...")
    loo = LeaveOneOut()
    preds = np.full(len(y), -1, dtype=int)
    ks = []
    t0 = time.time()
    for fold, (tr, te) in enumerate(loo.split(X)):
        ft = time.time()
        X_tr, X_te = X[tr], X[te]
        y_tr = y[tr]
        best_k, _ = _inner_select_k(X_tr, y_tr, SELECTOR, GRID,
                                    INNER_FOLDS, args.seed)
        ks.append(best_k)
        scaler = StandardScaler().fit(X_tr)
        Xs_tr, Xs_te = scaler.transform(X_tr), scaler.transform(X_te)
        sel = SELECTORS[SELECTOR](k=best_k, random_state=args.seed).fit(Xs_tr, y_tr)
        Xtr_sel, Xte_sel = sel.transform(Xs_tr), sel.transform(Xs_te)
        clf = build_classifiers(random_state=args.seed)[CLASSIFIER]
        sw = compute_sample_weight("balanced", y_tr)
        try:
            clf.fit(Xtr_sel, y_tr, sample_weight=sw)
        except (TypeError, ValueError):
            clf.fit(Xtr_sel, y_tr)
        preds[te[0]] = int(clf.predict(Xte_sel)[0])
        dur = time.time() - ft
        correct = "ok" if preds[te[0]] == y[te[0]] else "MISS"
        log(f"      fold {fold+1:2d}/{len(y)}  k={best_k:3d}  pred={preds[te[0]]} true={y[te[0]]}  [{correct}]  ({dur:.1f}s)")

    total = time.time() - t0
    log(f"\nTotal LOSO time: {total:.1f}s")

    acc = accuracy_score(y, preds)
    kappa = cohen_kappa_score(y, preds)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    TN, FP = int(cm[0, 0]), int(cm[0, 1])
    FN, TP = int(cm[1, 0]), int(cm[1, 1])

    sens = TP / (TP + FN) if (TP + FN) else float("nan")
    spec = TN / (TN + FP) if (TN + FP) else float("nan")

    log(f"\n=== RESULT: TSFEL-only, {SELECTOR} + {CLASSIFIER} (LOSO, n={len(y)}) ===")
    log(f"median k = {int(np.median(ks))}  (IQR {int(np.percentile(ks,25))}-{int(np.percentile(ks,75))})")
    log(f"accuracy = {acc:.4f}  ({TP+TN}/{len(y)})")
    log(f"kappa    = {kappa:.4f}")
    log("")
    log("Confusion matrix (rows=true, cols=pred; 0=control, 1=depressed):")
    log(f"                 pred=control   pred=depressed")
    log(f"true=control        {TN:3d}            {FP:3d}")
    log(f"true=depressed      {FN:3d}            {TP:3d}")
    log(f"\nTP={TP}  TN={TN}  FP={FP}  FN={FN}")
    log(f"sensitivity = {sens:.3f}  ({TP}/{TP+FN})")
    log(f"specificity = {spec:.3f}  ({TN}/{TN+FP})")

    log("\n--- PASTE-READY VALUES ---")
    log(f"accuracy   = {acc:.4f}  ({TP+TN}/55)")
    log(f"kappa      = {kappa:.4f}")
    log(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
    log(f"sensitivity = {sens:.3f}")
    log(f"specificity = {spec:.3f}")
    log(f"median_k   = {int(np.median(ks))}")
    log(f"feature_dim= {X.shape[1]}")


if __name__ == "__main__":
    main()
