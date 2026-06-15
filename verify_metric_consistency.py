"""Verify whether classifier pairs that report identical metrics in
summary.csv actually classify the same subjects correctly.

Reads per_fold_correctness.csv (one row per LOSO fold, columns =
(selector, classifier) configurations, values = 0/1 correctness per
held-out subject) and computes pairwise Hamming distances for the
Information Gain selector.

Run from depresjon-clean project root:
    python verify_metric_consistency.py --results-dir results/full_subject_level
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/full_subject_level")
    args = ap.parse_args()

    pfc_path = Path(args.results_dir) / "per_fold_correctness.csv"
    summary_path = Path(args.results_dir) / "summary.csv"

    if not pfc_path.exists():
        sys.exit(f"missing: {pfc_path}")
    if not summary_path.exists():
        sys.exit(f"missing: {summary_path}")

    pfc = pd.read_csv(pfc_path)
    summary = pd.read_csv(summary_path)

    print(f"per_fold_correctness.csv columns: {list(pfc.columns)[:8]}...")
    print(f"per_fold_correctness.csv shape: {pfc.shape}")
    print()

    # Information Gain selector filtre
    # PFC kolon adları büyük olasılıkla "selector|classifier" formatında veya
    # ayrı kolonlar olarak. İki olası yapıyı kontrol et:
    if "selector" in pfc.columns and "classifier" in pfc.columns:
        # Long format: each row is (selector, classifier, subject_id, correct)
        ig = pfc[pfc["selector"] == "info_gain"]
        # Pivot: subjects as rows, classifiers as columns
        if "subject_id" in pfc.columns:
            wide = ig.pivot(index="subject_id", columns="classifier", values="correct")
        else:
            # fall back: use index
            wide = ig.pivot_table(values="correct", columns="classifier",
                                  aggfunc="first")
        classifiers = list(wide.columns)
    else:
        # Wide format: kolon adları "selector__classifier" olabilir
        ig_cols = [c for c in pfc.columns if c.startswith("info_gain")]
        if not ig_cols:
            # belki tek başına classifier adı kolonlar, selector ayrı bir kolonda olmadan
            print("Cannot identify info_gain columns; column dump:")
            print(list(pfc.columns))
            sys.exit(1)
        wide = pfc[ig_cols].copy()
        # column rename: info_gain__Bagging -> Bagging
        wide.columns = [c.split("__")[-1] if "__" in c else c.replace("info_gain_", "")
                        for c in wide.columns]
        classifiers = list(wide.columns)

    print(f"Info Gain classifiers found: {len(classifiers)}")
    print(f"Subjects: {len(wide)}")
    print(f"Wide shape: {wide.shape}")
    print()

    # Her classifier için doğru sayısı
    print("=== Correct count per classifier (Info Gain) ===")
    correct_counts = wide.sum().sort_values(ascending=False)
    print(correct_counts.to_string())
    print()

    # Bagging vs RandomForest karşılaştırma
    print("=== Pairwise per-subject agreement (Info Gain) ===")
    print("Hamming distance = # subjects on which the two classifiers disagree")
    print("If distance = 0, they have identical per-subject predictions.")
    print()
    suspected_pairs = [
        ("Bagging", "RandomForest"),
        ("XGBoost", "Stacking"),
        ("ExtraTrees", "MLP"),
        ("SVM", "AdaBoost"),
        ("SVM", "LogReg_EN"),
        ("AdaBoost", "LogReg_EN"),
        ("LogReg_L1", "LogReg_L2"),
    ]
    for a, b in suspected_pairs:
        if a not in wide.columns or b not in wide.columns:
            print(f"  {a} vs {b}: column missing (a={a in wide.columns}, b={b in wide.columns})")
            continue
        va, vb = wide[a].values, wide[b].values
        hd = int((va != vb).sum())
        agree = int((va == vb).sum())
        # subjects where A correct B wrong
        a_only = int(((va == 1) & (vb == 0)).sum())
        b_only = int(((va == 0) & (vb == 1)).sum())
        print(f"  {a:14s} vs {b:14s}: hamming={hd:2d} | A-only correct={a_only}  B-only correct={b_only}  agree={agree}")
    print()

    # Tüm çiftler için Hamming matrisi
    print("=== Full pairwise Hamming distance matrix (Info Gain, lower triangle) ===")
    n = len(classifiers)
    M = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            va = wide[classifiers[i]].values
            vb = wide[classifiers[j]].values
            M[i, j] = int((va != vb).sum())
    dfM = pd.DataFrame(M, index=classifiers, columns=classifiers)
    print(dfM.to_string())
    print()
    print("Çiftler için 0 hamming = tıpatıp aynı predictions.")


if __name__ == "__main__":
    main()
