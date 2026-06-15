"""For each (info_gain, classifier) configuration, compute the actual
per-class confusion matrix from per_fold_correctness.csv combined with
the true subject labels. Then compare against summary.csv to determine
whether identical metrics correspond to identical confusion matrices
or to a metric-computation bug.

Run from depresjon-clean project root:
    python verify_confusion.py --data-dir data/depresjon --results-dir results/full_subject_level
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/depresjon")
    ap.add_argument("--results-dir", default="results/full_subject_level")
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    # True subject labels — exactly as used during the run
    from src.data import load_depresjon
    from src.features import extract_tsfel, drop_unstable
    from src.aggregation import aggregate_per_subject

    X_raw, y_day, g = load_depresjon(args.data_dir)
    feats_day = extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                              tag="depresjon", n_jobs=-1)
    feats_day = drop_unstable(feats_day)
    _, y_subj, subj_ids = aggregate_per_subject(feats_day, g, y_day, raw_activity=None)
    y_subj = np.asarray(y_subj)
    print(f"Subjects: {len(y_subj)}  classes [0,1]: {np.bincount(y_subj).tolist()}")
    print(f"First 5 subject IDs: {subj_ids[:5]}  labels: {y_subj[:5].tolist()}")
    print(f"Last 5 subject IDs:  {subj_ids[-5:]}  labels: {y_subj[-5:].tolist()}")
    print()

    # Load predictions correctness
    pfc = pd.read_csv(Path(args.results_dir) / "per_fold_correctness.csv")
    ig_cols = [c for c in pfc.columns if c.startswith("info_gain__")]
    print(f"Info Gain columns in per_fold_correctness.csv: {len(ig_cols)}")
    print(f"Rows (should match subjects): {len(pfc)}")
    print()

    if len(pfc) != len(y_subj):
        print("WARNING: row count mismatch between per_fold_correctness and y_subj.")
        print("Assuming pfc rows are in the same order as the LOSO fold order.")
        print("If both come from the same load_depresjon() call, this should match.")

    # For each classifier, compute confusion from correctness vector + true labels
    summary = pd.read_csv(Path(args.results_dir) / "summary.csv")
    ig_summary = summary[summary["selector"] == "info_gain"].copy()
    ig_summary["classifier"] = ig_summary["classifier"].astype(str)

    print(f"{'classifier':14s}  TP  TN  FP  FN  | acc    sens    spec    kappa   F1w   | sum.acc sum.f1w sum.kappa  match")
    print("-" * 130)

    rows = []
    for col in ig_cols:
        clf = col.replace("info_gain__", "")
        correct = pfc[col].values
        if len(correct) != len(y_subj):
            print(f"  {clf}: size mismatch ({len(correct)} vs {len(y_subj)}), skipped")
            continue

        # Derive predictions from correctness:
        # if correct[i] == 1: pred = true
        # if correct[i] == 0: pred = 1 - true (binary)
        pred = np.where(correct == 1, y_subj, 1 - y_subj)

        TP = int(((y_subj == 1) & (pred == 1)).sum())
        TN = int(((y_subj == 0) & (pred == 0)).sum())
        FP = int(((y_subj == 0) & (pred == 1)).sum())
        FN = int(((y_subj == 1) & (pred == 0)).sum())

        acc = (TP + TN) / len(y_subj)
        sens = TP / (TP + FN) if (TP + FN) else 0.0
        spec = TN / (TN + FP) if (TN + FP) else 0.0

        # Cohen's kappa
        po = acc
        pe = ((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / len(y_subj) ** 2
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0

        # Weighted F1
        n_pos = TP + FN
        n_neg = TN + FP
        p_pos = TP / (TP + FP) if (TP + FP) else 0.0
        r_pos = TP / (TP + FN) if (TP + FN) else 0.0
        f1_pos = 2 * p_pos * r_pos / (p_pos + r_pos) if (p_pos + r_pos) else 0.0
        p_neg = TN / (TN + FN) if (TN + FN) else 0.0
        r_neg = TN / (TN + FP) if (TN + FP) else 0.0
        f1_neg = 2 * p_neg * r_neg / (p_neg + r_neg) if (p_neg + r_neg) else 0.0
        f1_w = (n_pos * f1_pos + n_neg * f1_neg) / len(y_subj)

        # Compare with summary.csv
        s = ig_summary[ig_summary["classifier"] == clf]
        if len(s) == 0:
            tag = "[no summary row]"
            s_acc = s_f1 = s_kappa = float("nan")
        else:
            s_acc = float(s["accuracy"].iloc[0])
            s_f1 = float(s["f1"].iloc[0])
            s_kappa = float(s["kappa"].iloc[0])
            close = abs(acc - s_acc) < 1e-4 and abs(kappa - s_kappa) < 1e-4 and abs(f1_w - s_f1) < 1e-4
            tag = "OK" if close else "MISMATCH"

        print(f"{clf:14s}  {TP:2d}  {TN:2d}  {FP:2d}  {FN:2d}  | "
              f"{acc:.4f}  {sens:.4f}  {spec:.4f}  {kappa:.4f}  {f1_w:.4f} | "
              f"{s_acc:.4f}  {s_f1:.4f}  {s_kappa:.4f}    {tag}")

        rows.append({
            "classifier": clf, "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "accuracy": acc, "sensitivity": sens, "specificity": spec,
            "kappa": kappa, "f1_weighted": f1_w
        })

    print()
    df = pd.DataFrame(rows)
    df = df.sort_values("accuracy", ascending=False).reset_index(drop=True)

    print("=== TRUE Per-Class Confusion Matrices (Info Gain) ===")
    print(df.to_string(index=False))
    print()

    print("=== GROUP BY (TP, TN, FP, FN): which classifiers actually share confusion matrices? ===")
    groups = df.groupby(["TP", "TN", "FP", "FN"])
    for key, gdf in groups:
        clfs = list(gdf["classifier"])
        if len(clfs) > 1:
            print(f"  TP={key[0]:2d}  TN={key[1]:2d}  FP={key[2]:2d}  FN={key[3]:2d}  ({len(clfs)} classifiers): {clfs}")
        else:
            print(f"  TP={key[0]:2d}  TN={key[1]:2d}  FP={key[2]:2d}  FN={key[3]:2d}  (unique): {clfs}")

    # Save corrected table
    out = Path(args.results_dir) / "info_gain_true_confusion.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
