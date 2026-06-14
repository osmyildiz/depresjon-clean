"""Diagnose why circadian features failed to improve accuracy.

Checks four hypotheses:
  1. Cosinor params have out-of-band values (scaling issue)
  2. NaN imputation hides failed circadian computations
  3. IS values are abnormally low due to minute-resolution
  4. Circadian features rank high in selectors but provide no real signal

Run:
    python diagnose_circadian.py --data-dir data/depresjon
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject
from src.circadian import compute_circadian_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()

    # Load data
    X_raw, y_day, g = load_depresjon(args.data_dir)
    print(f"Data: {X_raw.shape}, {len(set(g))} subjects, "
          f"class counts {np.bincount(y_day).tolist()}")
    print()

    # Compute circadian features per subject (raw, before any imputation)
    print("=" * 72)
    print("STEP 1: Raw circadian features per subject (before imputation)")
    print("=" * 72)
    subjects_arr = np.asarray(g)
    rows = []
    for sid in sorted(set(subjects_arr)):
        mask = subjects_arr == sid
        subj_X = X_raw[mask]
        label = int(y_day[mask][0])
        feats = compute_circadian_features(subj_X)
        feats["subject_id"] = sid
        feats["label"] = label
        feats["n_days"] = int(mask.sum())
        rows.append(feats)
    circ_df = pd.DataFrame(rows)

    # Hypothesis 1: out-of-band values
    print("\n[H1] Value ranges per circadian feature:")
    print("    Feature                          min        max     median       NaN")
    print("    " + "-" * 76)
    circ_cols = [c for c in circ_df.columns if c.startswith("circ_")]
    for c in circ_cols:
        v = circ_df[c]
        n_nan = int(v.isna().sum())
        if v.notna().any():
            print(f"    {c:30s}  {v.min():9.3f}  {v.max():9.3f}  "
                  f"{v.median():9.3f}  {n_nan:4d}/55")
        else:
            print(f"    {c:30s}  ALL NaN                              {n_nan:4d}/55")

    # Hypothesis 2: NaN counts
    print(f"\n[H2] NaN summary across all circadian features:")
    nan_per_subj = circ_df[circ_cols].isna().sum(axis=1)
    print(f"    Subjects with no NaN in circadian: {(nan_per_subj == 0).sum()}/55")
    print(f"    Subjects with at least 1 NaN     : {(nan_per_subj > 0).sum()}/55")
    print(f"    Mean NaN per subject             : {nan_per_subj.mean():.2f}")
    bad_subjects = circ_df.loc[nan_per_subj > 0, ["subject_id", "n_days"]]
    if len(bad_subjects):
        print(f"    Affected subjects (id, n_days):")
        for _, r in bad_subjects.iterrows():
            print(f"      {r['subject_id']}: {r['n_days']} days")

    # Hypothesis 3: IS values too low (minute vs hour resolution)
    print(f"\n[H3] IS distribution (Witting 1990 typical band: 0.4–0.8 for healthy):")
    is_vals = circ_df["circ_IS"].dropna()
    print(f"    min={is_vals.min():.4f}, "
          f"max={is_vals.max():.4f}, "
          f"median={is_vals.median():.4f}")
    if is_vals.median() < 0.2:
        print("    ⚠️  IS median below 0.2 — likely minute-resolution effect.")
        print("       Witting's IS is computed on hourly bins (p=24), not p=1440.")
    elif is_vals.median() < 0.4:
        print("    ⚠️  IS median lower than typical literature range.")
    else:
        print("    ✓  IS in expected literature range.")

    # Per-class comparison: do circadian values actually differ between
    # depressed and control?
    print(f"\n[H4] Class separation (Mann-Whitney U on each circadian feature):")
    print("    Feature                       healthy_med    depr_med   U-stat   p")
    print("    " + "-" * 76)
    from scipy.stats import mannwhitneyu
    healthy = circ_df[circ_df["label"] == 0]
    depressed = circ_df[circ_df["label"] == 1]
    for c in circ_cols:
        h = healthy[c].dropna()
        d = depressed[c].dropna()
        if len(h) > 3 and len(d) > 3:
            try:
                u, p = mannwhitneyu(h, d, alternative="two-sided")
                marker = "  ***" if p < 0.01 else ("  *" if p < 0.05 else "")
                print(f"    {c:28s}  {h.median():10.3f}  {d.median():10.3f}  "
                      f"{u:7.1f}  {p:.4f}{marker}")
            except Exception:
                print(f"    {c:28s}  comparison failed")

    print()
    print("=" * 72)
    print("STEP 2: After scaling — do circadian features dominate?")
    print("=" * 72)
    from sklearn.preprocessing import StandardScaler
    # Compare scale impact for circadian vs first few TSFEL features
    feats_day = extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                              tag="depresjon", n_jobs=-1)
    feats_day = drop_unstable(feats_day)
    feats_subj, y_subj, sids = aggregate_per_subject(
        feats_day, g, y_day, raw_activity=X_raw)

    cols_circ = [c for c in feats_subj.columns if c.startswith("circ_")]
    cols_zcr = [c for c in feats_subj.columns if "Zero crossing rate" in c][:6]
    cols_show = cols_zcr + cols_circ

    print(f"\nRaw value ranges (pre-scaling):")
    for c in cols_show:
        v = feats_subj[c]
        print(f"    {c:50s}  min={v.min():10.3f}  max={v.max():10.3f}  "
              f"std={v.std():9.3f}")

    print(f"\nMutual Information with label (no scaling needed for MI):")
    from sklearn.feature_selection import mutual_info_classif
    X_subj = feats_subj.values
    mi = mutual_info_classif(X_subj, y_subj, discrete_features=False,
                             random_state=0)
    mi_df = pd.DataFrame({"feature": feats_subj.columns, "mi": mi})
    mi_df["is_circadian"] = mi_df["feature"].str.startswith("circ_")
    mi_df = mi_df.sort_values("mi", ascending=False)

    print(f"\nTop 20 features by Mutual Information:")
    for _, r in mi_df.head(20).iterrows():
        tag = " [CIRCADIAN]" if r["is_circadian"] else ""
        print(f"    MI={r['mi']:.4f}  {r['feature']:50s}{tag}")

    print(f"\nCircadian features by MI rank:")
    for _, r in mi_df[mi_df["is_circadian"]].iterrows():
        rank = (mi_df["feature"] == r["feature"]).idxmax()
        rank_pos = mi_df.index.get_loc(rank) + 1
        print(f"    Rank #{rank_pos:4d}  MI={r['mi']:.4f}  {r['feature']}")

    # Save circ_df for later inspection
    circ_df.to_csv("circadian_diagnostic.csv", index=False)
    mi_df.to_csv("mi_ranking.csv", index=False)
    print(f"\nWrote: circadian_diagnostic.csv, mi_ranking.csv")


if __name__ == "__main__":
    main()
