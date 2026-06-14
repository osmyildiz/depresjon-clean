"""Subject-level feature aggregation.

Each subject contributes one row of features summarizing the distribution
of their daily TSFEL feature values across all valid days, plus a set of
classical circadian biomarkers computed from their raw multi-day activity
record. With 156 TSFEL features × 7 order statistics + ~12 circadian
features, this yields ~1104 columns per subject.
"""
import numpy as np
import pandas as pd

from .circadian import compute_circadian_features


_STATS = ["mean", "std", "min", "max", "p25", "p50", "p75"]


def aggregate_per_subject(features: pd.DataFrame, subjects, labels,
                          raw_activity=None):
    """Aggregate day-level features into one row per subject.

    Parameters
    ----------
    features : DataFrame (n_days, n_features) — TSFEL daily features
    subjects : array-like of length n_days — subject id per day
    labels   : array-like of length n_days — class label per day
    raw_activity : np.ndarray (n_days, 1440), optional
        Raw daily activity vectors. If provided, circadian features are
        computed per subject and concatenated.

    Returns
    -------
    agg_features : DataFrame (n_subjects, n_aggregated_features)
    agg_labels   : ndarray  (n_subjects,)
    subject_ids  : ndarray  (n_subjects,)
    """
    df = features.copy()
    df["__subject"] = np.asarray(subjects)
    df["__label"] = np.asarray(labels)

    label_check = df.groupby("__subject")["__label"].nunique()
    bad = label_check[label_check > 1]
    if len(bad):
        raise ValueError(
            f"Subjects with inconsistent labels across days: {bad.to_dict()}"
        )

    base_cols = [c for c in df.columns if c not in ("__subject", "__label")]
    subjects_arr = np.asarray(subjects)

    rows = []
    subj_ids = []
    subj_labels = []
    for sid, g in df.groupby("__subject", sort=True):
        x = g[base_cols].to_numpy(dtype=np.float32)
        row = {}
        for stat in _STATS:
            if stat == "mean":
                vals = x.mean(axis=0)
            elif stat == "std":
                vals = x.std(axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1])
            elif stat == "min":
                vals = x.min(axis=0)
            elif stat == "max":
                vals = x.max(axis=0)
            elif stat == "p25":
                vals = np.percentile(x, 25, axis=0)
            elif stat == "p50":
                vals = np.percentile(x, 50, axis=0)
            elif stat == "p75":
                vals = np.percentile(x, 75, axis=0)
            for col, v in zip(base_cols, vals):
                row[f"{stat}_{col}"] = float(v)

        # Circadian features from raw daily activity, if provided
        if raw_activity is not None:
            mask = subjects_arr == sid
            subj_activity = np.asarray(raw_activity)[mask]
            row.update(compute_circadian_features(subj_activity))

        rows.append(row)
        subj_ids.append(sid)
        subj_labels.append(int(g["__label"].iloc[0]))

    agg = pd.DataFrame(rows)
    # NaN'ları median ile doldur (circadian fits bazı durumlarda NaN dönebilir)
    agg = agg.fillna(agg.median(numeric_only=True))
    return agg, np.asarray(subj_labels), np.asarray(subj_ids)
