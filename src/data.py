from pathlib import Path
import re
import numpy as np
import pandas as pd


SUBJECT_RE = re.compile(r"(condition|control|schizophrenia|patient)_(\d+)")


def _parse_subject_id(filename: str, class_label: str) -> str:
    # Stable subject id across runs. Class prefix prevents collision if
    # different folders happen to use overlapping numeric ids.
    m = SUBJECT_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot parse subject id from {filename}")
    return f"{class_label}_{m.group(2)}"


def load_depresjon(data_dir, expected_minutes=1440):
    """Two-class Depresjon: condition vs control.

    Returns
    -------
    X : np.ndarray (N, 1440) — daily activity vectors
    y : np.ndarray (N,)      — 0 control, 1 condition
    subjects : np.ndarray (N,) of str — subject id per row (group key for CV)
    """
    data_dir = Path(data_dir)
    classes = {"control": 0, "condition": 1}
    return _load_folders(data_dir, classes, expected_minutes)


def load_psychiatric(data_dir, expected_minutes=1440):
    """Three-class extension: control / depression / schizophrenia."""
    data_dir = Path(data_dir)
    classes = {"control": 0, "depression": 1, "schizophrenia": 2}
    return _load_folders(data_dir, classes, expected_minutes)


def _load_folders(root, classes, expected_minutes):
    X, y, subjects = [], [], []
    for folder, label in classes.items():
        folder_path = root / folder
        if not folder_path.exists():
            raise FileNotFoundError(folder_path)
        for csv_path in sorted(folder_path.iterdir()):
            if csv_path.suffix != ".csv":
                continue
            sid = _parse_subject_id(csv_path.name, folder)
            df = pd.read_csv(csv_path)
            for date, day_df in df.groupby("date"):
                if len(day_df) != expected_minutes:
                    continue
                X.append(day_df["activity"].to_numpy(dtype=np.float32))
                y.append(label)
                subjects.append(sid)
    if not X:
        raise RuntimeError(f"No valid days found under {root}")
    return np.stack(X), np.asarray(y), np.asarray(subjects)


def summarize(X, y, subjects):
    n_subj = len(np.unique(subjects))
    by_subj = pd.Series(subjects).value_counts()
    days_per_subj = by_subj.describe().to_dict()
    cls_counts = pd.Series(y).value_counts().to_dict()
    return {
        "n_samples": len(X),
        "n_subjects": n_subj,
        "class_counts": cls_counts,
        "days_per_subject": days_per_subj,
    }
