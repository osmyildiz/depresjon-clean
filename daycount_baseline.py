"""Recording length as a nuisance baseline.

Depressed subjects were recorded for fewer days than controls, so day count
on its own carries label information. This reproduces the number quoted in
Section 5.5: a single feature, the number of complete 1440-minute days, under
the same leave-one-subject-out protocol as the main model.

Two learners are reported. The manuscript quotes the higher one (random
forest, 0.782); logistic regression on the same feature gives 0.673, and the
model-free AUC of day count is 0.737.

    python daycount_baseline.py --data-dir data/depresjon
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as ss
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from src.data import load_depresjon


def loso(x, y, make):
    pred = np.zeros(len(y), int)
    for tr, te in LeaveOneOut().split(x):
        sc = StandardScaler().fit(x[tr])
        m = make()
        try:
            m.fit(sc.transform(x[tr]), y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        except TypeError:
            m.fit(sc.transform(x[tr]), y[tr])
        pred[te[0]] = m.predict(sc.transform(x[te]))[0]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/depresjon")
    ap.add_argument("--out", default="results/daycount_baseline.json")
    args = ap.parse_args()

    _, y_day, g = load_depresjon(args.data_dir)
    g = np.asarray(g); y_day = np.asarray(y_day)
    ids = np.unique(g)
    y = np.array([int(y_day[g == s][0]) for s in ids])
    days = np.array([int((g == s).sum()) for s in ids], float).reshape(-1, 1)

    u, p = ss.mannwhitneyu(days[y == 1, 0], days[y == 0, 0])
    out = {
        "n": len(y),
        "days_median_control": float(np.median(days[y == 0])),
        "days_median_depressed": float(np.median(days[y == 1])),
        "mannwhitney_U": float(u), "mannwhitney_p": float(p),
        "auc_model_free": float(roc_auc_score(y, -days[:, 0])),
        "majority": float(max(np.bincount(y)) / len(y)),
    }
    for name, make in [
        ("random_forest", lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=1)),
        ("logistic_regression", lambda: LogisticRegression(max_iter=2000)),
        ("decision_stump", lambda: DecisionTreeClassifier(max_depth=1, random_state=0)),
    ]:
        pred = loso(days, y, make)
        k = int((pred == y).sum())
        out[name] = {"correct": k, "n": len(y), "accuracy": k / len(y)}
        print(f"{name:20s} {k}/{len(y)} = {k/len(y):.3f}")
    print(f"model-free AUC of day count: {out['auc_model_free']:.3f}   majority {out['majority']:.3f}")
    print(f"day counts: control median {out['days_median_control']:.0f}, "
          f"depressed {out['days_median_depressed']:.0f}, U = {u:.1f}, p = {p:.4f}")

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
