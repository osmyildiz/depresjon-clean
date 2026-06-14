"""Subject-level (LOSO, aggregated features) entry point.

Mirrors `run.py` but uses per-subject feature aggregation and
leave-one-subject-out evaluation. Intended as a complementary setting,
not a replacement: day-level (`run.py`) and subject-level (this script)
answer different clinical questions.
"""
import argparse
import json
import sys
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_depresjon, load_psychiatric, summarize
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject
from src.experiment_subject import (
    SubjectExperimentConfig, run_loso, selection_frequency_subject,
)


def pairwise_wilcoxon_from_correctness(correctness: pd.DataFrame,
                                       by: str = "classifier",
                                       fix: str = None):
    """Paired Wilcoxon on per-fold correctness (0/1).

    correctness columns are formatted as '{selector}__{classifier}'.
    `by` selects which axis varies; `fix` pins the other axis.
    """
    pairs = []
    cols = correctness.columns.tolist()
    parsed = [c.split("__") for c in cols]  # [(sel, clf), ...]

    if by == "classifier":
        if fix is None:
            raise ValueError("fix selector required when by='classifier'")
        keep = [(s, c) for s, c in parsed if s == fix]
        labels = [c for _, c in keep]
        cols_keep = [f"{s}__{c}" for s, c in keep]
    elif by == "selector":
        if fix is None:
            raise ValueError("fix classifier required when by='selector'")
        keep = [(s, c) for s, c in parsed if c == fix]
        labels = [s for s, _ in keep]
        cols_keep = [f"{s}__{c}" for s, c in keep]
    else:
        raise ValueError(by)

    sub = correctness[cols_keep].dropna(how="any")
    rows = []
    for (a, ca), (b, cb) in itertools.combinations(zip(labels, cols_keep), 2):
        x = sub[ca].to_numpy()
        y = sub[cb].to_numpy()
        try:
            stat, p = ss.wilcoxon(x, y)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({
            "a": a, "b": b,
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "diff": float(x.mean() - y.mean()),
            "stat": stat, "p": p, "n_pairs": int(len(sub)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("p").reset_index(drop=True)
        m = len(out)
        out["p_holm"] = (out["p"] * (m - np.arange(m))).clip(upper=1.0).cummax()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dataset", choices=["depresjon", "psychiatric"], default="depresjon")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--inner-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selectors", nargs="+", default=None)
    ap.add_argument("--classifiers", nargs="+", default=None)
    ap.add_argument("--num-features-grid", type=int, nargs="+",
                    default=[10, 20, 30, 50, 70, 100, 150, 200])
    ap.add_argument("--tag", default="subject_level")
    ap.add_argument("--tsfel-n-jobs", type=int, default=-1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Day-level load + TSFEL (cached from the day-level run)
    loader = load_depresjon if args.dataset == "depresjon" else load_psychiatric
    X_raw, y_day, g = loader(args.data_dir)
    summary = summarize(X_raw, y_day, g)
    print(json.dumps(summary, indent=2, default=str))

    feats_day = extract_tsfel(X_raw, g, cache_dir=args.cache_dir,
                              tag=args.dataset, n_jobs=args.tsfel_n_jobs)
    feats_day = drop_unstable(feats_day)
    print(f"Day-level features: {feats_day.shape}")

    # Subject aggregation: TSFEL stats + circadian features from raw activity
    feats_subj, y_subj, subj_ids = aggregate_per_subject(
        feats_day, g, y_day, raw_activity=X_raw
    )
    print(f"Subject-level features (TSFEL stats + circadian): {feats_subj.shape}, "
          f"class counts={np.bincount(y_subj).tolist()}")

    cfg = SubjectExperimentConfig(
        inner_folds=args.inner_folds,
        num_features_grid=tuple(args.num_features_grid),
        selectors=tuple(args.selectors) if args.selectors else
                  ("lasso", "ridge", "elastic_net", "kbest_chi2",
                   "fisher", "info_gain", "rf_importance"),
        classifiers=tuple(args.classifiers) if args.classifiers else None,
    )
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    summary_df, correctness, selected_log = run_loso(
        feats_subj, y_subj, subj_ids, cfg, random_state=args.seed
    )

    summary_df.to_csv(out_dir / "summary.csv", index=False)
    correctness.to_csv(out_dir / "per_fold_correctness.csv", index=False)
    selected_log.to_csv(out_dir / "selected_features.csv", index=False)

    # Pairwise Wilcoxon: classifiers within each selector
    wil_clf = []
    for sel in cfg.selectors:
        w = pairwise_wilcoxon_from_correctness(correctness, by="classifier", fix=sel)
        w["selector"] = sel
        wil_clf.append(w)
    pd.concat(wil_clf, ignore_index=True).to_csv(
        out_dir / "wilcoxon_classifiers_per_selector.csv", index=False
    )

    # Pairwise Wilcoxon: selectors within each classifier
    if cfg.classifiers is None:
        from src.experiment import build_classifiers
        clfs = tuple(build_classifiers().keys())
    else:
        clfs = cfg.classifiers
    wil_sel = []
    for clf in clfs:
        w = pairwise_wilcoxon_from_correctness(correctness, by="selector", fix=clf)
        w["classifier"] = clf
        wil_sel.append(w)
    pd.concat(wil_sel, ignore_index=True).to_csv(
        out_dir / "wilcoxon_selectors_per_classifier.csv", index=False
    )

    # Feature selection frequency
    freq = selection_frequency_subject(selected_log, top=30)
    freq.to_csv(out_dir / "feature_selection_frequency.csv")

    print(f"\nDone. Results in {out_dir}")
    print("\nTop 5 by accuracy:")
    print(summary_df.sort_values("accuracy", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
