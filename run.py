import argparse
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.data import load_depresjon, load_psychiatric, summarize
from src.features import extract_tsfel, drop_unstable
from src.experiment import ExperimentConfig, run
from src.stats import aggregate, pairwise_wilcoxon, selection_frequency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Folder containing condition/, control/ (or 3-class equivalents).")
    ap.add_argument("--dataset", choices=["depresjon", "psychiatric"], default="depresjon")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--use-smote", action="store_true")
    ap.add_argument("--outer-folds", type=int, default=10)
    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--selectors", nargs="+", default=None,
                    help="Subset of selectors to run; default = all.")
    ap.add_argument("--classifiers", nargs="+", default=None,
                    help="Subset of classifiers to run; default = all.")
    ap.add_argument("--num-features-grid", type=int, nargs="+",
                    default=[20, 30, 50, 70, 90, 110, 130, 150])
    ap.add_argument("--tag", default="run1")
    ap.add_argument("--tsfel-n-jobs", type=int, default=-1,
                    help="Parallelism for TSFEL extraction; -1=all cores.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = load_depresjon if args.dataset == "depresjon" else load_psychiatric
    X_raw, y, groups = loader(args.data_dir)
    summary = summarize(X_raw, y, groups)
    print(json.dumps(summary, indent=2, default=str))
    (out_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    feats = extract_tsfel(X_raw, groups, cache_dir=args.cache_dir,
                          tag=args.dataset, n_jobs=args.tsfel_n_jobs)
    feats = drop_unstable(feats)
    print(f"Features after stability filter: {feats.shape}")

    cfg = ExperimentConfig(
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seeds=tuple(args.seeds),
        num_features_grid=tuple(args.num_features_grid),
        selectors=tuple(args.selectors) if args.selectors else
                  ("lasso", "ridge", "elastic_net", "kbest_chi2",
                   "fisher", "info_gain", "rf_importance"),
        use_smote=args.use_smote,
        classifiers=tuple(args.classifiers) if args.classifiers else None,
    )
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=str))

    results, selected_log = run(feats, y, groups, cfg)
    results.to_csv(out_dir / "raw_results.csv", index=False)
    selected_log.to_csv(out_dir / "selected_features.csv", index=False)

    for metric in ["accuracy", "f1", "auc", "kappa"]:
        agg = aggregate(results, metric=metric)
        agg.to_csv(out_dir / f"summary_{metric}.csv", index=False)

    # Significance: classifiers within each selector
    wil_clf = []
    for sel in results["selector"].unique():
        w = pairwise_wilcoxon(results, metric="accuracy",
                              by="classifier", fix_selector=sel)
        w["selector"] = sel
        wil_clf.append(w)
    pd.concat(wil_clf).to_csv(out_dir / "wilcoxon_classifiers_per_selector.csv", index=False)

    # Significance: selectors within each classifier
    wil_sel = []
    for clf in results["classifier"].unique():
        sub = results[results["classifier"] == clf]
        pivot = sub.pivot_table(
            index=["seed", "fold"], columns="selector", values="accuracy"
        ).dropna(how="any")
        # quick paired summary
        import itertools
        from scipy import stats as ss
        for a, b in itertools.combinations(pivot.columns, 2):
            try:
                stat, p = ss.wilcoxon(pivot[a], pivot[b])
            except ValueError:
                stat, p = float("nan"), float("nan")
            wil_sel.append({
                "classifier": clf, "a": a, "b": b,
                "mean_a": pivot[a].mean(), "mean_b": pivot[b].mean(),
                "stat": stat, "p": p, "n_pairs": len(pivot),
            })
    pd.DataFrame(wil_sel).to_csv(
        out_dir / "wilcoxon_selectors_per_classifier.csv", index=False
    )

    freq = selection_frequency(selected_log, top=30)
    freq.to_csv(out_dir / "feature_selection_frequency.csv")

    print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
