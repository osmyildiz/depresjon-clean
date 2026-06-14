"""Aggregation and significance testing for nested CV results."""
import itertools
import numpy as np
import pandas as pd
from scipy import stats


def aggregate(results: pd.DataFrame, metric="accuracy"):
    """Mean, std, 95% bootstrap CI per (selector, classifier)."""
    def _ci(x):
        x = x.dropna().to_numpy()
        if len(x) < 2:
            return (np.nan, np.nan)
        rng = np.random.default_rng(0)
        boots = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    g = results.groupby(["selector", "classifier"])[metric]
    summary = g.agg(["mean", "std", "count"]).reset_index()
    cis = g.apply(_ci).reset_index()
    cis.columns = ["selector", "classifier", "ci"]
    cis["ci_low"] = cis["ci"].apply(lambda t: t[0])
    cis["ci_high"] = cis["ci"].apply(lambda t: t[1])
    cis = cis.drop(columns=["ci"])
    return summary.merge(cis, on=["selector", "classifier"])


def pairwise_wilcoxon(results: pd.DataFrame, metric="accuracy",
                      by="classifier", fix_selector=None):
    """Paired Wilcoxon across folds.

    Pairing key: (seed, fold) within a fixed `fix_selector` if provided;
    otherwise within (seed, fold, selector). Returns long-format pvals.
    """
    df = results.copy()
    if fix_selector is not None:
        df = df[df["selector"] == fix_selector]
        pair_keys = ["seed", "fold"]
    else:
        pair_keys = ["seed", "fold", "selector" if by == "classifier" else "classifier"]

    pivot = df.pivot_table(
        index=pair_keys, columns=by, values=metric
    ).dropna(how="any")
    items = list(pivot.columns)
    rows = []
    for a, b in itertools.combinations(items, 2):
        x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
        try:
            stat, p = stats.wilcoxon(x, y)
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append({
            "a": a, "b": b,
            "mean_a": float(np.mean(x)), "mean_b": float(np.mean(y)),
            "diff": float(np.mean(x) - np.mean(y)),
            "stat": stat, "p": p, "n_pairs": len(x),
        })
    out = pd.DataFrame(rows)
    if len(out):
        # Holm correction across all pairs
        out = out.sort_values("p").reset_index(drop=True)
        m = len(out)
        out["p_holm"] = (out["p"] * (m - np.arange(m))).clip(upper=1.0)
        out["p_holm"] = out["p_holm"].cummax()
    return out


def selection_frequency(selected_log: pd.DataFrame, top=20):
    """How often each feature was retained, across (seed, fold, selector)."""
    rows = []
    total = 0
    for _, row in selected_log.iterrows():
        for feat in row["features"].split(";"):
            rows.append(feat)
        total += 1
    s = pd.Series(rows).value_counts().head(top).rename("count").to_frame()
    s["rate"] = s["count"] / total
    return s
