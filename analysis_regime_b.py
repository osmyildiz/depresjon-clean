"""Post-hoc analysis for Regime B results.

Adds two things to the existing summary.csv / per_fold_correctness.csv:
  1. Wilson and bootstrap 95% CIs for accuracy per (selector, classifier).
  2. Exact McNemar tests for pairwise classifier comparisons, with Holm
     correction within each selector family. This replaces the Wilcoxon
     pass for Regime B, which is suboptimal on binary correctness vectors
     due to ties.

Usage:
    python analysis_regime_b.py --results-dir results/full_subject_level

Reads:    summary.csv, per_fold_correctness.csv
Writes:   summary_with_ci.csv, mcnemar_classifiers_per_selector.csv
"""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar


def wilson_ci(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(center - half), float(center + half)


def bootstrap_ci_accuracy(correct: np.ndarray, n_boot: int = 5000,
                          conf: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=int)
    n = len(correct)
    if n == 0:
        return (np.nan, np.nan)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = correct[idx].mean()
    lo = float(np.quantile(boots, (1 - conf) / 2))
    hi = float(np.quantile(boots, 1 - (1 - conf) / 2))
    return lo, hi


def exact_mcnemar(a_correct: np.ndarray, b_correct: np.ndarray):
    a = np.asarray(a_correct, dtype=int)
    b = np.asarray(b_correct, dtype=int)
    # 2x2 table:
    #            B correct  B wrong
    # A correct     n11        n10
    # A wrong       n01        n00
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    table = [[n11, n10], [n01, n00]]
    # exact=True uses binomial test on discordant pairs; recommended for
    # small samples (Regime B: n=55).
    res = sm_mcnemar(table, exact=True, correction=False)
    return float(res.pvalue), n10, n01


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, p[idx] * (m - rank))
        adj[idx] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    summary = pd.read_csv(rd / "summary.csv")
    correctness = pd.read_csv(rd / "per_fold_correctness.csv")

    # 1. CIs for each (selector, classifier)
    rows = []
    for _, r in summary.iterrows():
        col = f"{r['selector']}__{r['classifier']}"
        if col not in correctness.columns:
            continue
        c = correctness[col].dropna().astype(int).to_numpy()
        n = len(c)
        k = int(c.sum())
        w_lo, w_hi = wilson_ci(k, n)
        b_lo, b_hi = bootstrap_ci_accuracy(c)
        rows.append({
            **r.to_dict(),
            "n_correct": k,
            "wilson_lo": w_lo, "wilson_hi": w_hi,
            "boot_lo": b_lo, "boot_hi": b_hi,
        })
    summary_ci = pd.DataFrame(rows)
    summary_ci.to_csv(rd / "summary_with_ci.csv", index=False)

    # 2. Pairwise McNemar within each selector
    selectors = sorted(summary["selector"].unique())
    all_rows = []
    for sel in selectors:
        cols = [c for c in correctness.columns if c.startswith(sel + "__")]
        labels = [c.split("__", 1)[1] for c in cols]
        sub = correctness[cols].dropna(how="any").astype(int)
        pairs_in_family = list(itertools.combinations(range(len(cols)), 2))
        pvals = []
        meta = []
        for i, j in pairs_in_family:
            p, n10, n01 = exact_mcnemar(sub.iloc[:, i], sub.iloc[:, j])
            pvals.append(p)
            meta.append((labels[i], labels[j], n10, n01,
                         float(sub.iloc[:, i].mean()),
                         float(sub.iloc[:, j].mean())))
        adj = holm(pvals)
        for (a, b, n10, n01, ma, mb), p, pa in zip(meta, pvals, adj):
            all_rows.append({
                "selector": sel,
                "a": a, "b": b,
                "mean_a": ma, "mean_b": mb,
                "diff": ma - mb,
                "n_a_only": n10, "n_b_only": n01,
                "p_exact": p, "p_holm": pa,
            })
    mc = pd.DataFrame(all_rows).sort_values(["selector", "p_exact"]).reset_index(drop=True)
    mc.to_csv(rd / "mcnemar_classifiers_per_selector.csv", index=False)

    # Quick console preview
    print("Top 10 configurations by accuracy, with Wilson 95% CI:")
    top = summary_ci.sort_values("accuracy", ascending=False).head(10)
    cols_show = ["selector", "classifier", "accuracy",
                 "wilson_lo", "wilson_hi", "auc", "kappa"]
    print(top[cols_show].to_string(index=False))
    print()
    print("McNemar (exact, Holm-corrected) — top 10 most significant pairs:")
    top_mc = mc.sort_values("p_holm").head(10)
    print(top_mc[["selector", "a", "b", "mean_a", "mean_b",
                  "diff", "p_exact", "p_holm"]].to_string(index=False))

    print(f"\nWrote: {rd/'summary_with_ci.csv'}")
    print(f"Wrote: {rd/'mcnemar_classifiers_per_selector.csv'}")


if __name__ == "__main__":
    main()
