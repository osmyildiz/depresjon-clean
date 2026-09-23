"""Day-count confound checks requested in §5.5 of the manuscript.

The submitted manuscript defers two checks to future work:
  (a) whether per-subject recording length differs between the depressed
      and control groups (a Mann-Whitney test), and
  (b) whether the min/max order statistics -- which scale with the number
      of available days by construction -- drive the result.

Both are inexpensive. This script runs them.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as ss

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable
from src.aggregation import aggregate_per_subject

X_raw, y_day, g = load_depresjon("data/depresjon")
feats_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir="cache",
                                        tag="depresjon", n_jobs=-1))
feats, y, ids = aggregate_per_subject(feats_day, g, y_day, raw_activity=None)
y = np.asarray(y)

days = pd.Series(g).value_counts().reindex(ids).to_numpy()
ctrl, dep = days[y == 0], days[y == 1]

u, p = ss.mannwhitneyu(dep, ctrl, alternative="two-sided")
# rank-biserial effect size
rb = 1 - 2 * u / (len(dep) * len(ctrl))

out = {
    "n_control": int(len(ctrl)), "n_depressed": int(len(dep)),
    "days_control": {"median": float(np.median(ctrl)),
                     "q25": float(np.percentile(ctrl, 25)),
                     "q75": float(np.percentile(ctrl, 75)),
                     "min": int(ctrl.min()), "max": int(ctrl.max()),
                     "mean": float(ctrl.mean())},
    "days_depressed": {"median": float(np.median(dep)),
                       "q25": float(np.percentile(dep, 25)),
                       "q75": float(np.percentile(dep, 75)),
                       "min": int(dep.min()), "max": int(dep.max()),
                       "mean": float(dep.mean())},
    "mannwhitney_U": float(u), "mannwhitney_p": float(p),
    "rank_biserial": float(-rb),
}

# Correlation between recording length and per-subject feature values, for the
# order statistics that scale with day count versus those that do not.
cols = list(feats.columns)
groups = {"min": [], "max": [], "p25": [], "p50": [], "p75": [], "mean": [], "std": []}
for i, c in enumerate(cols):
    stat = c.rsplit("_", 1)[-1] if c.rsplit("_", 1)[-1] in groups else c.split("_")[0]
    for gname in groups:
        if c.startswith(gname + "_") or c.endswith("_" + gname):
            groups[gname].append(i)
            break
M = feats.to_numpy(dtype=np.float64)
corr = {}
for gname, idx in groups.items():
    if not idx:
        continue
    rs = []
    for i in idx:
        v = M[:, i]
        if np.std(v) == 0 or not np.all(np.isfinite(v)):
            continue
        rs.append(abs(ss.spearmanr(days, v).statistic))
    corr[gname] = {"n_features": len(rs),
                   "median_abs_spearman_with_daycount": float(np.median(rs)),
                   "frac_abs_rho_gt_0.4": float(np.mean(np.array(rs) > 0.4))}
out["daycount_feature_correlation"] = corr

print(json.dumps(out, indent=2))
Path("results").mkdir(exist_ok=True)
Path("results/daycount_confound.json").write_text(json.dumps(out, indent=2))
