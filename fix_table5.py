"""Correct the Table 5 selection rates.

src.experiment_subject.selection_frequency_subject pools the selected-feature
log over ALL seven selectors (7 x 55 = 385 rows) and divides the count by 385.
Table 5 of the submitted manuscript reports those numbers but describes them
as "the fraction of the 55 folds in which the feature was retained" under the
Information Gain selector. The two are different quantities, and the pooled
counts (e.g. 315) exceed the 55 folds the caption refers to.

This script recomputes the rate per selector with the correct denominator.
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))

log = pd.read_csv("results/full_subject_level/selected_features.csv")
Path("tables").mkdir(exist_ok=True)

out = {}
for sel, grp in log.groupby("selector"):
    n_folds = grp["fold"].nunique()
    counts = (grp["features"].str.split(";").explode()
              .value_counts().rename("folds").to_frame())
    counts["selection_rate"] = counts["folds"] / n_folds
    counts["n_folds"] = n_folds
    out[sel] = counts

ig = out["info_gain"]
ig.head(20).to_csv("tables/table5_info_gain_corrected.csv")

pooled = (log["features"].str.split(";").explode().value_counts()
          .rename("count").to_frame())
pooled["as_published_rate"] = pooled["count"] / len(log)

cmp = ig.head(14).join(pooled["as_published_rate"])
cmp.index.name = "Feature"
print("Information Gain selector only, denominator = 55 folds (correct);")
print("'as published' = pooled over all 7 selectors, denominator = 385.\n")
print(cmp.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\nfolds per selector: {ig['n_folds'].iloc[0]}  |  rows in log: {len(log)}")
cmp.to_csv("tables/table5_comparison.csv")
