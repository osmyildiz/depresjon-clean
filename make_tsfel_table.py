"""Appendix table of the TSFEL features used (reviewer minor comment 5).

Lists every day-level TSFEL feature that survives drop_unstable(), grouped by
TSFEL domain, together with the number of columns each contributes and the
seven order statistics applied to it during per-subject aggregation.
"""
import sys, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.features import extract_tsfel, drop_unstable

import tsfel
cfg = tsfel.get_features_by_domain()
domain_of = {}
for domain, feats in cfg.items():
    for fname in feats:
        domain_of[fname] = domain.capitalize()

X, y, g = load_depresjon("data/depresjon")
raw = extract_tsfel(X, g, cache_dir="cache", tag="depresjon", n_jobs=-1)
kept = drop_unstable(raw)
dropped = [c for c in raw.columns if c not in set(kept.columns)]

def base_name(col):
    c = re.sub(r"^0_", "", col)
    c = re.sub(r"_\d+$", "", c)                 # MFCC_3 -> MFCC
    c = re.sub(r"_[\d.]+Hz$", "", c)            # ..._5.0Hz -> ...
    return c.strip()

rows = {}
for col in kept.columns:
    b = base_name(col)
    d = domain_of.get(b, "Unclassified")
    key = (d, b)
    rows[key] = rows.get(key, 0) + 1

tbl = (pd.DataFrame([{"Domain": d, "TSFEL feature": b, "Day-level columns": n}
                     for (d, b), n in rows.items()])
       .sort_values(["Domain", "TSFEL feature"])
       .reset_index(drop=True))
tbl["Subject-level columns (x7 order statistics)"] = tbl["Day-level columns"] * 7

Path("tables").mkdir(exist_ok=True)
tbl.to_csv("tables/appendix_tsfel_features.csv", index=False)

print(f"TSFEL raw columns: {raw.shape[1]}   after drop_unstable: {kept.shape[1]}   "
      f"dropped: {len(dropped)}")
print(f"distinct TSFEL features retained: {tbl['TSFEL feature'].nunique()}")
print(f"subject-level total: {kept.shape[1]} x 7 = {kept.shape[1]*7}")
print()
print(tbl.groupby("Domain")[["Day-level columns",
                             "Subject-level columns (x7 order statistics)"]].sum()
      .assign(**{"Distinct features": tbl.groupby("Domain")["TSFEL feature"].nunique()})
      .to_string())
print()
print(tbl.to_string(index=False))
if dropped:
    print("\ndropped by drop_unstable (zero variance / non-finite):")
    for c in dropped:
        print("  ", c)
