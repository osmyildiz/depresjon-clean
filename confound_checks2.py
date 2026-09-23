"""Separating the recording-length confound from simple loss of observation time.

confound_checks.py truncates every subject to their first 12 days. That removes
the confound completely -- day count becomes constant, so it carries no label
information -- but it also discards about a third of the observed days, so the
distributional summaries are estimated from fewer samples. The resulting 0.764
therefore mixes two effects and is a lower bound on confound-free performance.

This script adds the complementary design: keep every day of every retained
subject, but restrict the cohort to the day-count range the two groups share
(13-28 days), which leaves 47 subjects (25 control, 22 depressed) whose day
counts no longer differ significantly between groups (Mann-Whitney p = 0.068,
against p = 0.0021 in the full cohort). Information is preserved; the confound
is attenuated rather than eliminated.

Read together:
    full        confound present,    all information        upper bound
    overlap     confound attenuated, all information        key estimate
    first12     confound eliminated, information reduced    lower bound

A first-14-day variant is also run; the cohort cannot be pushed past 14 days
because only 5 depressed subjects have 16 or more.
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
from confound_checks import loso

X_raw, y_day, g = load_depresjon("data/depresjon")
feats_day = drop_unstable(extract_tsfel(X_raw, g, cache_dir="cache",
                                        tag="depresjon", n_jobs=-1))
g = np.asarray(g); y_day = np.asarray(y_day)
dc = pd.Series(g).value_counts()

def build(day_mask, label):
    F, y, ids = aggregate_per_subject(
        feats_day.iloc[day_mask].reset_index(drop=True),
        g[day_mask], y_day[day_mask], raw_activity=None)
    y = np.asarray(y)
    days = pd.Series(g[day_mask]).value_counts().reindex(ids).to_numpy()
    u, p = ss.mannwhitneyu(days[y == 1], days[y == 0])
    print(f"{label}: n={len(y)} (ctrl {int((y==0).sum())}, dep {int((y==1).sum())}), "
          f"{F.shape[1]} features, "
          f"day-count Mann-Whitney p={p:.4f}", flush=True)
    return F, y, float(p)

out = {}

# first 14 chronological days, subjects with at least 14 days
keep14 = np.zeros(len(g), bool)
for sid in np.unique(g):
    if dc[sid] >= 14:
        keep14[np.flatnonzero(g == sid)[:14]] = True
F, y, p = build(keep14, "first14 (>=14 days)")
r = loso(F.to_numpy(dtype=np.float32), y); r["daycount_p"] = p
out["first14"] = r

# overlapping day-count range, every day retained
keep_ov = np.isin(g, [s for s in dc.index if 13 <= dc[s] <= 28])
F, y, p = build(keep_ov, "overlap 13-28 days")
r = loso(F.to_numpy(dtype=np.float32), y); r["daycount_p"] = p
out["overlap_13_28"] = r

for name, r in out.items():
    print(f"{name:16s} acc={r['accuracy']:.4f} ({r['correct']}/{r['n']})  "
          f"bal={r['balanced_accuracy']:.4f}  kappa={r['kappa']:.3f}  "
          f"majority={max(r['TP']+r['FN'], r['TN']+r['FP'])/r['n']:.3f}", flush=True)

Path("results/confound_checks2.json").write_text(json.dumps(out, indent=2))
print("\nwritten to results/confound_checks2.json")
