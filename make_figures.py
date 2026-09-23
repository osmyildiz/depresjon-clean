"""Regenerate Figures 2 and 3 at publication quality.

Fixes three reviewer comments:
  * Figure 2: the vertical axis label was clipped in the submitted version.
  * Figure 3: the horizontal axis label was clipped.
  * Figure 2: the two example subjects were shown without a stated selection
    rule. Here the rule is prespecified and reproducible -- within each group
    we plot the subject whose Relative Amplitude (RA) is the median of that
    group, and within that subject the day whose total activity is the median
    of that subject's days. RA is the circadian biomarker most directly tied
    to the depressed/control contrast (Van Someren et al., 1999), so the
    median-RA subject is the group's typical case on the axis the figure is
    meant to illustrate, not its most favourable one.

A fourth issue found while regenerating: the submitted Figure 3 caption reads
"cool colors indicate lower activity; warm colors indicate higher activity",
but the YlGnBu colormap runs pale-yellow (low) to dark-blue (high), i.e. the
opposite. The caption text in CAPTIONS below is corrected.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from src.data import load_depresjon
from src.circadian import compute_circadian_features

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    "figure.constrained_layout.use": True,
})

X, y, g = load_depresjon("data/depresjon")
g = np.asarray(g); y = np.asarray(y)

# --- prespecified subject selection: median Relative Amplitude within group ---
ra = {}
for sid in np.unique(g):
    m = g == sid
    feats = compute_circadian_features(X[m])
    key = "circ_RA"
    ra[sid] = float(feats[key]) if key else np.nan
ra_s = pd.Series(ra)
lab = pd.Series({sid: int(y[g == sid][0]) for sid in ra_s.index})

picked = {}
for cls, name in [(0, "control"), (1, "depressed")]:
    sub = ra_s[lab == cls].dropna().sort_values()
    sid = sub.index[(len(sub) - 1) // 2]          # lower median, deterministic
    days = np.flatnonzero(g == sid)
    totals = X[days].sum(axis=1)
    day = days[np.argsort(totals)[(len(days) - 1) // 2]]  # median-activity day
    picked[name] = {"subject": sid, "ra": float(sub.loc[sid]),
                    "day_row": int(day), "n_days": int(len(days)),
                    "ra_rank": f"{list(sub.index).index(sid) + 1}/{len(sub)}"}
print("Figure 2 selection (prespecified: median-RA subject, median-activity day):")
for k, v in picked.items():
    print(f"  {k:10s} subject={v['subject']:12s} RA={v['ra']:.4f} "
          f"(rank {v['ra_rank']})  days={v['n_days']}")

# --- Figure 2 ---
t = np.arange(1440) / 60.0
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for ax, (name, info), colour, title in zip(
        axes, picked.items(), ["#1f4e9c", "#c0392b"],
        ["Control (median-RA subject)", "Depressed (median-RA subject)"]):
    ax.plot(t, X[info["day_row"]], color=colour, lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.grid(alpha=0.3, lw=0.5)
axes[0].set_ylabel("Activity counts per minute")
fig.savefig(OUT / "figure2_daily_profiles.png")
fig.savefig(OUT / "figure2_daily_profiles.pdf")
plt.close(fig)

# --- Figure 3 ---
# Reviewer 3 asked for two colour-coded group curves instead of a heatmap: a
# difference in amplitude is easier to read as two lines than as two rows of
# colour. Shaded bands are the standard error of the hourly mean across the
# subjects of each group, so the reader can see where the groups separate.
hourly = X.reshape(len(X), 24, 60).mean(axis=2)          # (days, 24)
subj_hourly, subj_lab = [], []
for sid in np.unique(g):
    m = g == sid
    subj_hourly.append(hourly[m].mean(axis=0))
    subj_lab.append(int(y[m][0]))
subj_hourly = np.array(subj_hourly); subj_lab = np.array(subj_lab)

hours = np.arange(24)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
for cls, name, colour in [(0, "Control", "#1f4e9c"), (1, "Depressed", "#c0392b")]:
    grp = subj_hourly[subj_lab == cls]
    mean = grp.mean(axis=0)
    sem = grp.std(axis=0, ddof=1) / np.sqrt(len(grp))
    ax.plot(hours, mean, color=colour, lw=1.8, label=f"{name} (n = {len(grp)})")
    ax.fill_between(hours, mean - sem, mean + sem, color=colour, alpha=0.18, lw=0)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean activity counts per minute")
ax.set_xlim(0, 23)
ax.set_xticks(range(0, 24, 3))
ax.grid(alpha=0.3, lw=0.5)
ax.legend(frameon=False, loc="upper left")
fig.savefig(OUT / "figure3_hourly_profiles.png")
fig.savefig(OUT / "figure3_hourly_profiles.pdf")
plt.close(fig)

# peak difference, for the caption
ctrl = subj_hourly[subj_lab == 0].mean(axis=0); dep = subj_hourly[subj_lab == 1].mean(axis=0)
print(f"Figure 3: control peak {ctrl.max():.0f} at {ctrl.argmax():02d}:00, "
      f"depressed peak {dep.max():.0f} at {dep.argmax():02d}:00; "
      f"largest gap {np.max(ctrl - dep):.0f} counts at {np.argmax(ctrl - dep):02d}:00")

pd.DataFrame(picked).T.to_csv(OUT / "figure2_selection.csv")
print(f"\nwritten to {OUT}/")
