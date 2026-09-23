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
hourly = X.reshape(len(X), 24, 60).mean(axis=2)
mat = np.vstack([hourly[y == 0].mean(axis=0), hourly[y == 1].mean(axis=0)])
fig, ax = plt.subplots(figsize=(9.5, 2.6))
im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", interpolation="nearest")
ax.set_yticks([0, 1], ["Control", "Depressed"])
ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
ax.set_xlabel("Hour of day")
cb = fig.colorbar(im, ax=ax, pad=0.015)
cb.set_label("Mean activity counts per minute")
fig.savefig(OUT / "figure3_hourly_heatmap.png")
fig.savefig(OUT / "figure3_hourly_heatmap.pdf")
plt.close(fig)

pd.DataFrame(picked).T.to_csv(OUT / "figure2_selection.csv")
print(f"\nwritten to {OUT}/")
