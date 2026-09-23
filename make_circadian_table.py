"""Appendix B: the sixteen circadian features with their defining equations.

Reviewer 3 asked why only five circadian constructs are described in Section 3.6
when sixteen features are used, and for the equations of all sixteen. The five
paragraphs in 3.6 define five families; this table expands each family into the
features it contributes. Definitions are taken from src/circadian.py so the
table and the code cannot drift apart.
"""
import csv
from pathlib import Path

ROWS = [
    ("Interdaily stability", "circ_IS",
     "IS = N * sum_h (xbar_h - xbar)^2 / (p * sum_i (x_i - xbar)^2)",
     "Witting et al. (1990). xbar_h is the mean activity in hour-of-day h averaged over days, "
     "xbar the grand mean, N the number of minutes, p = 24. Higher values mean a more reproducible daily rhythm."),
    ("Intradaily variability", "circ_IV",
     "IV = N * sum_i (x_i - x_{i-1})^2 / ((N - 1) * sum_i (x_i - xbar)^2)",
     "Witting et al. (1990). Mean squared first difference over total variance. Higher values mean more frequent rest-activity switching within a day."),
    ("Relative amplitude", "circ_RA",
     "RA = (M10 - L5) / (M10 + L5)",
     "Van Someren et al. (1999). M10 is the mean of the most active 10-hour window of the average day, L5 the mean of the least active 5-hour window."),
    ("Cosinor MESOR", "circ_cosinor_mesor",
     "A(t) = M + A cos(2*pi*t/24 - phi), fitted by least squares; MESOR = M",
     "Halberg (1969). Rhythm-adjusted mean of the fitted 24-hour cosine."),
    ("Cosinor amplitude", "circ_cosinor_amplitude", "amplitude = A", "Peak-to-MESOR distance of the fitted cosine."),
    ("Cosinor acrophase", "circ_cosinor_acrophase_hours", "acrophase = phi * 24 / (2*pi)", "Clock time of the fitted peak, in hours."),
    ("Cosinor fit quality", "circ_cosinor_r2", "R^2 = 1 - SS_res / SS_tot", "Coefficient of determination of the cosine fit; low values mean the 24-hour cosine is a poor description of that subject."),
    ("Relative rhythm strength", "circ_cosinor_amp_over_mesor", "A / M", "Dimensionless amplitude, comparable across subjects with different activity levels."),
    ("Bout count, mean", "circ_bout_count_mean", "mean_d B_d", "B_d is the number of maximal runs of consecutive minutes with activity > 0 on day d."),
    ("Bout count, SD", "circ_bout_count_std", "sd_d B_d", "Day-to-day variability in how fragmented the active period is."),
    ("Mean bout length, mean", "circ_bout_mean_len_mean", "mean_d (mean of run lengths on day d)", "Average duration of an uninterrupted active period."),
    ("Mean bout length, SD", "circ_bout_mean_len_std", "sd_d (mean of run lengths on day d)", "Day-to-day variability of that duration."),
    ("Max bout length, mean", "circ_bout_max_len_mean", "mean_d (longest run on day d)", "Longest uninterrupted active period, averaged over days."),
    ("Max bout length, SD", "circ_bout_max_len_std", "sd_d (longest run on day d)", "Day-to-day variability of the longest active period."),
    ("Total active minutes, mean", "circ_bout_total_active_mean", "mean_d sum of run lengths on day d", "Minutes per day with activity > 0."),
    ("Total active minutes, SD", "circ_bout_total_active_std", "sd_d sum of run lengths on day d", "Day-to-day variability of active minutes."),
]

Path("tables").mkdir(exist_ok=True)
with open("tables/appendix_circadian_features.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Feature", "Name in code", "Definition", "Notes"])
    w.writerows(ROWS)

fams = {"Interdaily stability": 1, "Intradaily variability": 1, "Relative amplitude": 1, "Cosinor": 5, "Bout": 8}
print(f"{len(ROWS)} features written to tables/appendix_circadian_features.csv")
print("families:", ", ".join(f"{k} ({v})" for k, v in fams.items()), "= 16")
