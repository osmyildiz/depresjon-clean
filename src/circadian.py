"""Circadian rhythm features for actigraphy — bug-fixed version.

Changes from v4 (diagnostic exposed three issues):

  1. Bout features are now computed per-day and then aggregated across
     a subject's days (mean and std). The previous version concatenated
     all of a subject's days into one long vector before bout detection,
     which made every subject with continuous non-zero activity register
     as a single bout of length (n_days × 1440 minutes). This produced
     artefactual high-MI bout features that polluted feature selection.

  2. IS and IV are now computed on hourly-binned signal (p = 24), as in
     Witting et al. 1990, rather than on minute-level signal (p = 1440).
     Minute-level resolution suppressed IS values to ~0.05–0.30 versus
     the literature range of 0.4–0.8 for healthy subjects.

  3. cosinor_amplitude is reported both raw and normalized by MESOR
     (relative amplitude); cosinor_r2 is retained as a fit-quality flag.

Reference: Witting et al. 1990, Van Someren et al. 1999, Halberg 1969.
"""
import numpy as np
from scipy.optimize import curve_fit


_MIN_PER_DAY = 1440
_MIN_PER_HOUR = 60


def _to_hourly(activity_long: np.ndarray) -> np.ndarray:
    """Bin minute-level activity to hourly averages."""
    n = len(activity_long)
    n_hours = n // _MIN_PER_HOUR
    if n_hours == 0:
        return np.array([])
    return (activity_long[: n_hours * _MIN_PER_HOUR]
            .reshape(n_hours, _MIN_PER_HOUR).mean(axis=1))


def interdaily_stability(activity_long: np.ndarray) -> float:
    """Witting 1990 IS at hourly resolution (p = 24)."""
    hourly = _to_hourly(activity_long)
    n = len(hourly)
    p = 24
    if n < p:
        return np.nan
    n_days = n // p
    arr = hourly[: n_days * p].reshape(n_days, p)
    hour_of_day_mean = arr.mean(axis=0)
    total_mean = hourly.mean()
    num = n * np.sum((hour_of_day_mean - total_mean) ** 2)
    den = p * np.sum((hourly - total_mean) ** 2)
    if den == 0:
        return np.nan
    return float(num / den)


def intradaily_variability(activity_long: np.ndarray) -> float:
    """Witting 1990 IV at hourly resolution."""
    hourly = _to_hourly(activity_long)
    n = len(hourly)
    if n < 2:
        return np.nan
    diff_sq = np.sum(np.diff(hourly) ** 2)
    mean_dev_sq = np.sum((hourly - hourly.mean()) ** 2)
    if mean_dev_sq == 0:
        return np.nan
    return float(n * diff_sq / ((n - 1) * mean_dev_sq))


def relative_amplitude(activity_long: np.ndarray,
                       m_hours: int = 10, l_hours: int = 5) -> float:
    """Van Someren 1999 RA using averaged daily profile (minute-level)."""
    n_days = len(activity_long) // _MIN_PER_DAY
    if n_days < 1:
        return np.nan
    arr = activity_long[: n_days * _MIN_PER_DAY].reshape(n_days, _MIN_PER_DAY)
    daily_profile = arr.mean(axis=0)
    m_win = m_hours * _MIN_PER_HOUR
    l_win = l_hours * _MIN_PER_HOUR
    if len(daily_profile) < max(m_win, l_win):
        return np.nan
    roll_m = np.convolve(daily_profile, np.ones(m_win) / m_win, mode='valid')
    roll_l = np.convolve(daily_profile, np.ones(l_win) / l_win, mode='valid')
    M10 = roll_m.max()
    L5 = roll_l.min()
    if M10 + L5 == 0:
        return np.nan
    return float((M10 - L5) / (M10 + L5))


def cosinor_fit(activity_long: np.ndarray) -> dict:
    """24-hour cosinor fit. Returns MESOR, amplitude, acrophase, R²,
    and amplitude/MESOR ratio (cohort-comparable relative amplitude).
    """
    if len(activity_long) < _MIN_PER_DAY:
        return {"mesor": np.nan, "amplitude": np.nan,
                "acrophase_hours": np.nan, "r2": np.nan,
                "amp_over_mesor": np.nan}
    t = np.arange(len(activity_long)) / 60.0

    def model(t, mesor, amplitude, phase):
        return mesor + amplitude * np.cos(2 * np.pi * t / 24 + phase)

    try:
        p0 = [activity_long.mean(), activity_long.std(), 0.0]
        popt, _ = curve_fit(model, t, activity_long, p0=p0, maxfev=2000)
        mesor, amplitude, phase = popt
        amplitude = abs(amplitude)
        acrophase = (-24 * phase / (2 * np.pi)) % 24
        y_pred = model(t, *popt)
        ss_res = np.sum((activity_long - y_pred) ** 2)
        ss_tot = np.sum((activity_long - activity_long.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        amp_over_mesor = amplitude / mesor if mesor > 0 else np.nan
        return {"mesor": float(mesor), "amplitude": float(amplitude),
                "acrophase_hours": float(acrophase), "r2": float(r2),
                "amp_over_mesor": float(amp_over_mesor)}
    except Exception:
        return {"mesor": np.nan, "amplitude": np.nan,
                "acrophase_hours": np.nan, "r2": np.nan,
                "amp_over_mesor": np.nan}


def daily_bout_stats(day_activity: np.ndarray) -> dict:
    """Bout statistics for a SINGLE day (1440 minutes).

    A bout is a contiguous block of non-zero activity minutes.
    """
    is_active = day_activity > 0
    if not is_active.any():
        return {"count": 0.0, "mean_len": 0.0, "max_len": 0.0,
                "total_active": 0.0}
    pad = np.concatenate(([False], is_active, [False]))
    diff = np.diff(pad.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    bout_lens = ends - starts
    if len(bout_lens) == 0:
        return {"count": 0.0, "mean_len": 0.0, "max_len": 0.0,
                "total_active": 0.0}
    return {
        "count": float(len(bout_lens)),
        "mean_len": float(np.mean(bout_lens)),
        "max_len": float(np.max(bout_lens)),
        "total_active": float(np.sum(bout_lens)),
    }


def compute_circadian_features(activity_per_day: np.ndarray) -> dict:
    """Compute all circadian features for a subject.

    activity_per_day : np.ndarray (n_days, 1440) — daily activity matrix
    """
    activity_long = activity_per_day.reshape(-1).astype(np.float64)

    # 1. Rhythm metrics on the multi-day signal
    features = {
        "circ_IS": interdaily_stability(activity_long),
        "circ_IV": intradaily_variability(activity_long),
        "circ_RA": relative_amplitude(activity_long),
    }
    cos = cosinor_fit(activity_long)
    features.update({f"circ_cosinor_{k}": v for k, v in cos.items()})

    # 2. Bout features computed PER DAY, then aggregated across days
    per_day_bouts = [daily_bout_stats(activity_per_day[i])
                     for i in range(len(activity_per_day))]
    for stat in ["count", "mean_len", "max_len", "total_active"]:
        vals = np.array([b[stat] for b in per_day_bouts])
        features[f"circ_bout_{stat}_mean"] = float(np.mean(vals))
        features[f"circ_bout_{stat}_std"] = float(
            np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return features
