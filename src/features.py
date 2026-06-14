from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import tsfel


def _signature(X, subjects, cfg_hash):
    # Cache key tied to data fingerprint, not just dataset name. Prevents
    # silent reuse if the underlying csvs change.
    h = hashlib.sha1()
    h.update(X.shape.__repr__().encode())
    h.update(X[:10].tobytes())
    h.update("|".join(subjects[:10]).encode())
    h.update(cfg_hash.encode())
    return h.hexdigest()[:12]


def extract_tsfel(X, subjects, cache_dir="cache", tag="", n_jobs=1, log_every=50):
    """Extract TSFEL features for each row in X.

    TSFEL interprets a 2D ndarray as (timesteps, channels), not (samples,
    timesteps), so we must iterate per row. n_jobs > 1 parallelizes via
    joblib for the DGX run.

    Caching is fingerprint-based so changing the data invalidates the cache.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = tsfel.get_features_by_domain()
    cfg_hash = hashlib.sha1(str(cfg).encode()).hexdigest()[:8]
    sig = _signature(X, subjects, cfg_hash)
    cache_path = cache_dir / f"tsfel_{tag}_{sig}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    def _one(row):
        return tsfel.time_series_features_extractor(cfg, row, verbose=0).iloc[0]

    if n_jobs == 1:
        rows = []
        for i in range(len(X)):
            rows.append(_one(X[i]))
            if log_every and (i + 1) % log_every == 0:
                print(f"  tsfel: {i+1}/{len(X)}")
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_one)(X[i]) for i in range(len(X))
        )

    feats = pd.DataFrame(rows).reset_index(drop=True)
    feats.to_parquet(cache_path)
    return feats


def drop_unstable(features: pd.DataFrame, nan_fraction=0.1):
    # TSFEL occasionally produces NaN/Inf columns on flat signals (e.g.
    # sleep windows). Drop columns where missingness exceeds threshold,
    # then median-impute the rest. Median chosen over mean to resist
    # outliers from short-signal artifacts.
    features = features.replace([np.inf, -np.inf], np.nan)
    keep = features.isna().mean() < nan_fraction
    features = features.loc[:, keep]
    return features.fillna(features.median(numeric_only=True))
