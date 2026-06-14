"""Feature selectors as sklearn transformers.

Every selector exposes fit(X, y) and transform(X). fit() must see only
training data; transform() applies the cached column selection to any
matrix. This contract makes leakage structurally impossible inside the
CV pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    chi2,
    RFE,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet


class _ColumnSelector(BaseEstimator, TransformerMixin):
    """Caches column indices after fit; transform is a pure indexing op."""

    def __init__(self, k=50, random_state=42):
        self.k = k
        self.random_state = random_state
        self.selected_ = None

    def transform(self, X):
        if self.selected_ is None:
            raise RuntimeError("Selector not fitted.")
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_].to_numpy()
        return X[:, self.selected_]

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y).transform(X)

    def get_selected_columns(self, feature_names):
        return [feature_names[i] for i in self.selected_]


def _top_k(scores, k):
    k = min(k, len(scores))
    return np.argsort(scores)[-k:]


class LassoSelector(_ColumnSelector):
    def __init__(self, k=50, alpha=0.01, random_state=42):
        super().__init__(k=k, random_state=random_state)
        self.alpha = alpha

    def fit(self, X, y):
        m = Lasso(alpha=self.alpha, max_iter=5000, random_state=self.random_state)
        m.fit(X, y)
        self.selected_ = _top_k(np.abs(m.coef_), self.k)
        return self


class RidgeSelector(_ColumnSelector):
    def __init__(self, k=50, alpha=1.0, random_state=42):
        super().__init__(k=k, random_state=random_state)
        self.alpha = alpha

    def fit(self, X, y):
        m = Ridge(alpha=self.alpha, random_state=self.random_state)
        m.fit(X, y)
        self.selected_ = _top_k(np.abs(m.coef_), self.k)
        return self


class ElasticNetSelector(_ColumnSelector):
    def __init__(self, k=50, alpha=0.1, l1_ratio=0.5, random_state=42):
        super().__init__(k=k, random_state=random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        m = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=5000,
            random_state=self.random_state,
        )
        m.fit(X, y)
        self.selected_ = _top_k(np.abs(m.coef_), self.k)
        return self


class KBestChi2Selector(_ColumnSelector):
    # chi2 requires non-negative inputs, so we MinMax-scale within fit().
    # Scaler is also fit on train only.
    def fit(self, X, y):
        self.scaler_ = MinMaxScaler().fit(X)
        Xs = self.scaler_.transform(X)
        sel = SelectKBest(chi2, k=min(self.k, Xs.shape[1])).fit(Xs, y)
        self.selected_ = np.where(sel.get_support())[0]
        return self

    def transform(self, X):
        if self.selected_ is None:
            raise RuntimeError("Selector not fitted.")
        return self.scaler_.transform(X)[:, self.selected_]


class FisherScoreSelector(_ColumnSelector):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n = X.shape[0]
        mu = X.mean(axis=0)
        num = np.zeros(X.shape[1])
        den = np.zeros(X.shape[1])
        for c in classes:
            mask = y == c
            nc = mask.sum()
            mu_c = X[mask].mean(axis=0)
            var_c = X[mask].var(axis=0) + 1e-12
            num += nc * (mu_c - mu) ** 2
            den += nc * var_c
        scores = num / (den + 1e-12)
        self.selected_ = _top_k(scores, self.k)
        return self


class InformationGainSelector(_ColumnSelector):
    def fit(self, X, y):
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        self.selected_ = _top_k(scores, self.k)
        return self


class RFImportanceSelector(_ColumnSelector):
    def fit(self, X, y):
        rf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=self.random_state
        )
        rf.fit(X, y)
        self.selected_ = _top_k(rf.feature_importances_, self.k)
        return self


class RFESelector(_ColumnSelector):
    # Costly; included for parity with paper but disabled by default.
    def fit(self, X, y):
        base = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=self.random_state
        )
        rfe = RFE(base, n_features_to_select=self.k, step=10).fit(X, y)
        self.selected_ = np.where(rfe.get_support())[0]
        return self


SELECTORS = {
    "lasso": LassoSelector,
    "ridge": RidgeSelector,
    "elastic_net": ElasticNetSelector,
    "kbest_chi2": KBestChi2Selector,
    "fisher": FisherScoreSelector,
    "info_gain": InformationGainSelector,
    "rf_importance": RFImportanceSelector,
    "rfe": RFESelector,
}
