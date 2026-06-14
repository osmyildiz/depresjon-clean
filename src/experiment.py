"""Nested cross-validation with subject-aware splitting.

Layout:
  outer (StratifiedGroupKFold, K_outer) — true generalization estimate
    inner (GroupKFold, K_inner)         — selects num_features only
      pipeline: [optional SMOTE] -> StandardScaler -> Selector(k) -> Clf

Test data never reaches SMOTE, feature selector fit, or scaler fit.
"""
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

from .selectors import SELECTORS


def build_classifiers(random_state=42):
    # class_weight='balanced' wherever supported; this alone handles the
    # 23/32 imbalance without SMOTE. SMOTE remains optional for parity
    # with the prior pipeline.
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        ),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(
                class_weight="balanced", random_state=random_state
            ),
            n_estimators=100, n_jobs=-1, random_state=random_state,
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, C=1.0, gamma="scale",
            class_weight="balanced", random_state=random_state,
        ),
        "XGBoost": _xgb(random_state),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100, random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), max_iter=300,
            random_state=random_state,
        ),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),
        "GaussianNB": GaussianNB(),
    }


def _xgb(random_state):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=random_state,
    )


@dataclass
class ExperimentConfig:
    outer_folds: int = 10
    inner_folds: int = 3
    seeds: tuple = (0, 1, 2)
    num_features_grid: tuple = (20, 30, 50, 70, 90, 110, 130, 150)
    selectors: tuple = (
        "lasso", "ridge", "elastic_net", "kbest_chi2",
        "fisher", "info_gain", "rf_importance",
    )
    use_smote: bool = False
    smote_k_neighbors: int = 5
    classifiers: Optional[tuple] = None    # None = all from build_classifiers


def _smote_train(X, y, k_neighbors, random_state):
    counts = np.bincount(y)
    if counts.min() < k_neighbors + 1:
        # SMOTE needs >= k+1 minority samples; otherwise skip silently
        # rather than crash a single fold.
        return X, y
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    return sm.fit_resample(X, y)


def _score(y_true, y_pred, y_proba):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    try:
        if y_proba.shape[1] == 2:
            out["auc"] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            out["auc"] = roc_auc_score(
                y_true, y_proba, average="weighted", multi_class="ovr"
            )
    except ValueError:
        out["auc"] = np.nan
    return out


def _inner_select_k(X_tr, y_tr, groups_tr, selector_name, k_grid,
                    inner_folds, random_state, use_smote, smote_k):
    # Returns the k that maximizes mean inner accuracy for a fixed cheap
    # classifier. We do NOT tune the classifier here; that would explode
    # the search and is reported as the outer grid instead.
    from .selectors import SELECTORS  # local import to keep module light
    inner = GroupKFold(n_splits=inner_folds)
    cheap = ExtraTreesClassifier(
        n_estimators=100, class_weight="balanced",
        n_jobs=-1, random_state=random_state,
    )
    best_k, best_score = k_grid[0], -np.inf
    for k in k_grid:
        if k > X_tr.shape[1]:
            continue
        accs = []
        for in_tr, in_va in inner.split(X_tr, y_tr, groups_tr):
            Xa, ya = X_tr[in_tr], y_tr[in_tr]
            Xb, yb = X_tr[in_va], y_tr[in_va]
            if use_smote:
                Xa, ya = _smote_train(Xa, ya, smote_k, random_state)
            sc = StandardScaler().fit(Xa)
            Xa_s = sc.transform(Xa)
            Xb_s = sc.transform(Xb)
            sel = SELECTORS[selector_name](k=k, random_state=random_state).fit(Xa_s, ya)
            cheap.fit(sel.transform(Xa_s), ya)
            accs.append(accuracy_score(yb, cheap.predict(sel.transform(Xb_s))))
        mean = float(np.mean(accs)) if accs else -np.inf
        if mean > best_score:
            best_score, best_k = mean, k
    return best_k, best_score


def run(features, y, groups, cfg: ExperimentConfig):
    """Returns a long-format DataFrame: one row per (seed, fold, selector, classifier)."""
    X = features.to_numpy(dtype=np.float32) if hasattr(features, "to_numpy") else np.asarray(features, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)

    feature_names = list(features.columns) if hasattr(features, "columns") else None
    classifiers = cfg.classifiers or tuple(build_classifiers().keys())

    rows = []
    selected_feature_log = []

    for seed in cfg.seeds:
        outer = StratifiedGroupKFold(
            n_splits=cfg.outer_folds, shuffle=True, random_state=seed
        )
        for fold_idx, (tr, te) in enumerate(outer.split(X, y, groups)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            g_tr = groups[tr]
            # Sanity: groups in train and test must be disjoint
            assert set(groups[tr]).isdisjoint(set(groups[te])), \
                "Subject leakage detected between train and test."

            for sel_name in cfg.selectors:
                t0 = time.time()
                best_k, inner_acc = _inner_select_k(
                    X_tr, y_tr, g_tr, sel_name, cfg.num_features_grid,
                    cfg.inner_folds, seed, cfg.use_smote, cfg.smote_k_neighbors,
                )

                # Outer training pass with the inner-selected k.
                X_tr_use, y_tr_use = X_tr, y_tr
                if cfg.use_smote:
                    X_tr_use, y_tr_use = _smote_train(
                        X_tr_use, y_tr_use, cfg.smote_k_neighbors, seed
                    )
                scaler = StandardScaler().fit(X_tr_use)
                X_tr_s = scaler.transform(X_tr_use)
                X_te_s = scaler.transform(X_te)
                selector = SELECTORS[sel_name](k=best_k, random_state=seed).fit(X_tr_s, y_tr_use)
                X_tr_sel = selector.transform(X_tr_s)
                X_te_sel = selector.transform(X_te_s)

                if feature_names is not None:
                    selected_feature_log.append({
                        "seed": seed, "fold": fold_idx, "selector": sel_name,
                        "k": best_k,
                        "features": ";".join(selector.get_selected_columns(feature_names)),
                    })

                clfs = build_classifiers(random_state=seed)
                for clf_name in classifiers:
                    clf = clfs[clf_name]
                    try:
                        clf.fit(X_tr_sel, y_tr_use)
                        y_pred = clf.predict(X_te_sel)
                        y_proba = clf.predict_proba(X_te_sel)
                        metrics = _score(y_te, y_pred, y_proba)
                    except Exception as e:
                        metrics = {k: np.nan for k in
                                   ["accuracy", "precision", "recall", "f1", "kappa", "auc"]}
                        metrics["error"] = str(e)[:120]
                    rows.append({
                        "seed": seed,
                        "fold": fold_idx,
                        "selector": sel_name,
                        "classifier": clf_name,
                        "k": best_k,
                        "inner_cv_acc": inner_acc,
                        "n_train": len(y_tr_use),
                        "n_test": len(y_te),
                        **metrics,
                    })
                print(
                    f"[seed={seed} fold={fold_idx} sel={sel_name:>13s} "
                    f"k={best_k:>3d}] {time.time()-t0:5.1f}s"
                )

    return pd.DataFrame(rows), pd.DataFrame(selected_feature_log)
