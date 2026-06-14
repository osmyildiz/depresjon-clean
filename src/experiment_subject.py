"""Subject-level LOSO experiment.

One sample = one subject. Outer evaluation is LOSO (n_subjects folds).
Inner CV (5-fold StratifiedKFold) is used to pick num_features only.
SMOTE is disabled in this setting (minority class within a train fold
could be a handful of samples; synthetic interpolation is not meaningful).

The classifier zoo and selector list are inherited from `experiment` to
keep table layout identical to the day-level setting.
"""
from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, roc_auc_score,
)

from .selectors import SELECTORS

# Note: We intentionally do NOT import build_classifiers from .experiment
# here. The day-level pipeline (experiment.py) uses class_weight='balanced'
# baked into each estimator's init. In Regime B we apply sample_weight at
# fit time; sklearn multiplies class_weight by sample_weight, so using
# both would produce a quadratic ('balanced squared') weighting. We
# therefore build classifiers with class_weight disabled and rely on
# sample_weight alone.
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _build_classifiers_subject(random_state=42):
    """Classifier zoo for Regime B: class_weight disabled; class imbalance
    handled via sample_weight. Extended with linear models (LogReg L1/L2/EN),
    histogram gradient boosting (lightgbm-equivalent in sklearn), and a
    stacking ensemble built from the best tree-based learners.
    """
    base = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_state),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_state),
        "Bagging": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=100, n_jobs=-1, random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale",
                   random_state=random_state),
        "XGBoost": _xgb_subject(random_state),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                             random_state=random_state),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),
        "GaussianNB": GaussianNB(),
        # New: linear models well-suited to p>>n
        "LogReg_L1": LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=2000,
            random_state=random_state),
        "LogReg_L2": LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, random_state=random_state),
        "LogReg_EN": LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            max_iter=3000, random_state=random_state),
        # New: histogram gradient boosting (sklearn's lightgbm equivalent)
        "HistGB": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6,
            random_state=random_state),
    }
    # Stacking: top tree-based learners + linear meta-learner
    base["Stacking"] = StackingClassifier(
        estimators=[
            ("bag", base["Bagging"]),
            ("rf", base["RandomForest"]),
            ("xgb", base["XGBoost"]),
            ("hgb", base["HistGB"]),
        ],
        final_estimator=LogisticRegression(
            C=1.0, max_iter=2000, random_state=random_state),
        cv=5, n_jobs=-1, passthrough=False,
    )
    return base


def _xgb_subject(random_state):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", tree_method="hist",
        n_jobs=-1, random_state=random_state)


# Public alias for backwards compatibility with run_subject.py
def build_classifiers(random_state=42):
    return _build_classifiers_subject(random_state)


@dataclass
class SubjectExperimentConfig:
    inner_folds: int = 5
    num_features_grid: tuple = (10, 20, 30, 50, 70, 100, 150, 200)
    selectors: tuple = (
        "lasso", "ridge", "elastic_net", "kbest_chi2",
        "fisher", "info_gain", "rf_importance",
    )
    classifiers: Optional[tuple] = None
    inner_classifier_selector_name: str = "lasso"  # for inner k tuning


def _safe_auc(y_true, y_proba):
    try:
        if y_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, average="weighted", multi_class="ovr")
    except ValueError:
        return np.nan


def _inner_select_k(X_tr, y_tr, selector_name, k_grid, inner_folds, random_state):
    # Inner classifier kept lightweight; logistic regression is natural for
    # p >> n with sparse selection but ExtraTrees stays consistent with the
    # day-level pipeline.
    from sklearn.ensemble import ExtraTreesClassifier
    cheap = ExtraTreesClassifier(
        n_estimators=100, class_weight="balanced",
        n_jobs=-1, random_state=random_state,
    )
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
    best_k, best_score = k_grid[0], -np.inf
    for k in k_grid:
        if k > X_tr.shape[1]:
            continue
        accs = []
        for in_tr, in_va in inner.split(X_tr, y_tr):
            Xa, ya = X_tr[in_tr], y_tr[in_tr]
            Xb, yb = X_tr[in_va], y_tr[in_va]
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


def run_loso(features, y, subject_ids, cfg: SubjectExperimentConfig, random_state=0):
    """LOSO over subjects. Returns long-format results and per-fold selection log.

    Because LOSO produces one test prediction per fold (binary 0/1 hit),
    fold-level "accuracy" is degenerate. We aggregate all out-of-fold
    predictions and compute the metrics once at the end (standard LOSO
    practice). For Wilcoxon-style classifier comparisons we also report
    per-fold correctness so that paired tests remain possible.
    """
    X = features.to_numpy(dtype=np.float32) if hasattr(features, "to_numpy") else np.asarray(features, dtype=np.float32)
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    feature_names = list(features.columns) if hasattr(features, "columns") else None

    classifiers = cfg.classifiers or tuple(build_classifiers().keys())
    loo = LeaveOneOut()

    # Per-(selector, classifier): collect predictions across all folds
    preds = {(s, c): np.full(len(y), np.nan) for s in cfg.selectors for c in classifiers}
    probs = {(s, c): np.full(len(y), np.nan) for s in cfg.selectors for c in classifiers}
    k_per_fold = {s: [] for s in cfg.selectors}
    selected_log = []

    n = len(y)
    for fold_idx, (tr, te) in enumerate(loo.split(X)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        t0 = time.time()
        for sel_name in cfg.selectors:
            best_k, inner_acc = _inner_select_k(
                X_tr, y_tr, sel_name, cfg.num_features_grid,
                cfg.inner_folds, random_state,
            )
            k_per_fold[sel_name].append(best_k)

            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)
            selector = SELECTORS[sel_name](k=best_k, random_state=random_state).fit(X_tr_s, y_tr)
            X_tr_sel = selector.transform(X_tr_s)
            X_te_sel = selector.transform(X_te_s)

            if feature_names is not None:
                selected_log.append({
                    "fold": fold_idx, "selector": sel_name, "k": best_k,
                    "features": ";".join(selector.get_selected_columns(feature_names)),
                })

            clfs = build_classifiers(random_state=random_state)
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", y_tr)
            for clf_name in classifiers:
                clf = clfs[clf_name]
                try:
                    # Try sample_weight first (XGBoost, AdaBoost, GaussianNB,
                    # Bagging, RF, ExtraTrees, DT all accept it). Estimators
                    # that don't (KNN, MLP, QDA) fall through to plain fit.
                    try:
                        clf.fit(X_tr_sel, y_tr, sample_weight=sw)
                    except (TypeError, ValueError):
                        clf.fit(X_tr_sel, y_tr)
                    preds[(sel_name, clf_name)][te[0]] = int(clf.predict(X_te_sel)[0])
                    p = clf.predict_proba(X_te_sel)
                    if p.shape[1] == 2:
                        probs[(sel_name, clf_name)][te[0]] = float(p[0, 1])
                    else:
                        probs[(sel_name, clf_name)][te[0]] = float(np.max(p[0]))
                except Exception:
                    pass
        print(f"[fold {fold_idx+1}/{n}] {time.time()-t0:5.1f}s")

    # Aggregate metrics
    rows = []
    per_fold_correct = {}
    for (sel_name, clf_name), yp in preds.items():
        mask = ~np.isnan(yp)
        if mask.sum() == 0:
            continue
        yt = y[mask]
        yh = yp[mask].astype(int)
        yprob = probs[(sel_name, clf_name)][mask]
        metrics = {
            "selector": sel_name,
            "classifier": clf_name,
            "n_eval": int(mask.sum()),
            "median_k": float(np.median(k_per_fold[sel_name])),
            "accuracy": accuracy_score(yt, yh),
            "precision": precision_score(yt, yh, average="weighted", zero_division=0),
            "recall": recall_score(yt, yh, average="weighted", zero_division=0),
            "f1": f1_score(yt, yh, average="weighted", zero_division=0),
            "kappa": cohen_kappa_score(yt, yh),
        }
        # AUC needs scores; for binary, use class-1 probability.
        try:
            metrics["auc"] = roc_auc_score(yt, yprob)
        except ValueError:
            metrics["auc"] = np.nan
        rows.append(metrics)
        per_fold_correct[(sel_name, clf_name)] = (yh == yt).astype(int)

    summary = pd.DataFrame(rows)

    # Per-fold correctness for paired Wilcoxon (per (selector, classifier))
    correctness = pd.DataFrame({
        f"{s}__{c}": per_fold_correct[(s, c)]
        for (s, c) in per_fold_correct
    })
    selected_log_df = pd.DataFrame(selected_log)
    return summary, correctness, selected_log_df


def selection_frequency_subject(selected_log: pd.DataFrame, top: int = 30) -> pd.DataFrame:
    if len(selected_log) == 0:
        return pd.DataFrame(columns=["count", "rate"])
    rows = []
    total = 0
    for _, r in selected_log.iterrows():
        for f in r["features"].split(";"):
            rows.append(f)
        total += 1
    s = pd.Series(rows).value_counts().head(top).rename("count").to_frame()
    s["rate"] = s["count"] / total
    return s
