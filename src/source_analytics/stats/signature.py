"""Multivariate Pattern Analysis (MVPA) with permutation testing.

A choice of linear classifier with Leave-One-Out Cross-Validation (LOOCV)
classifies groups based on whole-brain spatial patterns. For linear models the
feature importance is derived from the model coefficients (mean |coef| across
folds); non-linear models report accuracy only. Significance is assessed by
permuting group labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# Supported classifiers → a factory returning a FRESH sklearn estimator each
# call (one per LOOCV fold / permutation). The interpretable linear trio
# (svm_linear/logistic/lda) exposes coef_ for feature-importance maps; svm_rbf is
# non-linear (accuracy only, no coef_). All are scaled by a StandardScaler fit on
# the training fold in run_mvpa.
CLASSIFIERS: dict[str, str] = {
    "svm_linear": "Linear SVM",
    "svm_rbf": "RBF SVM",
    "logistic": "Logistic regression",
    "lda": "LDA",
}

# Aliases accepted from config, normalised to a CLASSIFIERS key.
_CLASSIFIER_ALIASES = {
    "svm": "svm_linear", "linear_svm": "svm_linear", "svc": "svm_linear",
    "logreg": "logistic", "logistic_regression": "logistic",
    "lineardiscriminantanalysis": "lda", "linear_discriminant": "lda",
    "rbf_svm": "svm_rbf", "svm_nonlinear": "svm_rbf",
}


def normalize_classifier(name: str) -> str:
    """Resolve a config classifier name/alias to a CLASSIFIERS key."""
    key = (name or "svm_linear").strip().lower()
    key = _CLASSIFIER_ALIASES.get(key, key)
    if key not in CLASSIFIERS:
        raise ValueError(
            f"Unknown classifier '{name}'. Supported: {', '.join(CLASSIFIERS)}"
        )
    return key


def classifier_label(name: str) -> str:
    """Human-readable label for a classifier key (for tables/figures)."""
    return CLASSIFIERS.get(normalize_classifier(name), name)


def make_classifier(name: str):
    """Return a fresh sklearn estimator for a CLASSIFIERS key."""
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    key = normalize_classifier(name)
    if key == "svm_linear":
        return SVC(kernel="linear", C=1.0)
    if key == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale")
    if key == "logistic":
        # L2 is the default penalty; passing penalty="l2" is deprecated in sklearn ≥1.8.
        return LogisticRegression(C=1.0, max_iter=5000)
    if key == "lda":
        return LinearDiscriminantAnalysis()
    raise ValueError(f"Unhandled classifier '{key}'")  # pragma: no cover


def _linear_weights(clf) -> np.ndarray | None:
    """Mean-fold feature weights = |coef_| for a linear model, else None."""
    coef = getattr(clf, "coef_", None)
    if coef is None:
        return None
    return np.abs(np.ravel(coef))


@dataclass
class SignatureResult:
    """Results from a neural-signature classification run."""

    accuracy: float
    p_value: float
    sensitivity: float
    specificity: float
    auc: float
    accuracy_ci: tuple[float, float]  # 95% CI from permutation null
    feature_weights: np.ndarray  # (n_features,) — mean |coef| across folds (NaN if non-linear)
    null_distribution: np.ndarray  # (n_permutations,) — null accuracies
    predictions: np.ndarray  # (n_subjects,) — predicted labels
    true_labels: np.ndarray  # (n_subjects,) — actual labels
    n_permutations: int
    classifier: str = "svm_linear"  # normalised classifier key
    has_weights: bool = True  # False for non-linear models (feature_weights all-NaN)


def run_signature(
    features: np.ndarray,
    labels: np.ndarray,
    classifier: str = "svm_linear",
    cv_method: str = "loocv",
    n_permutations: int = 1000,
    seed: int | None = None,
) -> SignatureResult:
    """Run neural-signature classification with permutation testing.

    Parameters
    ----------
    features : ndarray, shape (n_subjects, n_features)
        Feature matrix (e.g., band power at each vertex).
    labels : ndarray, shape (n_subjects,)
        Group labels (0 or 1).
    classifier : str
        Classifier key or alias (see :data:`CLASSIFIERS`): svm_linear, svm_rbf,
        logistic, lda. Linear models yield feature-importance weights; non-linear
        models (svm_rbf) report accuracy only (feature_weights all-NaN).
    cv_method : str
        Cross-validation method. Currently only "loocv" supported.
    n_permutations : int
        Number of permutations for significance testing.
    seed : int, optional
        Random seed.

    Returns
    -------
    SignatureResult
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    clf_key = normalize_classifier(classifier)
    rng = np.random.default_rng(seed)
    n_subjects, n_features = features.shape

    def _run_loocv(feats, labs):
        """Run LOOCV; return accuracy, predictions, and mean |feature weights|
        (all-NaN for a non-linear classifier that exposes no coef_)."""
        loo = LeaveOneOut()
        preds = np.zeros(n_subjects, dtype=int)
        all_weights = np.zeros(n_features)
        weights_ok = True
        n_folds = 0

        for train_idx, test_idx in loo.split(feats):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(feats[train_idx])
            X_test = scaler.transform(feats[test_idx])

            clf = make_classifier(clf_key)
            clf.fit(X_train, labs[train_idx])
            preds[test_idx] = clf.predict(X_test)

            w = _linear_weights(clf)
            if w is None:
                weights_ok = False
            else:
                all_weights += w
            n_folds += 1

        acc = float(np.mean(preds == labs))
        weights = (all_weights / n_folds) if weights_ok else np.full(n_features, np.nan)
        return acc, preds, weights

    # Observed classification
    accuracy, predictions, feature_weights = _run_loocv(features, labels)

    # Sensitivity and specificity
    pos_mask = labels == 1
    neg_mask = labels == 0
    sensitivity = float(np.mean(predictions[pos_mask] == 1)) if pos_mask.sum() > 0 else 0.0
    specificity = float(np.mean(predictions[neg_mask] == 0)) if neg_mask.sum() > 0 else 0.0

    # AUC (if both classes present in predictions)
    try:
        auc = float(roc_auc_score(labels, predictions))
    except ValueError:
        auc = 0.5

    # Permutation test
    logger.info(
        "Signature [%s] observed accuracy: %.1f%%, running %d permutations...",
        clf_key, accuracy * 100, n_permutations,
    )
    null_accuracies = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_labels = rng.permutation(labels)
        null_acc, _, _ = _run_loocv(features, perm_labels)
        null_accuracies[i] = null_acc

    p_value = float(np.mean(null_accuracies >= accuracy))

    # 95% CI from null distribution
    ci_lower = float(np.percentile(null_accuracies, 2.5))
    ci_upper = float(np.percentile(null_accuracies, 97.5))

    logger.info(
        "Signature [%s]: accuracy=%.1f%%, p=%.4f, sensitivity=%.1f%%, specificity=%.1f%%",
        clf_key, accuracy * 100, p_value, sensitivity * 100, specificity * 100,
    )

    return SignatureResult(
        accuracy=accuracy,
        p_value=p_value,
        sensitivity=sensitivity,
        specificity=specificity,
        auc=auc,
        accuracy_ci=(ci_lower, ci_upper),
        feature_weights=feature_weights,
        null_distribution=null_accuracies,
        predictions=predictions,
        true_labels=labels,
        n_permutations=n_permutations,
        classifier=clf_key,
        has_weights=bool(not np.all(np.isnan(feature_weights))),
    )
