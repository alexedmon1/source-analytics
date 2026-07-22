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
    # Balanced accuracy = (sensitivity + specificity)/2. This is the honest
    # primary metric for the unequal-n contrasts (treated groups are n=8-9 vs
    # n=17-18 vehicle), where raw accuracy is inflated by the majority class.
    balanced_accuracy: float = float("nan")
    balanced_p_value: float = float("nan")
    balanced_accuracy_ci: tuple[float, float] = (float("nan"), float("nan"))


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

    def _decision_score(clf, X_test) -> float:
        """Continuous score for the held-out subject, oriented so that larger =
        more class-1. Needed for a real ROC AUC; hard 0/1 labels collapse the ROC
        to a single operating point (AUC then just re-states balanced accuracy)."""
        if hasattr(clf, "decision_function"):
            return float(np.ravel(clf.decision_function(X_test))[0])
        if hasattr(clf, "predict_proba"):
            return float(clf.predict_proba(X_test)[0, 1])
        return float(np.ravel(clf.predict(X_test))[0])

    def _balanced(preds, labs) -> float:
        pos, neg = labs == 1, labs == 0
        sens = float(np.mean(preds[pos] == 1)) if pos.sum() else 0.0
        spec = float(np.mean(preds[neg] == 0)) if neg.sum() else 0.0
        return (sens + spec) / 2.0

    def _run_loocv(feats, labs):
        """Run LOOCV; return accuracy, balanced accuracy, predictions, continuous
        decision scores, and mean |feature weights| (all-NaN for a non-linear
        classifier that exposes no coef_)."""
        loo = LeaveOneOut()
        preds = np.zeros(n_subjects, dtype=int)
        scores = np.zeros(n_subjects, dtype=float)
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
            scores[test_idx] = _decision_score(clf, X_test)

            w = _linear_weights(clf)
            if w is None:
                weights_ok = False
            else:
                all_weights += w
            n_folds += 1

        acc = float(np.mean(preds == labs))
        weights = (all_weights / n_folds) if weights_ok else np.full(n_features, np.nan)
        return acc, _balanced(preds, labs), preds, scores, weights

    # Observed classification
    (accuracy, balanced_accuracy, predictions,
     decision_scores, feature_weights) = _run_loocv(features, labels)

    # Sensitivity and specificity
    pos_mask = labels == 1
    neg_mask = labels == 0
    sensitivity = float(np.mean(predictions[pos_mask] == 1)) if pos_mask.sum() > 0 else 0.0
    specificity = float(np.mean(predictions[neg_mask] == 0)) if neg_mask.sum() > 0 else 0.0

    # AUC from the CONTINUOUS cross-validated decision scores (not hard labels).
    try:
        auc = float(roc_auc_score(labels, decision_scores))
    except ValueError:
        auc = 0.5

    # Permutation test
    logger.info(
        "Signature [%s] observed accuracy: %.1f%%, running %d permutations...",
        clf_key, accuracy * 100, n_permutations,
    )
    null_accuracies = np.zeros(n_permutations)
    null_balanced = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_labels = rng.permutation(labels)
        null_acc, null_bal, _, _, _ = _run_loocv(features, perm_labels)
        null_accuracies[i] = null_acc
        null_balanced[i] = null_bal

    # Add-one (Phipson & Smyth 2010) — the observed labelling is one of the
    # permutations, so an exact 0 is not attainable and `mean(null >= obs)` is
    # anti-conservative. p is now bounded below by 1/(n_perm+1).
    p_value = float((1 + np.sum(null_accuracies >= accuracy)) / (n_permutations + 1))
    balanced_p_value = float(
        (1 + np.sum(null_balanced >= balanced_accuracy)) / (n_permutations + 1))

    # 95% CI from null distribution
    ci_lower = float(np.percentile(null_accuracies, 2.5))
    ci_upper = float(np.percentile(null_accuracies, 97.5))
    bal_ci = (float(np.percentile(null_balanced, 2.5)),
              float(np.percentile(null_balanced, 97.5)))

    logger.info(
        "Signature [%s]: accuracy=%.1f%% (balanced %.1f%%, p=%.4f), auc=%.3f, "
        "p=%.4f, sensitivity=%.1f%%, specificity=%.1f%%",
        clf_key, accuracy * 100, balanced_accuracy * 100, balanced_p_value,
        auc, p_value, sensitivity * 100, specificity * 100,
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
        balanced_accuracy=balanced_accuracy,
        balanced_p_value=balanced_p_value,
        balanced_accuracy_ci=bal_ci,
    )
