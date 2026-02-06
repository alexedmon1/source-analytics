"""Multivariate Pattern Analysis (MVPA) with permutation testing.

Uses a linear SVM with Leave-One-Out Cross-Validation (LOOCV) to classify
groups based on whole-brain spatial patterns. Feature importance is derived
from SVM coefficients. Significance is assessed by permuting group labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MVPAResult:
    """Results from MVPA classification."""

    accuracy: float
    p_value: float
    sensitivity: float
    specificity: float
    auc: float
    accuracy_ci: tuple[float, float]  # 95% CI from permutation null
    feature_weights: np.ndarray  # (n_features,) — mean |coef| across folds
    null_distribution: np.ndarray  # (n_permutations,) — null accuracies
    predictions: np.ndarray  # (n_subjects,) — predicted labels
    true_labels: np.ndarray  # (n_subjects,) — actual labels
    n_permutations: int


def run_mvpa(
    features: np.ndarray,
    labels: np.ndarray,
    classifier: str = "svm_linear",
    cv_method: str = "loocv",
    n_permutations: int = 1000,
    seed: int | None = None,
) -> MVPAResult:
    """Run MVPA classification with permutation testing.

    Parameters
    ----------
    features : ndarray, shape (n_subjects, n_features)
        Feature matrix (e.g., band power at each vertex).
    labels : ndarray, shape (n_subjects,)
        Group labels (0 or 1).
    classifier : str
        Classifier type. Currently only "svm_linear" supported.
    cv_method : str
        Cross-validation method. Currently only "loocv" supported.
    n_permutations : int
        Number of permutations for significance testing.
    seed : int, optional
        Random seed.

    Returns
    -------
    MVPAResult
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n_subjects, n_features = features.shape

    def _run_loocv(feats, labs):
        """Run LOOCV, return accuracy and mean |feature weights|."""
        loo = LeaveOneOut()
        preds = np.zeros(n_subjects, dtype=int)
        all_weights = np.zeros(n_features)
        n_folds = 0

        for train_idx, test_idx in loo.split(feats):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(feats[train_idx])
            X_test = scaler.transform(feats[test_idx])

            clf = SVC(kernel="linear", C=1.0)
            clf.fit(X_train, labs[train_idx])
            preds[test_idx] = clf.predict(X_test)

            all_weights += np.abs(clf.coef_[0])
            n_folds += 1

        acc = float(np.mean(preds == labs))
        weights = all_weights / n_folds
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
        "MVPA observed accuracy: %.1f%%, running %d permutations...",
        accuracy * 100, n_permutations,
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
        "MVPA: accuracy=%.1f%%, p=%.4f, sensitivity=%.1f%%, specificity=%.1f%%",
        accuracy * 100, p_value, sensitivity * 100, specificity * 100,
    )

    return MVPAResult(
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
    )
