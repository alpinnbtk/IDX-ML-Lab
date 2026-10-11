"""Evaluation helpers for classification models in IDX-ML-Lab.

One function that every model in the project (logistic regression now;
random forest, XGBoost, and LSTM later) is scored with, so results are
always directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Score a binary classifier. Class 1 ("price goes up") is positive.

    Parameters
    ----------
    y_true : array-like
        Actual labels (0/1).
    y_pred : array-like
        Predicted labels (0/1).
    y_proba : array-like
        Predicted probability of class 1 (needed for ROC-AUC).

    Returns
    -------
    dict[str, float]
        Keys: accuracy, precision, recall, f1, roc_auc.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),   # TODO 1
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),  # TODO 2
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),     # TODO 3
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),         # TODO 4
        "roc_auc": float(roc_auc_score(y_true, y_proba)),    # TODO 5
    }

def bootstrap_auc_ci(
    y_true: pd.Series | np.ndarray,
    y_proba: pd.Series | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ROC-AUC.

    Rows are resampled with replacement; resamples containing only one
    class (AUC undefined) are discarded. Rows are treated as independent,
    which ignores time dependence, so the interval is a rough guide.

    Returns
    -------
    tuple[float, float]
        (lower, upper) bounds of the (1 - alpha) interval.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_proba)
    rng = np.random.default_rng(seed)
    n = len(y)

    aucs: list[float] = []
    while len(aucs) < n_boot:
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], p[idx])))

    lower, upper = np.quantile(aucs, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def paired_bootstrap_auc_diff(
    y_true: pd.Series | np.ndarray,
    proba_a: pd.Series | np.ndarray,
    proba_b: pd.Series | np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap interval for AUC(a) - AUC(b) on the same rows.

    Both models are scored on the same resampled rows each time, so the
    interval reflects the difference between them, not the noise of the
    sample itself.

    Returns
    -------
    tuple[float, float, float]
        (observed difference, lower bound, upper bound).
    """
    y = np.asarray(y_true)
    a = np.asarray(proba_a)
    b = np.asarray(proba_b)
    rng = np.random.default_rng(seed)
    n = len(y)

    observed = float(roc_auc_score(y, a) - roc_auc_score(y, b))

    diffs: list[float] = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(
            float(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        )

    lower, upper = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return observed, float(lower), float(upper)