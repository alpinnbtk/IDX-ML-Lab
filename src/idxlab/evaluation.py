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