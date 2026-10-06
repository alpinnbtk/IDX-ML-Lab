"""Tests for the classifier evaluation harness (src/idxlab/evaluation.py)."""

import numpy as np
import pytest

from idxlab.evaluation import evaluate_classifier

# Small enough to verify by hand:
# TP=2, FP=1, FN=1, TN=2  ->  accuracy 4/6, precision 2/3, recall 2/3.
# ROC-AUC: 8 of the 9 (positive, negative) pairs are ranked correctly.
Y_TRUE = np.array([1, 0, 1, 1, 0, 0])
Y_PRED = np.array([1, 0, 0, 1, 0, 1])
Y_PROBA = np.array([0.9, 0.2, 0.4, 0.8, 0.1, 0.6])


def test_returns_expected_keys():
    result = evaluate_classifier(Y_TRUE, Y_PRED, Y_PROBA)

    assert set(result) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_metrics_match_hand_calculation():
    result = evaluate_classifier(Y_TRUE, Y_PRED, Y_PROBA)

    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["roc_auc"] == pytest.approx(8 / 9)


def test_no_positive_predictions_gives_zero_precision_not_error():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.zeros(4, dtype=int)  # model never predicts "up"
    y_proba = np.array([0.4, 0.3, 0.2, 0.1])

    result = evaluate_classifier(y_true, y_pred, y_proba)

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0