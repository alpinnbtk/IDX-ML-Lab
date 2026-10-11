"""Tests for the classifier evaluation harness (src/idxlab/evaluation.py)."""

import numpy as np
import pytest

from idxlab.evaluation import evaluate_classifier, bootstrap_auc_ci, paired_bootstrap_auc_diff

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

def _noisy_scores(n: int = 300, strength: float = 0.5, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return y, y * strength + rng.normal(size=n)


def test_bootstrap_ci_is_reproducible_and_ordered():
    y, p = _noisy_scores()

    first = bootstrap_auc_ci(y, p, n_boot=200, seed=1)
    second = bootstrap_auc_ci(y, p, n_boot=200, seed=1)

    assert first == second
    assert first[0] < first[1]


def test_bootstrap_ci_contains_observed_auc():
    from sklearn.metrics import roc_auc_score

    y, p = _noisy_scores()

    lo, hi = bootstrap_auc_ci(y, p, n_boot=500)

    assert lo <= roc_auc_score(y, p) <= hi


def test_paired_diff_is_zero_for_identical_models():
    y, p = _noisy_scores()

    diff, lo, hi = paired_bootstrap_auc_diff(y, p, p, n_boot=200)

    assert diff == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_paired_diff_detects_clearly_better_model():
    y, good = _noisy_scores(strength=3.0, seed=0)
    _, bad = _noisy_scores(strength=0.0, seed=5)  # same seed-independent noise, no signal

    diff, lo, hi = paired_bootstrap_auc_diff(y, good, bad, n_boot=500)

    assert diff > 0.2
    assert lo > 0