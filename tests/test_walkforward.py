"""Tests for src/idxlab/walkforward.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from idxlab.walkforward import walk_forward_proba

FEATURES = ["f1", "f2"]


def _data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    direction = (f1 + 0.5 * rng.normal(size=n) > 0).astype(float)
    return pd.DataFrame({"f1": f1, "f2": f2, "direction": direction}, index=index)


class _Spy(ClassifierMixin, BaseEstimator):
    """Records the rows each fit() sees; always predicts 0.5."""

    calls: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        _Spy.calls.append((X.index.min(), X.index.max(), len(X)))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        return np.full((len(X), 2), 0.5)


def test_predictions_start_after_first_window_and_gap_and_cover_the_rest():
    data = _data(200)

    proba = walk_forward_proba(
        data, LogisticRegression(), FEATURES, train_size=100, step=20, gap=1
    )

    assert proba.index[0] == data.index[101]
    assert proba.index[-1] == data.index[-1]
    assert proba.index.is_unique
    assert len(proba) == 200 - 101
    assert proba.between(0, 1).all()
    assert proba.name == "proba"


def test_training_data_always_ends_before_the_block_by_gap():
    _Spy.calls = []
    data = _data(200)

    proba = walk_forward_proba(
        data, _Spy(), FEATURES, train_size=100, step=20, gap=3
    )

    block_starts = [data.index[103 + 20 * i] for i in range((200 - 103 + 19) // 20)]
    assert len(_Spy.calls) == len(block_starts)
    for (_, train_last, _), block_start in zip(_Spy.calls, block_starts):
        pos_last = data.index.get_loc(train_last)
        pos_block = data.index.get_loc(block_start)
        assert pos_last == pos_block - 3 - 1   # `gap` rows are skipped
    assert len(proba) == 200 - 103


def test_rolling_window_has_constant_length_and_expanding_grows():
    data = _data(200)

    _Spy.calls = []
    walk_forward_proba(data, _Spy(), FEATURES, train_size=100, step=20, rolling=True)
    assert {n for _, _, n in _Spy.calls} == {100}

    _Spy.calls = []
    walk_forward_proba(data, _Spy(), FEATURES, train_size=100, step=20, rolling=False)
    sizes = [n for _, _, n in _Spy.calls]
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0]
    assert all(first == data.index[0] for first, _, _ in _Spy.calls)


def test_future_rows_cannot_change_earlier_predictions():
    data = _data(200)
    cutoff = 150

    changed = data.copy()
    rng = np.random.default_rng(99)
    changed.iloc[cutoff:, 0] = rng.normal(size=200 - cutoff) * 10   # f1
    changed.iloc[cutoff:, 2] = 1.0 - changed.iloc[cutoff:, 2]       # flip labels

    a = walk_forward_proba(data, LogisticRegression(), FEATURES, train_size=100, step=20)
    b = walk_forward_proba(changed, LogisticRegression(), FEATURES, train_size=100, step=20)

    pd.testing.assert_series_equal(a.iloc[: cutoff - 101], b.iloc[: cutoff - 101])
    assert not a.iloc[cutoff - 101 :].equals(b.iloc[cutoff - 101 :])


def test_rolling_window_forgets_old_data_but_expanding_does_not():
    data = _data(300)
    changed = data.copy()
    changed.iloc[:20, 2] = 1.0 - changed.iloc[:20, 2]               # corrupt old labels
    kwargs = dict(model=LogisticRegression(), feature_cols=FEATURES, train_size=60, step=20, gap=1)

    roll_a = walk_forward_proba(data, rolling=True, **kwargs)
    roll_b = walk_forward_proba(changed, rolling=True, **kwargs)
    exp_a = walk_forward_proba(data, rolling=False, **kwargs)
    exp_b = walk_forward_proba(changed, rolling=False, **kwargs)

    late = roll_a.index[-40:]                                       # windows no longer touch rows 0-19
    pd.testing.assert_series_equal(roll_a.loc[late], roll_b.loc[late])
    assert not exp_a.loc[late].equals(exp_b.loc[late])


def test_model_passed_in_is_not_fitted_or_modified():
    model = LogisticRegression()
    walk_forward_proba(_data(200), model, FEATURES, train_size=100)
    assert not hasattr(model, "coef_")


def test_one_class_window_falls_back_to_that_class():
    data = _data(200)
    data["direction"] = 1.0

    proba = walk_forward_proba(data, LogisticRegression(), FEATURES, train_size=100)

    assert (proba == 1.0).all()


def test_walk_forward_beats_chance_on_a_learnable_signal():
    from sklearn.metrics import roc_auc_score

    data = _data(600)
    proba = walk_forward_proba(data, LogisticRegression(), FEATURES, train_size=200, step=20)
    auc = roc_auc_score(data.loc[proba.index, "direction"], proba)
    assert auc > 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step": 0},
        {"gap": -1},
        {"train_size": 1},
        {"train_size": 500},          # more than the data has
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        walk_forward_proba(_data(200), LogisticRegression(), FEATURES, **kwargs)


def test_target_in_features_raises():
    with pytest.raises(ValueError):
        walk_forward_proba(_data(200), LogisticRegression(), FEATURES + ["direction"])


def test_unsorted_index_raises():
    with pytest.raises(ValueError):
        walk_forward_proba(_data(200).iloc[::-1], LogisticRegression(), FEATURES)