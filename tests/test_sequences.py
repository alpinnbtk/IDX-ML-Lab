"""Tests for src/idxlab/sequences.py."""

import numpy as np
import pandas as pd
import pytest

from idxlab.sequences import (
    WindowedData,
    build_sequence_splits,
    make_windows,
    scale_features,
    select_by_dates,
    to_tf_dataset,
)
from idxlab.split import time_based_split

FEATURES = ["f1", "f2"]


def _frame(n: int = 30) -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {
            "f1": np.arange(n, dtype=float),
            "f2": np.arange(n, dtype=float) * 10,
            "direction": (np.arange(n) % 2).astype(float),
        },
        index=index,
    )


def test_shapes_and_dtypes():
    frame = _frame(30)

    data = make_windows(frame, FEATURES, window=5)

    assert data.X.shape == (26, 5, 2)
    assert data.y.shape == (26,)
    assert len(data.dates) == 26
    assert data.X.dtype == np.float32 and data.y.dtype == np.float32


def test_first_and_last_window_contents():
    frame = _frame(30)

    data = make_windows(frame, FEATURES, window=5)

    np.testing.assert_array_equal(data.X[0], frame[FEATURES].to_numpy()[0:5])
    np.testing.assert_array_equal(data.X[-1], frame[FEATURES].to_numpy()[-5:])
    assert data.dates[0] == frame.index[4]
    assert data.dates[-1] == frame.index[-1]


def test_every_window_is_aligned_with_its_own_date_and_label():
    frame = _frame(30)

    data = make_windows(frame, FEATURES, window=7)

    for i, date in enumerate(data.dates):
        np.testing.assert_array_equal(data.X[i, -1], frame.loc[date, FEATURES].to_numpy())
        assert data.y[i] == frame.loc[date, "direction"]
        assert (data.X[i, :, 0] == np.arange(7) + i).all()    # consecutive rows


def test_window_of_one_equals_plain_rows():
    frame = _frame(10)

    data = make_windows(frame, FEATURES, window=1)

    np.testing.assert_array_equal(data.X[:, 0, :], frame[FEATURES].to_numpy())
    np.testing.assert_array_equal(data.y, frame["direction"].to_numpy())


def test_window_equal_to_length_gives_one_sample():
    data = make_windows(_frame(8), FEATURES, window=8)
    assert len(data) == 1


def test_future_rows_cannot_change_earlier_windows():
    frame = _frame(40)
    changed = frame.copy()
    changed.iloc[25:, 0] = -999.0
    changed.iloc[25:, 2] = 1.0 - changed.iloc[25:, 2]

    a = make_windows(frame, FEATURES, window=6)
    b = make_windows(changed, FEATURES, window=6)

    before = a.dates < frame.index[25]
    np.testing.assert_array_equal(a.X[before], b.X[before])
    np.testing.assert_array_equal(a.y[before], b.y[before])
    assert not np.array_equal(a.X[~before], b.X[~before])


@pytest.mark.parametrize("window", [0, -3, 99])
def test_invalid_window_raises(window):
    with pytest.raises(ValueError):
        make_windows(_frame(30), FEATURES, window=window)


def test_nan_target_in_features_and_unsorted_raise():
    frame = _frame(30)
    with_nan = frame.copy()
    with_nan.iloc[3, 0] = np.nan
    with pytest.raises(ValueError):
        make_windows(with_nan, FEATURES, window=5)
    with pytest.raises(ValueError):
        make_windows(frame, FEATURES + ["direction"], window=5)
    with pytest.raises(ValueError):
        make_windows(frame.iloc[::-1], FEATURES, window=5)


def test_select_by_dates_keeps_matching_samples_in_order():
    frame = _frame(30)
    data = make_windows(frame, FEATURES, window=5)

    part = select_by_dates(data, frame.index[10:20])

    assert list(part.dates) == list(frame.index[10:20])
    assert len(part) == 10
    np.testing.assert_array_equal(part.X, data.X[data.dates.isin(frame.index[10:20])])
    assert len(select_by_dates(data, pd.DatetimeIndex([]))) == 0


def test_windowed_data_validates_shapes():
    with pytest.raises(ValueError):
        WindowedData(X=np.zeros((3, 4)), y=np.zeros(3), dates=pd.RangeIndex(3))
    with pytest.raises(ValueError):
        WindowedData(X=np.zeros((3, 4, 2)), y=np.zeros(2), dates=pd.RangeIndex(3))


def test_scale_features_uses_only_fit_rows():
    frame = _frame(40)
    fit_index = frame.index[:25]

    scaled, scaler = scale_features(frame, FEATURES, fit_index)

    np.testing.assert_allclose(scaler.mean_, frame.loc[fit_index, FEATURES].mean().to_numpy())
    assert scaled.loc[fit_index, FEATURES].mean().abs().max() < 1e-9
    assert scaled.loc[fit_index, FEATURES].std(ddof=0).sub(1).abs().max() < 1e-9
    assert (scaled["direction"] == frame["direction"]).all()          # target untouched
    assert (frame["f1"] == np.arange(40)).all()                       # input not modified


def test_scaling_is_not_influenced_by_rows_after_the_fit_period():
    frame = _frame(40)
    changed = frame.copy()
    changed.iloc[25:, 0] = 1e6
    fit_index = frame.index[:25]

    a, _ = scale_features(frame, FEATURES, fit_index)
    b, _ = scale_features(changed, FEATURES, fit_index)

    pd.testing.assert_frame_equal(a.loc[fit_index], b.loc[fit_index])


def test_build_sequence_splits_counts_dates_and_no_overlap():
    frame = _frame(100)
    split = time_based_split(frame)                      # 70 / 15 / 15

    seq = build_sequence_splits(frame, split, FEATURES, window=10)

    assert len(seq.train) == 70 - 9                      # first 9 rows are history only
    assert len(seq.val) == 15
    assert len(seq.test) == 15
    assert seq.train.dates.isin(split.train.index).all()
    assert seq.val.dates.isin(split.val.index).all()
    assert seq.test.dates.isin(split.test.index).all()
    assert not set(seq.train.dates) & set(seq.val.dates)
    assert not set(seq.val.dates) & set(seq.test.dates)
    assert seq.train.X.shape[1:] == (10, 2)
    assert seq.feature_cols == FEATURES


def test_build_sequence_splits_scaler_comes_from_train_only():
    frame = _frame(100)
    split = time_based_split(frame)

    seq = build_sequence_splits(frame, split, FEATURES, window=10)

    np.testing.assert_allclose(seq.scaler.mean_, split.train[FEATURES].mean().to_numpy())
    last_train_rows = seq.train.X[:, -1, :]               # last row of each window = that sample's own day
    assert np.abs(last_train_rows.mean(axis=0)).max() < 0.5


def test_validation_window_may_reach_back_into_training_period_but_not_the_reverse():
    frame = _frame(100)
    split = time_based_split(frame)
    seq = build_sequence_splits(frame, split, FEATURES, window=10)
    scaled, _ = scale_features(frame, FEATURES, split.train.index)

    first_val_row = scaled.loc[split.train.index[-9:].append(split.val.index[:1]), FEATURES].to_numpy()
    np.testing.assert_allclose(seq.val.X[0], first_val_row, rtol=1e-5)
    assert seq.train.dates.max() == split.train.index[-1]
    assert seq.train.dates.max() < seq.val.dates.min()


def test_to_tf_dataset_batches_and_keeps_order_without_shuffle():
    pytest.importorskip("tensorflow")
    data = make_windows(_frame(30), FEATURES, window=5)        # 26 samples

    ds = to_tf_dataset(data, batch_size=8)
    batches = list(ds.as_numpy_iterator())

    assert [len(x) for x, _ in batches] == [8, 8, 8, 2]
    assert batches[0][0].shape == (8, 5, 2)
    np.testing.assert_array_equal(np.concatenate([x for x, _ in batches]), data.X)
    np.testing.assert_array_equal(np.concatenate([y for _, y in batches]), data.y)


def test_to_tf_dataset_shuffle_is_reproducible_and_keeps_all_samples():
    pytest.importorskip("tensorflow")
    data = make_windows(_frame(30), FEATURES, window=5)

    def first_epoch(seed: int) -> np.ndarray:
        ds = to_tf_dataset(data, batch_size=26, shuffle=True, seed=seed)
        return next(iter(ds.as_numpy_iterator()))[0]

    a, b = first_epoch(1), first_epoch(1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, data.X)                       # actually shuffled
    assert sorted(a[:, 0, 0]) == sorted(data.X[:, 0, 0])       # same samples