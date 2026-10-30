"""Sliding-window sequence datasets for recurrent and other sequence models.

A sequence model does not look at one row at a time. For a prediction made
at the close of day ``t`` it sees the last ``window`` days of features
(rows ``t-window+1`` .. ``t``) and tries to predict the label of day ``t``
(the next-day direction). This module turns a feature table into such
windows, keeps the dates attached, scales features with statistics from the
training period only, and hands the result to Keras.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from idxlab.split import SplitResult


@dataclass(frozen=True)
class WindowedData:
    """Windows ready for a sequence model.

    Attributes
    ----------
    X : np.ndarray
        Shape ``(n_samples, window, n_features)``, float32.
    y : np.ndarray
        Shape ``(n_samples,)``, float32 0/1 labels.
    dates : pd.Index
        Date of the *last* row of each window, which is also the date the
        label belongs to. Same length as ``X``.
    """

    X: np.ndarray
    y: np.ndarray
    dates: pd.Index

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError("X must have shape (n_samples, window, n_features)")
        if not (len(self.X) == len(self.y) == len(self.dates)):
            raise ValueError("X, y and dates must have the same length")

    def __len__(self) -> int:
        return len(self.X)


@dataclass(frozen=True)
class SequenceSplits:
    """Train / validation / test windows plus the fitted scaler."""

    train: WindowedData
    val: WindowedData
    test: WindowedData
    scaler: StandardScaler
    feature_cols: list[str]


def make_windows(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "direction",
    window: int = 20,
) -> WindowedData:
    """Cut a date-indexed table into overlapping windows of ``window`` rows.

    Window ``i`` holds rows ``i .. i+window-1``; its label and date are those
    of the last row. Windows only look backwards, so a window never contains
    a row later than its own date.

    Parameters
    ----------
    frame : pd.DataFrame
        Sorted by date ascending, no NaN in the used columns.
    feature_cols : list[str]
        Input columns (the target must not be one of them).
    target_col : str
        0/1 label column.
    window : int
        Number of consecutive rows per sample (at least 1).

    Returns
    -------
    WindowedData
        ``len(frame) - window + 1`` samples.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(frame) < window:
        raise ValueError(f"need at least window={window} rows, got {len(frame)}")
    if target_col in feature_cols:
        raise ValueError("target_col must not be one of feature_cols")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("frame must be sorted by date ascending")
    if frame[[*feature_cols, target_col]].isna().any().any():
        raise ValueError("frame contains NaN in feature or target columns")

    values = frame[feature_cols].to_numpy(dtype=np.float32)
    targets = frame[target_col].to_numpy(dtype=np.float32)

    windows = []
    for end in range(window, len(frame) + 1):               
        windows.append(values[end - window:end])         
    X = np.stack(windows)
    y = targets[window - 1:]                            
    dates = frame.index[window - 1:]                  

    return WindowedData(X=X, y=y, dates=dates)


def select_by_dates(data: WindowedData, dates: pd.Index) -> WindowedData:
    """Keep only the samples whose own date (last row) is in ``dates``."""
    mask = data.dates.isin(dates)            
    return WindowedData(X=data.X[mask], y=data.y[mask], dates=data.dates[mask])


def scale_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    fit_index: pd.Index,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize features using mean and std from the rows in ``fit_index``.

    Pass the *training* dates as ``fit_index``. The statistics are then applied
    to every row, so validation and test data never influence them.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        A scaled copy of ``frame`` and the fitted scaler (keep it: serving
        later needs the same transformation).
    """
    scaler = StandardScaler()
    scaler.fit(frame.loc[fit_index, feature_cols])    

    scaled = frame.copy()
    scaled[feature_cols] = scaler.transform(frame[feature_cols])
    return scaled, scaler


def build_sequence_splits(
    dataset: pd.DataFrame,
    split: SplitResult,
    feature_cols: list[str],
    window: int = 20,
    target_col: str = "direction",
) -> SequenceSplits:
    """Scale on train, window the whole table, then assign samples by date.

    Windows are built over the full table and assigned to train / validation
    / test by the date of their last row. A validation window may therefore
    start in the training period (those are past features, known at
    prediction time), but a training window never contains later data. The
    first ``window - 1`` training rows only serve as history and produce no
    sample of their own.

    Parameters
    ----------
    dataset : pd.DataFrame
        Features and label for train + validation + test (output of
        ``build_dataset``).
    split : SplitResult
        Chronological split of ``dataset``.
    feature_cols : list[str]
        Input columns.
    window : int
        Rows per sample.
    target_col : str
        0/1 label column.
    """
    scaled, scaler = scale_features(dataset, feature_cols, split.train.index)
    windows = make_windows(scaled, feature_cols, target_col, window)

    return SequenceSplits(
        train=select_by_dates(windows, split.train.index),
        val=select_by_dates(windows, split.val.index),
        test=select_by_dates(windows, split.test.index),
        scaler=scaler,
        feature_cols=list(feature_cols),
    )


def to_tf_dataset(
    data: WindowedData,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
) -> Any:
    """Wrap windows in a ``tf.data.Dataset`` of ``(X_batch, y_batch)``.

    Shuffle only the training set. Validation and test must keep their date
    order so predictions line up with dates for the backtest.
    """
    import tensorflow as tf  # imported here so the rest of the module needs no TensorFlow

    ds = tf.data.Dataset.from_tensor_slices((data.X, data.y))  
    if shuffle:
        ds = ds.shuffle(buffer_size=len(data), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)