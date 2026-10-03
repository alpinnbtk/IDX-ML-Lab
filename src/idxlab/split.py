"""Train/validation/test splitting for IDX-ML-Lab.

Splits are strictly time-based (walk-forward): train on the earliest
period, validate on the middle period, test on the most recent
period. Never shuffled — shuffling before splitting would let the
model "see the future" during training (a row from next month ending
up in the training set while a row from last month ends up in test),
which silently inflates every metric computed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import FeaturePipeline
from .labels import build_labels


def _assert_chronological(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if any split's dates overlap or go out of order.

    This is a safety net, not just documentation: a bug that lets
    even one future row leak into training would otherwise pass
    silently and only show up later as suspiciously good backtest
    results.
    """
    if not train.empty and not val.empty:
        if train.index.max() >= val.index.min():
            raise ValueError("train and val overlap or are out of order")
    if not val.empty and not test.empty:
        if val.index.max() >= test.index.min():
            raise ValueError("val and test overlap or are out of order")
    elif not train.empty and not test.empty and val.empty:
        if train.index.max() >= test.index.min():
            raise ValueError("train and test overlap or are out of order")


@dataclass
class SplitResult:
    """Container for a chronological train/val/test split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __post_init__(self) -> None:
        _assert_chronological(self.train, self.val, self.test)


def time_based_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
) -> SplitResult:
    """Split a date-indexed DataFrame chronologically, no shuffling.

    The DataFrame must already be sorted by date ascending (true for
    everything produced earlier in this pipeline). The first
    `train_size` fraction of rows becomes train, the next `val_size`
    fraction becomes val, and everything remaining becomes test.

    Parameters
    ----------
    df : pd.DataFrame
        Date-indexed data, sorted ascending.
    train_size : float
        Fraction of rows for training, e.g. 0.7 = 70%.
    val_size : float
        Fraction of rows for validation, e.g. 0.15 = 15%. The
        remaining fraction (1 - train_size - val_size) becomes test.

    Returns
    -------
    SplitResult
        .train, .val, .test — each a DataFrame, in chronological
        order with no gaps or overlaps between them.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("df must be sorted by date ascending before splitting")
    if train_size <= 0 or val_size < 0 or train_size + val_size >= 1:
        raise ValueError("train_size + val_size must be less than 1, both non-negative")

    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return SplitResult(train=train, val=val, test=test)


def build_dataset(
    ohlcv: pd.DataFrame,
    feature_pipeline: FeaturePipeline | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Build a model-ready dataset: features + labels, NaN rows dropped.

    Formalizes the feature/label-combining pattern from Day 9 into a
    single reusable function, so every caller (notebooks, training
    scripts, the API later) builds datasets the same way.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Cleaned OHLCV data for one ticker (e.g. clean_ticker_data output).
    feature_pipeline : FeaturePipeline, optional
        Defaults to FeaturePipeline() with its default windows.
    horizon : int
        Days ahead the label looks (passed to build_labels).

    Returns
    -------
    pd.DataFrame
        Feature columns + "forward_return" + "direction", with every
        row fully populated (leading rows from indicator warmup and
        trailing rows from label horizon both dropped).
    """
    pipeline = feature_pipeline or FeaturePipeline()

    features = pipeline.transform(ohlcv)
    labels = build_labels(ohlcv, horizon=horizon)

    combined = features.join(labels)
    return combined.dropna()