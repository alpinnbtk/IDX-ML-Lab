"""Tests for the feature engineering pipeline (src/idxlab/features.py)."""

import numpy as np
import pandas as pd

from idxlab.features import FeaturePipeline
from idxlab.indicators import momentum, sma


def _fake_ohlcv(n_days: int = 60) -> pd.DataFrame:
    """Fake OHLCV data, long enough for every indicator's warmup window
    (MACD's slow=26 is the longest) to produce valid rows near the end."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(seed=42)
    close = 100 + np.cumsum(rng.normal(0, 1, n_days))
    high = close + rng.uniform(0.5, 2, n_days)
    low = close - rng.uniform(0.5, 2, n_days)
    volume = rng.integers(1000, 5000, n_days)

    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_transform_returns_expected_columns():
    df = _fake_ohlcv()

    features = FeaturePipeline().transform(df)

    expected = {
        "sma", "ema", "rsi", "macd", "macd_signal", "macd_histogram",
        "bb_upper", "bb_lower", "bb_width", "atr", "obv", "momentum",
    }
    assert set(features.columns) == expected


def test_transform_preserves_date_index():
    df = _fake_ohlcv()

    features = FeaturePipeline().transform(df)

    pd.testing.assert_index_equal(features.index, df.index)


def test_transform_has_valid_rows_after_longest_warmup():
    df = _fake_ohlcv(n_days=60)

    features = FeaturePipeline().transform(df)

    # 60 days is well past MACD's slow=26 warmup, so the last row
    # should be fully populated, no NaN left.
    assert features.iloc[-1].notna().all()


def test_custom_windows_are_respected():
    df = _fake_ohlcv(n_days=30)

    pipeline = FeaturePipeline(sma_window=5, momentum_window=3)
    features = pipeline.transform(df)

    pd.testing.assert_series_equal(
        features["sma"], sma(df["Close"], window=5), check_names=False
    )
    pd.testing.assert_series_equal(
        features["momentum"], momentum(df["Close"], window=3), check_names=False
    )


def test_transform_many_applies_to_every_ticker():
    data = {"AAA.JK": _fake_ohlcv(), "BBB.JK": _fake_ohlcv()}

    result = FeaturePipeline().transform_many(data)

    assert set(result.keys()) == {"AAA.JK", "BBB.JK"}
    assert all(isinstance(df, pd.DataFrame) for df in result.values())