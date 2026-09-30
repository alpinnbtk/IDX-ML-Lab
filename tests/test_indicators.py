"""Tests for technical indicators (src/idxlab/indicators.py)."""

import numpy as np
import pandas as pd

from idxlab.indicators import bollinger_bands, ema, macd, rsi, sma


def _price_series(values: list[float]) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates, name="Close")


def test_sma_matches_manual_mean():
    series = _price_series([10, 20, 30, 40, 50])

    result = sma(series, window=3)

    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 20  # mean(10, 20, 30)
    assert result.iloc[4] == 40  # mean(30, 40, 50)


def test_ema_reacts_faster_than_sma_after_price_jump():
    # Flat at 100 for a while, then a sudden jump to 200.
    series = _price_series([100] * 10 + [200] * 5)

    sma_result = sma(series, window=10)
    ema_result = ema(series, span=10)

    # Right after the jump, EMA should have moved closer to 200
    # than SMA, since EMA weights recent prices more heavily.
    idx = 10  # first day at the new price level
    assert ema_result.iloc[idx] > sma_result.iloc[idx]


def test_rsi_is_100_for_strictly_increasing_prices():
    series = _price_series([100 + i for i in range(30)])  # always going up

    result = rsi(series, window=14)

    assert result.iloc[-1] == 100


def test_rsi_is_0_for_strictly_decreasing_prices():
    series = _price_series([100 - i for i in range(30)])  # always going down

    result = rsi(series, window=14)

    assert result.iloc[-1] == 0


def test_macd_returns_expected_columns():
    series = _price_series(list(range(100, 150)))

    result = macd(series, fast=12, slow=26, signal=9)

    assert list(result.columns) == ["macd", "signal", "histogram"]
    # histogram should always equal macd - signal, by definition
    pd.testing.assert_series_equal(
        result["histogram"],
        result["macd"] - result["signal"],
        check_names=False,
    )


def test_bollinger_middle_band_equals_sma():
    series = _price_series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])

    bands = bollinger_bands(series, window=5)
    expected_middle = sma(series, window=5)

    pd.testing.assert_series_equal(
        bands["bb_middle"], expected_middle, check_names=False
    )


def test_bollinger_upper_band_above_lower_band():
    series = _price_series([10, 12, 11, 13, 15, 14, 16, 18, 17, 19])

    bands = bollinger_bands(series, window=5)
    valid = bands.dropna()

    assert (valid["bb_upper"] > valid["bb_lower"]).all()