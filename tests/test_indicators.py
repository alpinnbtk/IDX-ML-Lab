"""Tests for technical indicators (src/idxlab/indicators.py)."""

import pytest  # tambahkan baris ini
import numpy as np
import pandas as pd

from idxlab.indicators import atr, bollinger_bands, ema, macd, momentum, obv, rsi, sma


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

def test_atr_equals_high_minus_low_when_no_gaps():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    high = pd.Series([10.0] * 20, index=dates)
    low = pd.Series([8.0] * 20, index=dates)
    close = pd.Series([9.0] * 20, index=dates)

    result = atr(high, low, close, window=14)

    valid = result.dropna()
    assert not valid.empty
    assert valid.apply(lambda v: v == pytest.approx(2.0)).all()


def test_atr_captures_overnight_gap():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    high = pd.Series([10, 10, 10, 10, 50], index=dates)  # big gap on the last day
    low = pd.Series([8, 8, 8, 8, 48], index=dates)
    close = pd.Series([9, 9, 9, 9, 49], index=dates)

    result = atr(high, low, close, window=3)

    # Last day's true range is dominated by the gap (|48 - 9| = 39),
    # not just high-low (=2), so ATR should jump sharply.
    assert result.iloc[-1] > result.iloc[-2]


def test_obv_tracks_price_direction():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    close = pd.Series([100, 105, 102, 102], index=dates)  # up, down, flat
    volume = pd.Series([1000, 2000, 1500, 1000], index=dates)

    result = obv(close, volume)

    assert result.iloc[0] == 0
    assert result.iloc[1] == 2000  # up day: +volume
    assert result.iloc[2] == 2000 - 1500  # down day: -volume
    assert result.iloc[3] == result.iloc[2]  # flat day: unchanged


def test_momentum_matches_manual_percentage():
    dates = pd.date_range("2024-01-01", periods=11, freq="D")
    series = pd.Series([100] * 10 + [110], index=dates)  # +10% vs 10 days ago

    result = momentum(series, window=10)

    assert result.iloc[-1] == pytest.approx(10.0)