"""Feature engineering pipeline for IDX-ML-Lab.

Combines the individual indicator functions in indicators.py into a
single, configurable pipeline that turns cleaned OHLCV data into a
feature DataFrame ready for modeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import atr, bollinger_bands, ema, macd, momentum, obv, rsi, sma


@dataclass
class FeaturePipeline:
    """Builds a feature DataFrame from cleaned OHLCV data.

    Each field controls the window/span of its corresponding
    indicator; defaults match the conventions used in indicators.py.
    """

    sma_window: int = 20
    ema_span: int = 20
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_num_std: float = 2.0
    atr_window: int = 14
    momentum_window: int = 10

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all configured indicators for one ticker's OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned OHLCV data with Open/High/Low/Close/Volume columns
            (e.g. the output of clean_ticker_data).

        Returns
        -------
        pd.DataFrame
            One row per date, one column per feature. Early rows will
            contain NaN until each indicator's window is satisfied —
            left in place, not dropped, so the caller (the split
            module, Day 10) decides how to handle them.
        """
        close = df["Close"]

        macd_df = macd(
            close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal
        )
        bb_df = bollinger_bands(close, window=self.bb_window, num_std=self.bb_num_std)

        features = pd.DataFrame(index=df.index)
        features["sma"] = sma(close, window=self.sma_window)
        features["ema"] = ema(close, span=self.ema_span)
        features["rsi"] = rsi(close, window=self.rsi_window)
        features["macd"] = macd_df["macd"]
        features["macd_signal"] = macd_df["signal"]
        features["macd_histogram"] = macd_df["histogram"]
        features["bb_upper"] = bb_df["bb_upper"]
        features["bb_lower"] = bb_df["bb_lower"]
        features["bb_width"] = bb_df["bb_upper"] - bb_df["bb_lower"]
        features["atr"] = atr(df["High"], df["Low"], close, window=self.atr_window)
        features["obv"] = obv(close, df["Volume"])
        features["momentum"] = momentum(close, window=self.momentum_window)

        return features

    def transform_many(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply transform() to every ticker in a dict of cleaned OHLCV data."""
        return {ticker: self.transform(df) for ticker, df in data.items()}