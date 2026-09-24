"""Data loading utilities for IDX-ML-Lab.

Fetches OHLCV price data for Indonesian stock tickers via yfinance,
with local parquet caching to avoid redundant API calls during
development.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


class DataLoader:
    """Fetches and caches OHLCV data for a set of stock tickers.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols, e.g. ["BBCA.JK", "BBRI.JK"].
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.
    cache_dir : str
        Folder where cached parquet files are stored.
    """

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        cache_dir: str = "data/raw",
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str) -> Path:
        """Return the parquet cache path for a given ticker."""
        safe_name = ticker.replace(".", "_")
        return self.cache_dir / f"{safe_name}.parquet"

    def _fetch_from_api(self, ticker: str) -> pd.DataFrame:
        """Download OHLCV data for one ticker from yfinance."""
        df = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")
        df.index.name = "date"
        return df

    def load(self, ticker: str, force_refresh: bool = False) -> pd.DataFrame:
        """Load OHLCV data for one ticker, using cache when available."""
        cache_path = self._cache_path(ticker)

        if cache_path.exists() and not force_refresh:
            return pd.read_parquet(cache_path)

        df = self._fetch_from_api(ticker)
        df.to_parquet(cache_path)
        return df

    def load_all(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        """Load OHLCV data for every ticker in self.tickers."""
        return {
            ticker: self.load(ticker, force_refresh=force_refresh)
            for ticker in self.tickers
        }