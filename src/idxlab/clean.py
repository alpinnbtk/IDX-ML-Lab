"""Data cleaning utilities for IDX-ML-Lab.

Cleans raw OHLCV data fetched by DataLoader: handles missing values,
removes duplicate dates, aligns multiple tickers to a common trading
calendar, and flags potential stock-split artifacts that yfinance's
auto-adjustment should already have corrected.
"""

from __future__ import annotations

import pandas as pd


def remove_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate index entries, keeping the first occurrence."""
    return df[~df.index.duplicated(keep="first")]


def handle_missing_values(df: pd.DataFrame, max_gap_days: int = 3) -> pd.DataFrame:
    """Fill small gaps in OHLCV data and drop rows that remain incomplete.

    Small gaps (e.g. a data provider hiccup) are forward-filled up to
    `max_gap_days` consecutive missing rows. Any row still containing
    NaNs after that (e.g. a long gap, or missing data at the very
    start of the series) is dropped rather than filled, to avoid
    inventing prices.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data indexed by date.
    max_gap_days : int
        Maximum number of consecutive missing rows to forward-fill.

    Returns
    -------
    pd.DataFrame
        Cleaned data with no remaining NaNs.
    """
    df = df.sort_index()
    df = df.ffill(limit=max_gap_days)
    df = df.dropna()
    return df


def flag_potential_split_artifacts(
    df: pd.DataFrame, threshold: float = 0.4
) -> pd.DataFrame:
    """Flag days with an abnormally large single-day price jump.

    yfinance's `auto_adjust` should already correct prices for stock
    splits, so this is a *sanity check*, not a fix: it flags rows
    where a split may not have been adjusted correctly, so they can
    be inspected manually rather than silently feeding bad data into
    later feature/model steps.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with a "Close" column.
    threshold : float
        Fractional day-over-day change considered abnormal (0.4 = 40%).

    Returns
    -------
    pd.DataFrame
        The same data with an added boolean "split_suspect" column.
    """
    df = df.copy()
    daily_change = df["Close"].pct_change().abs()
    df["split_suspect"] = daily_change > threshold
    return df


def align_trading_calendar(
    data: dict[str, pd.DataFrame], how: str = "inner"
) -> dict[str, pd.DataFrame]:
    """Align multiple tickers to a common set of trading dates.

    Different tickers occasionally have slightly different available
    dates (e.g. one had a temporary trading halt). This reindexes all
    tickers to the same date index so multi-ticker analysis later
    (e.g. the shared model in Week 6) can assume every ticker has a
    row for every date.

    Parameters
    ----------
    data : dict[str, pd.DataFrame]
        Mapping of ticker -> OHLCV DataFrame.
    how : str
        "inner" keeps only dates common to all tickers (safest,
        default). "outer" keeps all dates, forward-filling gaps.

    Returns
    -------
    dict[str, pd.DataFrame]
        Same tickers, reindexed to a common date index.
    """
    if not data:
        return {}

    indices = [df.index for df in data.values()]
    if how == "inner":
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
    elif how == "outer":
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.union(idx)
    else:
        raise ValueError(f"Unknown alignment mode: {how!r}")

    common_index = common_index.sort_values()

    aligned: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        reindexed = df.reindex(common_index)
        if how == "outer":
            reindexed = reindexed.ffill()
        aligned[ticker] = reindexed
    return aligned


def clean_ticker_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full single-ticker cleaning pipeline.

    Order: remove duplicate dates -> handle missing values -> flag
    potential split artifacts.
    """
    df = remove_duplicate_dates(df)
    df = handle_missing_values(df)
    df = flag_potential_split_artifacts(df)
    return df