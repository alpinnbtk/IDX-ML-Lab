"""Label construction for IDX-ML-Lab.

Builds supervised learning targets from cleaned price data: either a
binary next-N-day direction label (classification) or a forward
N-day return (regression). Both are deliberately *forward-looking*
(they use future prices) — that's what makes them valid labels. But
it also means the last `horizon` rows of any labeled DataFrame will
contain NaN (there's no future price for the most recent days yet),
and these rows must be dropped before training, not filled.

Look-ahead bias warning: a label must always reference the close
`horizon` days AFTER the row's date, aligned back to the row's own
date — representing "what happens after this day's features were
observed." Never derive a label from information available only on
or after the row's own date (e.g. that day's own Close vs Open) and
treat it as predictive; that is trivially knowable, not a forecast.
"""

from __future__ import annotations

import pandas as pd


def forward_return(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Forward percentage return, `horizon` days ahead.

    forward_return_t = (price_(t+horizon) / price_t - 1) * 100

    Returns
    -------
    pd.Series
        Aligned to the same index as `series`. The last `horizon`
        rows are NaN by construction (no future price exists yet for
        them) — left in place, not dropped, same convention as the
        leading NaNs in indicators.py.
    """
    return (series.shift(-horizon) / series - 1) * 100


def direction_label(series: pd.Series, horizon: int = 1) -> pd.Series:
    """Binary direction label: 1.0 if price is higher `horizon` days
    ahead, else 0.0 (a flat/unchanged price counts as "not up").

    Built from forward_return() rather than comparing raw prices
    directly, so both label functions share one definition of "what
    happens after this row."

    Returns
    -------
    pd.Series
        Float values (1.0/0.0), not int — int dtype cannot hold NaN,
        and the trailing rows need to stay NaN like forward_return.
    """
    ret = forward_return(series, horizon=horizon)
    return (ret > 0).astype(float).where(ret.notna())


def build_labels(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Build both label types from cleaned OHLCV data in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned OHLCV data with a "Close" column.
    horizon : int
        Number of days ahead the label looks.

    Returns
    -------
    pd.DataFrame
        Columns: "forward_return" (regression target, %), "direction"
        (classification target, 1.0=up / 0.0=down/flat). Same index
        as `df`; last `horizon` rows are NaN in both columns.
    """
    close = df["Close"]
    return pd.DataFrame(
        {
            "forward_return": forward_return(close, horizon=horizon),
            "direction": direction_label(close, horizon=horizon),
        },
        index=df.index,
    )


def drop_unlabeled_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no valid label (the trailing `horizon` rows).

    A thin, explicit wrapper around dropna() — kept as a named
    function so the intent ("these rows have no label yet, not
    missing data to impute") is clear at the call site, instead of a
    bare .dropna() with no explanation.
    """
    return df.dropna()