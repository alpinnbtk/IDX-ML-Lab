"""Tests for the DataLoader class (src/idxlab/data.py)."""

import pandas as pd
import pytest

import idxlab.data as data_module
from idxlab.data import DataLoader


def _fake_ohlcv(n_days: int = 5, multiindex_ticker: str | None = None) -> pd.DataFrame:
    """Build a small fake OHLCV DataFrame, same shape as yfinance output."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(n_days),
            "High": range(n_days),
            "Low": range(n_days),
            "Close": range(n_days),
            "Volume": range(n_days),
        },
        index=dates,
    )
    if multiindex_ticker:
        df.columns = pd.MultiIndex.from_product([df.columns, [multiindex_ticker]])
    return df


@pytest.fixture
def loader(tmp_path):
    """A DataLoader pointed at a temporary cache folder (not the real data/raw)."""
    return DataLoader(
        tickers=["FAKE.JK"],
        start_date="2024-01-01",
        end_date="2024-01-05",
        cache_dir=str(tmp_path),
    )


def test_cache_path_replaces_dot_with_underscore(loader):
    path = loader._cache_path("BBCA.JK")
    assert path.name == "BBCA_JK.parquet"


def test_load_fetches_from_api_when_no_cache_exists(loader, monkeypatch):
    call_count = {"n": 0}

    def fake_download(*args, **kwargs):
        call_count["n"] += 1
        return _fake_ohlcv()

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    df = loader.load("FAKE.JK")

    assert call_count["n"] == 1
    assert not df.empty
    assert loader._cache_path("FAKE.JK").exists()

def test_fetch_flattens_multiindex_columns(loader, monkeypatch):
    monkeypatch.setattr(
        data_module.yf,
        "download",
        lambda *a, **k: _fake_ohlcv(multiindex_ticker="FAKE.JK"),
    )

    df = loader.load("FAKE.JK")

    assert not isinstance(df.columns, pd.MultiIndex)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_load_uses_cache_on_second_call(loader, monkeypatch):
    call_count = {"n": 0}

    def fake_download(*args, **kwargs):
        call_count["n"] += 1
        return _fake_ohlcv()

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    loader.load("FAKE.JK")  # first call -> hits fake API, writes cache
    loader.load("FAKE.JK")  # second call -> should read from cache instead

    assert call_count["n"] == 1  # API only called once


def test_load_force_refresh_bypasses_cache(loader, monkeypatch):
    call_count = {"n": 0}

    def fake_download(*args, **kwargs):
        call_count["n"] += 1
        return _fake_ohlcv()

    monkeypatch.setattr(data_module.yf, "download", fake_download)

    loader.load("FAKE.JK")
    loader.load("FAKE.JK", force_refresh=True)

    assert call_count["n"] == 2


def test_fetch_raises_on_empty_data(loader, monkeypatch):
    monkeypatch.setattr(data_module.yf, "download", lambda *a, **k: pd.DataFrame())

    with pytest.raises(ValueError, match="No data returned"):
        loader.load("FAKE.JK")