"""Project configuration loading for IDX-ML-Lab.

Reads settings from a YAML file (configs/config.yaml) and allows
selected values to be overridden via environment variables / .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    """Immutable container for project settings."""

    tickers: list[str]
    start_date: str
    end_date: str
    cache_dir: str = "data/raw"
    log_level: str = "INFO"


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    """Load settings from a YAML file, with env-var override for log level.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file, relative to where the program
        is run (normally the project root).

    Returns
    -------
    Config
        Parsed, immutable configuration.
    """
    load_dotenv()

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data = raw["data"]
    log_level = os.getenv("IDXLAB_LOG_LEVEL", raw.get("logging", {}).get("level", "INFO"))

    return Config(
        tickers=data["tickers"],
        start_date=data["start_date"],
        end_date=data["end_date"],
        cache_dir=data.get("cache_dir", "data/raw"),
        log_level=log_level.upper(),
    )