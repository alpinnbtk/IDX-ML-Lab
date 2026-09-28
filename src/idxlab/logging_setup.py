"""Logging setup for IDX-ML-Lab."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, at the program entry point.

    Library modules (data.py, clean.py, ...) only create their own
    logger via logging.getLogger(__name__); they never configure
    handlers themselves. Whoever *runs* the program (a script, the
    API, a notebook) calls this function once.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )