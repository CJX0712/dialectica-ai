"""Structured logging. Falls back to stdlib when loguru is absent.

Author: 晨星
"""
from __future__ import annotations

import logging
import os

_LEVEL = os.environ.get("DIALECTICA_LOG_LEVEL", "INFO").upper()

try:  # pragma: no cover - environment dependent
    from loguru import logger as _loguru  # type: ignore

    _loguru.remove()
    _loguru.add(lambda m: print(m, end=""), level=_LEVEL, colorize=False)
    logger = _loguru
except Exception:  # pragma: no cover
    logging.basicConfig(level=_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logger = logging.getLogger("dialectica")


def get_logger(name: str):
    try:
        return logger.bind(module=name)  # type: ignore[union-attr]
    except Exception:  # pragma: no cover
        return logging.getLogger(f"dialectica.{name}")
