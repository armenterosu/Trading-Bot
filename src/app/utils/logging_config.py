from __future__ import annotations
"""Logging setup using loguru with JSON-friendly format and rotation."""
from typing import Dict, Any
from loguru import logger
import sys
import json
import os


def setup_logging(cfg: Dict[str, Any]) -> None:
    logger.remove()

    level = cfg.get("level", "INFO")
    logfile = cfg.get("file", "logs/bot.log")
    retention_days = cfg.get("retention_days", 7)

    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    # Log only to file (JSON), no console output
    logger.add(
        logfile,
        level=level,
        serialize=True,
        enqueue=True,
        rotation="1 day",
        retention=f"{retention_days} days",
    )
