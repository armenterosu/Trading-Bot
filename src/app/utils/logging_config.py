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

    def serialize(record: Dict[str, Any]) -> str:
        payload = {
            "time": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record["module"],
            "function": record["function"],
            "line": record["line"],
        }
        return json.dumps(payload)

    logger.add(sys.stdout, level=level, format=serialize, enqueue=True)
    logger.add(logfile, level=level, format=serialize, enqueue=True,
               rotation="1 day", retention=f"{retention_days} days")
