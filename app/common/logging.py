"""Logging helpers for local services."""

from __future__ import annotations

import json
import logging
from typing import Any


def configure_logging(level: str = "INFO", use_json: bool = False) -> None:
    """Configure root logging once for the current process."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    if use_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())


class JsonFormatter(logging.Formatter):
    """Very small JSON formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        """Return a JSON-encoded log line."""
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)
