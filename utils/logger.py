"""
Logging utilities
-----------------
Provide a single way to get a structured logger.

Key features:
- Single root handler configured once (stdout).
- Level from env via utils.config.Settings (LOG_LEVEL/DEBUG).
- Safe formatter that appends any `extra={}` fields as key=value.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict

# Import here to avoid circular imports (utils.config does not import logger)
try:
    from utils.config import Settings
except Exception:
    Settings = None  # Fallback if settings cannot be loaded at import time


# -----------------------------------------------------------------------------
# Custom formatter that prints unknown LogRecord attributes (i.e., extras)
# -----------------------------------------------------------------------------

class KeyValueExtrasFormatter(logging.Formatter):
    """Append custom attributes passed via `extra={}` as key=value pairs."""

    # Set of standard LogRecord attributes (won't be repeated as extras)
    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName", "process",
        "processName", "message", "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)

        # Collect extras (i.e., any non-reserved attributes)
        extras: Dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k not in self._RESERVED:
                # Avoid duplicating built-in fields commonly added by logging internals
                if k.startswith("_"):
                    continue
                extras[k] = v

        if extras:
            # Render extras as key=value pairs; use repr to be safe on complex types
            extra_str = " ".join(f"{k}={repr(v)}" for k, v in extras.items())
            return f"{base} | {extra_str}"

        return base


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

_CONFIGURED = False


def _level_from_settings() -> int:
    """Resolve log level from Settings (LOG_LEVEL/DEBUG); fallback to INFO."""
    default_level = logging.INFO
    if Settings is None:
        return default_level
    try:
        s = Settings()
        if getattr(s, "DEBUG", False):
            return logging.DEBUG
        level_name = str(getattr(s, "LOG_LEVEL", "INFO")).upper().strip()
        return getattr(logging, level_name, default_level)
    except Exception:
        return default_level


def configure_logging(level: int | None = None) -> None:
    """Configure the root logger once (stdout handler + extras-aware formatter)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    lvl = level if level is not None else _level_from_settings()

    root = logging.getLogger()
    root.setLevel(lvl)

    # Avoid duplicate handlers if called multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        formatter = KeyValueExtrasFormatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Be nice with common noisy libraries (optional: align level)
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "asyncio"):
        try:
            logging.getLogger(noisy).setLevel(lvl)
        except Exception:
            pass

    _CONFIGURED = True


def set_log_level(level_name: str) -> None:
    """Dynamically change root logger level (e.g., set_log_level('DEBUG'))."""
    configure_logging()  # ensure configured
    lvl = getattr(logging, level_name.upper(), logging.INFO)
    logging.getLogger().setLevel(lvl)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a logger with sane defaults.

    Behavior:
    - Configures the root logger once (stdout + structured formatter).
    - Returns a named child logger; no extra handlers are attached here,
      so messages propagate to the root (avoids duplicates).
    - Use `logger.info(..., extra={'key': val})` to attach structured fields.
    """
    configure_logging()
    return logging.getLogger(name)