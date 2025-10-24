"""
Metrics & timing helpers
------------------------
Provide a simple decorator to time function calls, and a place to add counters later.

Features:
- `@timed(name=None)`: works for sync *and* async callables.
- Structured logs via utils.logger (extras: span, func, duration_ms, ok).
- Context manager: `with time_block("span"):` … or `Timer("span")`.
"""

from __future__ import annotations

import inspect
import time
from functools import wraps
from typing import Any, Callable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["timed", "Timer", "time_block"]


class Timer:
    """Context manager to time arbitrary code blocks.

    Example:
        with Timer("indexing.batch"):
            bulk_index()
    """

    def __init__(self, span: str, *, extra: Optional[dict] = None) -> None:
        self.span = span
        self.extra = extra or {}
        self._start_ns: int | None = None
        self.duration_ms: float | None = None
        self.ok: bool = True

    def __enter__(self) -> "Timer":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start_ns is None:
            return
        end_ns = time.perf_counter_ns()
        self.duration_ms = (end_ns - self._start_ns) / 1_000_000.0
        self.ok = exc is None
        logger.info(
            f"{self.span} executed in {self.duration_ms:.2f} ms",
            extra={"metric": "timing", "span": self.span, "duration_ms": round(self.duration_ms, 2), "ok": self.ok, **self.extra},
        )
        # Do not suppress exceptions
        return False


def time_block(span: str, *, extra: Optional[dict] = None) -> Timer:
    """Convenience factory for Timer context manager."""
    return Timer(span, extra=extra)


def timed(name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log execution time of sync/async functions.

    Usage:
        @timed("fact_finder.process")
        def process(...):
            ...

        @timed()  # span auto = module.qualname
        async def fetch(...):
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        span = name or f"{fn.__module__}.{getattr(fn, '__qualname__', fn.__name__)}"

        if inspect.iscoroutinefunction(fn):
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_ns = time.perf_counter_ns()
                ok = True
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    ok = False
                    raise
                finally:
                    dur_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                    logger.info(
                        f"{span} executed in {dur_ms:.2f} ms",
                        extra={
                            "metric": "timing",
                            "span": span,
                            "func": getattr(fn, "__qualname__", fn.__name__),
                            "duration_ms": round(dur_ms, 2),
                            "ok": ok,
                        },
                    )
            return wrapper

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_ns = time.perf_counter_ns()
            ok = True
            try:
                return fn(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                dur_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                logger.info(
                    f"{span} executed in {dur_ms:.2f} ms",
                    extra={
                        "metric": "timing",
                        "span": span,
                        "func": getattr(fn, "__qualname__", fn.__name__),
                        "duration_ms": round(dur_ms, 2),
                        "ok": ok,
                    },
                )
        return wrapper

    return decorator