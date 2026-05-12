"""Decoradores propios (Semana 1 del curso): log de latencia."""
import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_latency(name: str | None = None) -> Callable[[F], F]:
    """Mide y registra la latencia de la funcion decorada.

    Aplica al endpoint o servicio. Logueamos en milisegundos. Soporta sync y async.
    """

    def _wrap(fn: F) -> F:
        label = name or fn.__qualname__

        if _is_coroutine(fn):

            @wraps(fn)
            async def _async_inner(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    dt_ms = (time.perf_counter() - t0) * 1000
                    logger.info("latency name=%s ms=%.2f", label, dt_ms)

            return _async_inner  # type: ignore[return-value]

        @wraps(fn)
        def _sync_inner(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000
                logger.info("latency name=%s ms=%.2f", label, dt_ms)

        return _sync_inner  # type: ignore[return-value]

    return _wrap


def _is_coroutine(fn: Callable[..., Any]) -> bool:
    import asyncio

    return asyncio.iscoroutinefunction(fn)
