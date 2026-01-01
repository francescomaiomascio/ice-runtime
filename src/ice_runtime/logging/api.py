from __future__ import annotations

from typing import Optional

from .event import LogEvent
from .router import LogRouter

_router: Optional[LogRouter] = None


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

def init(router: LogRouter) -> None:
    """
    Registra il LogRouter globale.
    Deve essere chiamato UNA sola volta nel bootstrap.
    """
    global _router
    _router = router


def is_initialized() -> bool:
    return _router is not None


# ---------------------------------------------------------------------------
# EMIT
# ---------------------------------------------------------------------------

def emit(event: LogEvent) -> None:
    """
    Entry point globale per il logging runtime-safe.
    """
    if not _router:
        return

    # Runtime ID viene risolto dal router/context
    if event.runtime_id is None:
        event = event.with_runtime_id(_router.ctx.runtime_id)

    _router.emit(event)


# ---------------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------------

def set_phase(phase: str) -> None:
    if not _router:
        return
    _router.ctx.set_phase(phase)


# ---------------------------------------------------------------------------
# SHORTHANDS (compatibilità)
# ---------------------------------------------------------------------------

def info(domain, owner, scope, msg, data=None):
    emit(LogEvent(LogEvent.now(), "INFO", domain, owner, scope, msg, data, None))


def error(domain, owner, scope, msg, data=None):
    emit(LogEvent(LogEvent.now(), "ERROR", domain, owner, scope, msg, data, None))
