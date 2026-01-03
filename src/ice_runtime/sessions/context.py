from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable, Dict

from ice_core.logging.bridge import get_logger
from .workspace import Workspace
from ..exceptions import SessionError

logger = get_logger(__name__)

# ============================================================================
# CONTEXT VAR (thread / task local)
# ============================================================================

_current_context: ContextVar[Optional["SessionContext"]] = ContextVar(
    "current_session_context",
    default=None,
)

# ============================================================================
# SESSION STATS
# ============================================================================

@dataclass
class SessionStats:
    operations: int = 0
    errors: int = 0
    warnings: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity_at: datetime = field(default_factory=datetime.utcnow)

    def touch(self) -> None:
        self.last_activity_at = datetime.utcnow()

    def inc_op(self) -> None:
        self.operations += 1
        self.touch()

    def inc_error(self) -> None:
        self.errors += 1
        self.touch()

    def inc_warning(self) -> None:
        self.warnings += 1
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": self.operations,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
        }

# ============================================================================
# SESSION CONFIG
# ============================================================================

@dataclass
class SessionConfig:
    timezone: str = "UTC"
    log_level: str = "INFO"
    settings: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.settings[key] = value

# ============================================================================
# SESSION CONTEXT
# ============================================================================

class SessionContext:
    """
    Runtime session context.

    Responsabilità:
    - workspace attivo
    - transient state
    - stats
    - metadata sessione
    - binding thread/task-local
    """

    def __init__(
        self,
        *,
        workspace: Workspace,
        config: Optional[SessionConfig] = None,
        context_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        panel_context: Optional[str] = None,
    ):
        self.workspace = workspace
        self.config = config or SessionConfig()
        self.context_id = context_id or self._generate_id()
        self.created_at = datetime.utcnow()
        self.expires_at = expires_at

        self.metadata: Dict[str, Any] = dict(metadata or {})
        self.panel_context = panel_context
        self.last_panel_context = panel_context

        self.stats = SessionStats()
        self._state: Dict[str, Any] = {}
        self._on_close: list[Callable[[SessionContext], None]] = []

        logger.debug(
            "SessionContext created id=%s workspace=%s",
            self.context_id,
            self.workspace.id,
        )

    # ------------------------------------------------------------------
    # GLOBAL ACCESS
    # ------------------------------------------------------------------

    @classmethod
    def current(cls) -> Optional["SessionContext"]:
        return _current_context.get()

    @classmethod
    def require_current(cls) -> "SessionContext":
        ctx = cls.current()
        if ctx is None:
            raise SessionError("No active SessionContext")
        return ctx

    @classmethod
    def set_current(cls, ctx: Optional["SessionContext"]) -> None:
        _current_context.set(ctx)

    @classmethod
    def create(
        cls,
        *,
        workspace: Workspace,
        config: Optional[SessionConfig] = None,
        set_current: bool = True,
        **kwargs,
    ) -> "SessionContext":
        ctx = cls(workspace=workspace, config=config, **kwargs)
        if set_current:
            cls.set_current(ctx)
        return ctx

    # ------------------------------------------------------------------
    # STATE
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value
        self.stats.inc_op()

    def delete(self, key: str) -> None:
        self._state.pop(key, None)

    def clear_state(self) -> None:
        self._state.clear()

    # ------------------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------------------

    def on_close(self, fn: Callable[[SessionContext], None]) -> None:
        self._on_close.append(fn)

    def _run_close_callbacks(self) -> None:
        for fn in self._on_close:
            try:
                fn(self)
            except Exception as e:
                logger.error("SessionContext close callback error: %s", e)

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def close(self) -> None:
        logger.debug("Closing SessionContext %s", self.context_id)
        self._run_close_callbacks()
        self.clear_state()

        if SessionContext.current() is self:
            SessionContext.set_current(None)

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------

    def is_expired(self) -> bool:
        return self.expires_at is not None and datetime.utcnow() >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "workspace": {
                "id": self.workspace.id,
                "name": self.workspace.name,
                "state": self.workspace.state.value,
            },
            "stats": self.stats.to_dict(),
            "state_keys": list(self._state.keys()),
            "metadata": dict(self.metadata),
        }

    def _generate_id(self) -> str:
        import uuid
        return f"ctx_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # CONTEXT MANAGER
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "SessionContext":
        SessionContext.set_current(self)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def __repr__(self) -> str:
        return f"SessionContext(id={self.context_id}, workspace={self.workspace.id})"

# ============================================================================
# DI HELPERS
# ============================================================================

def get_current_context() -> SessionContext:
    return SessionContext.require_current()

def with_context(fn: Callable) -> Callable:
    import functools
    import inspect

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        ctx = SessionContext.require_current()
        if inspect.iscoroutinefunction(fn):
            return await fn(ctx, *args, **kwargs)
        return fn(ctx, *args, **kwargs)

    return wrapper
