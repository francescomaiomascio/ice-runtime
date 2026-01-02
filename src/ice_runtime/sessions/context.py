from ice_core.logging.bridge import get_logger
"""
Session Context - Stato Corrente della Sessione

SessionContext mantiene lo stato runtime di una sessione:
- Workspace attivo
- Configurazione corrente
- Statistiche e metriche
- User preferences
- Transient state (cache, temp data)

Il context è thread-local e può essere acceduto globalmente tramite
context manager o dependency injection.

Usage:
    # Context manager
    async with SessionContext.create(workspace) as ctx:
        ctx.set("current_source", "app.log")
        source = ctx.get("current_source")

        # Accesso workspace/repositories
        events = ctx.workspace.get_repository("events")

    # Dependency injection (FastAPI-style)
    def process_logs(ctx: SessionContext = Depends(get_context)):
        events = ctx.get_repository("events")
"""

from typing import Any, Optional, TypeVar, Generic, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging

from .workspace import Workspace
from ..exceptions import SessionError

logger = get_logger(__name__)

T = TypeVar('T')


class _LightweightSessionContext:
    """Contesto sincrono minimale usato nei test che non richiedono workspace reale."""

    class _CodeRepo:
        def __init__(self):
            self._files: dict[str, Any] = {}

        def get_file(self, path: str):
            return self._files.get(path)

        def add_file(self, path: str, content: str):
            self._files[path] = type("CodeFile", (), {"content": content})

        def update_file(self, path: str, content: str):
            self._files[path] = type("CodeFile", (), {"content": content})

    class _KnowledgeRepo:
        def __init__(self):
            self._sources: list[Any] = []
            self._analyses: list[Any] = []
            self._timelines: list[Any] = []

        def register_source(self, source_id: str, source_type: str, metadata: dict | None = None):
            self._sources.append(
                type("Source", (), {"source_id": source_id, "source_type": source_type, "metadata": metadata or {}})
            )

        def list_sources(self):
            return list(self._sources)

        def get_source(self, source_id: str):
            for s in self._sources:
                if getattr(s, "source_id", None) == source_id:
                    return s
            return None

        def register_analysis(self, source: str, content: str, mode: str, reasoning: str | None = None):
            rec = type(
                "Analysis",
                (),
                {
                    "id": f"analysis_{len(self._analyses)+1}",
                    "source": source,
                    "content": content,
                    "mode": mode,
                    "reasoning": reasoning,
                },
            )
            self._analyses.append(rec)
            return rec.id

        def register_refactor(self, source: str, old_code: str, new_code: str, diff: str, reasoning: str | None = None):
            rec = type(
                "Refactor",
                (),
                {
                    "id": f"refactor_{len(self._analyses)+1}",
                    "source": source,
                    "old_code": old_code,
                    "new_code": new_code,
                    "diff": diff,
                    "reasoning": reasoning,
                },
            )
            self._analyses.append(rec)
            return rec.id

        def register_timeline(self, title: str, events: list[Any]):
            """Registra una timeline minimale per i test."""
            timeline = type(
                "Timeline",
                (),
                {
                    "id": f"timeline_{len(self._timelines)+1}",
                    "title": title,
                    "events": list(events),
                },
            )
            self._timelines.append(timeline)
            return timeline.id

    def __init__(self, connection_string: str, workspace_root: Optional[str] = None):
        self.connection_string = connection_string
        self.storage_backend = None
        self.context_id = f"light-{hash(connection_string)}"
        self.workspace_root = workspace_root

        class _Workspace:
            def __init__(self, root_path: Optional[str]):
                self.root_path = root_path
                self.workspace_id = "test-ws"

            def resolve_path(self, p: str) -> Path:
                return Path(self.root_path or ".") / p

        self.workspace = _Workspace(workspace_root)
        self.session = self
        self._code_repo = self._CodeRepo()
        self._knowledge_repo = self._KnowledgeRepo()

    def get_repository(self, repo_type: str):
        if repo_type == "code":
            return self._code_repo
        if repo_type == "knowledge":
            return self._knowledge_repo
        raise KeyError(f"Repository {repo_type} non disponibile nel contesto leggero.")


# ============================================================================
# CONTEXT VARIABLES (Thread-local storage)
# ============================================================================

# Current session context (thread-safe)
_current_context: ContextVar[Optional["SessionContext"]] = ContextVar(
    "current_context",
    default=None
)


# ============================================================================
# SESSION STATISTICS
# ============================================================================

@dataclass
class SessionStats:
    """
    Statistiche di una sessione.

    Traccia metriche runtime come:
    - Operazioni eseguite
    - Tempo di esecuzione
    - Errori/warning
    - Resource usage
    """

    # Counters
    operations_count: int = 0
    queries_count: int = 0
    errors_count: int = 0
    warnings_count: int = 0

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)

    # Resource usage
    memory_usage_mb: float = 0.0
    storage_usage_mb: float = 0.0

    def update_activity(self) -> None:
        """Aggiorna timestamp ultima attività."""
        self.last_activity_at = datetime.now()

    def increment_operation(self) -> None:
        """Incrementa counter operazioni."""
        self.operations_count += 1
        self.update_activity()

    def increment_query(self) -> None:
        """Incrementa counter query."""
        self.queries_count += 1
        self.update_activity()

    def increment_error(self) -> None:
        """Incrementa counter errori."""
        self.errors_count += 1
        self.update_activity()

    def increment_warning(self) -> None:
        """Incrementa counter warning."""
        self.warnings_count += 1
        self.update_activity()

    def get_session_duration(self) -> float:
        """
        Ottiene durata sessione in secondi.

        Returns:
            Durata in secondi
        """
        delta = datetime.now() - self.started_at
        return delta.total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serializza a dict."""
        return {
            "operations_count": self.operations_count,
            "queries_count": self.queries_count,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "session_duration_seconds": self.get_session_duration(),
            "memory_usage_mb": self.memory_usage_mb,
            "storage_usage_mb": self.storage_usage_mb,
        }


# ============================================================================
# SESSION CONFIGURATION
# ============================================================================

@dataclass
class SessionConfig:
    """
    Configurazione di una sessione.

    Contiene preferenze e impostazioni runtime:
    - User preferences
    - Feature flags
    - Performance tuning
    - Logging level
    """

    # User preferences
    default_source: Optional[str] = None
    default_level_filter: Optional[str] = None
    timezone: str = "UTC"

    # Feature flags
    enable_caching: bool = True
    enable_vector_search: bool = True
    enable_auto_indexing: bool = True

    # Performance
    batch_size: int = 1000
    max_results: int = 10000
    query_timeout_seconds: int = 30

    # Logging
    log_level: str = "INFO"
    log_queries: bool = False

    # Advanced
    settings: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Ottiene un setting custom."""
        return self.settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Imposta un setting custom."""
        self.settings[key] = value

    def to_dict(self) -> dict[str, Any]:
        """Serializza a dict."""
        return {
            "default_source": self.default_source,
            "default_level_filter": self.default_level_filter,
            "timezone": self.timezone,
            "enable_caching": self.enable_caching,
            "enable_vector_search": self.enable_vector_search,
            "enable_auto_indexing": self.enable_auto_indexing,
            "batch_size": self.batch_size,
            "max_results": self.max_results,
            "query_timeout_seconds": self.query_timeout_seconds,
            "log_level": self.log_level,
            "log_queries": self.log_queries,
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        """Deserializza da dict."""
        return cls(
            default_source=data.get("default_source"),
            default_level_filter=data.get("default_level_filter"),
            timezone=data.get("timezone", "UTC"),
            enable_caching=data.get("enable_caching", True),
            enable_vector_search=data.get("enable_vector_search", True),
            enable_auto_indexing=data.get("enable_auto_indexing", True),
            batch_size=data.get("batch_size", 1000),
            max_results=data.get("max_results", 10000),
            query_timeout_seconds=data.get("query_timeout_seconds", 30),
            log_level=data.get("log_level", "INFO"),
            log_queries=data.get("log_queries", False),
            settings=data.get("settings", {}),
        )


# ============================================================================
# SESSION CONTEXT
# ============================================================================

class SessionContext:
    """
    Context di una sessione attiva.

    Mantiene lo stato runtime di una sessione:
    - Workspace attivo
    - Configurazione
    - Statistiche
    - Transient state (cache temporanea)

    Il context è thread-local e può essere acceduto globalmente.

    Features:
    - Thread-safe access via ContextVar
    - Automatic stats tracking
    - Repository access shortcuts
    - Transient state management
    - Context manager support
    """

    def __init__(
        self,
        workspace: Workspace,
        config: Optional[SessionConfig] = None,
        context_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        workspace_id_override: Optional[str] = None,
        panel_context: Optional[str] = None,
    ):
        """
        Args:
            workspace: Workspace attivo
            config: Configurazione sessione
            context_id: ID univoco context (auto-generato se None)
            metadata: Dati aggiuntivi legati alla sessione
            expires_at: Timestamp di scadenza del context (opzionale)
        """
        self.workspace = workspace
        self.config = config or SessionConfig()
        self.context_id = context_id or self._generate_context_id()
        self.created_at = datetime.now()
        self.metadata: dict[str, Any] = metadata.copy() if metadata else {}
        self.expires_at = expires_at
        self.workspace_id = workspace_id_override or workspace.id
        self.panel_context = panel_context

        # Statistics
        self.stats = SessionStats()

        # UI tracking
        self.last_panel_context: Optional[str] = panel_context

        # Transient state (runtime cache)
        self._state: dict[str, Any] = {}
        self._on_close_callbacks = []

    # ============================================================================
    # BACKENDS PROXY (compat E2E)
    # ============================================================================
    @property
    def backends(self):
        """
        Proxy ai backend del workspace, richiesto dal test_07_full_pipeline.
        Restituisce la mappa completa dei backend registrati dal workspace.
        """
        # Workspace li conserva sempre in workspace._backends
        if hasattr(self.workspace, "_backends"):
            return self.workspace._backends

        # fallback (non dovrebbe mai accadere)
        return {}

    @classmethod
    def open(cls, connection_string: str, workspace_root: Optional[str] = None) -> "_LightweightSessionContext":
        """
        Helper sincrono per creare un contesto leggero (usato nei test).
        """
        return _LightweightSessionContext(connection_string, workspace_root=workspace_root)

    # ========================================================================
    # FACTORY & GLOBAL ACCESS
    # ========================================================================

    @classmethod
    def create(
        cls,
        workspace: Workspace,
        config: Optional[SessionConfig] = None,
        set_as_current: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        workspace_id: Optional[str] = None,
        panel_context: Optional[str] = None,
    ) -> "SessionContext":
        """
        Crea un nuovo context.
        """
        ctx = cls(
            workspace=workspace,
            config=config,
            metadata=metadata,
            expires_at=expires_at,
            workspace_id_override=workspace_id,
            panel_context=panel_context,
        )

        if set_as_current:
            cls.set_current(ctx)

        return ctx

    @classmethod
    def current(cls) -> Optional["SessionContext"]:
        """
        Ottiene il context corrente (thread-local).
        """
        return _current_context.get()

    @classmethod
    def set_current(cls, context: Optional["SessionContext"]) -> None:
        """
        Imposta il context corrente (thread-local).
        """
        _current_context.set(context)
        if context:
            logger.debug(f"Current context impostato: {context.context_id}")
        else:
            logger.debug("Current context cleared")

    @classmethod
    def require_current(cls) -> "SessionContext":
        """
        Ottiene il context corrente (richiesto).
        """
        ctx = cls.current()
        if ctx is None:
            raise SessionError(
                "Nessun SessionContext attivo. Usa SessionContext.create() o 'async with' block."
            )
        return ctx

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    def get(self, key: str, default: Any = None) -> Any:
        """Ottiene un valore dallo state transient."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Imposta un valore nello state transient."""
        self._state[key] = value
        logger.debug(f"State impostato: {key} = {value}")

    def delete(self, key: str) -> None:
        """Elimina un valore dallo state."""
        self._state.pop(key, None)

    def has(self, key: str) -> bool:
        """Verifica se una chiave esiste nello state."""
        return key in self._state

    def clear_state(self) -> None:
        """Pulisce tutto lo state transient."""
        self._state.clear()
        logger.debug("State cleared")

    def set_panel_context(self, panel: str) -> None:
        """Aggiorna il pannello UI associato a questa sessione."""
        self.panel_context = panel
        self.last_panel_context = panel

    # ========================================================================
    # WORKSPACE & REPOSITORY ACCESS
    # ========================================================================

    def get_repository(self, repo_type: str) -> Any:
        """
        Shortcut per accedere a un repository.
        """
        self.stats.increment_operation()
        return self.workspace.get_repository(repo_type)

    def get_backend(self, name: str) -> Any:
        """
        Shortcut per accedere a un backend.
        """
        return self.workspace.get_backend(name)

    # ========================================================================
    # AI REPOSITORIES (LLM, RAG, KNOWLEDGE)
    # ========================================================================

    def get_llm_repository(self):
        return self.workspace.get_llm_repository()

    def get_rag_repository(self):
        return self.workspace.get_rag_repository()

    def get_knowledge_repository(self):
        return self.workspace.get_knowledge_repository()

    # ========================================================================
    # AI ADAPTERS (LLM / CODER / EMBEDDINGS)
    # ========================================================================

    def get_llm_adapter(self):
        """
        Restituisce l'LLM principale associato al workspace (se disponibile).
        """
        if hasattr(self.workspace, "get_llm_main"):
            return self.workspace.get_llm_main()
        return None

    def get_coder_llm_adapter(self):
        """
        Restituisce l'LLM specializzato per il codice (se disponibile).
        """
        if hasattr(self.workspace, "get_llm_coder"):
            return self.workspace.get_llm_coder()
        return None

    def get_embedding_engine(self):
        """
        Restituisce il motore di embeddings o la sua configurazione (lazy).
        """
        if hasattr(self.workspace, "get_embedding_engine"):
            return self.workspace.get_embedding_engine()
        return None

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def on_close(self, callback: Callable[["SessionContext"], None]) -> None:
        """
        Registra callback da eseguire alla chiusura del context.
        """
        self._on_close_callbacks.append(callback)

    def _trigger_close_callbacks(self) -> None:
        """Esegue tutti i callbacks di chiusura."""
        for callback in self._on_close_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Errore in close callback: {e}")

    # ========================================================================
    # LIFECYCLE
    # ========================================================================

    async def close(self) -> None:
        """
        Chiude il context.
        """
        logger.debug(f"Chiusura SessionContext: {self.context_id}")

        # Callbacks
        self._trigger_close_callbacks()

        # Clear state
        self.clear_state()

        # Clear current se è questo context
        if SessionContext.current() == self:
            SessionContext.set_current(None)

        logger.debug(f"SessionContext chiuso: {self.context_id}")

    # ========================================================================
    # INFO & DEBUG
    # ========================================================================

    def get_info(self) -> dict[str, Any]:
        """
        Ottiene informazioni sul context.
        """
        return {
            "context_id": self.context_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "workspace": {
                "id": self.workspace.id,
                "name": self.workspace.name,
                "state": self.workspace.state.value,
            },
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "state_keys": list(self._state.keys()),
            "metadata": self.metadata.copy(),
        }

    def serialize(self) -> dict[str, Any]:
        """Serializza il context in un dizionario."""
        return self.get_info()

    def update_metadata(self, data: dict[str, Any]) -> None:
        """Aggiorna e merge i metadata della sessione."""
        self.metadata.update(data)

    def copy(self) -> "SessionContext":
        """
        Crea un clone indipendente del context (state/metadata copiati).
        """
        clone = SessionContext(
            workspace=self.workspace,
            config=self.config,
            metadata=self.metadata.copy(),
            expires_at=self.expires_at,
        )
        clone._state = self._state.copy()
        return clone

    def is_expired(self) -> bool:
        """
        Verifica se il context è scaduto (se expires_at è impostato).
        """
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def _generate_context_id(self) -> str:
        """Genera ID univoco per context."""
        import uuid
        return f"ctx_{uuid.uuid4().hex[:8]}"

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    async def __aenter__(self) -> "SessionContext":
        """Async context manager entry."""
        SessionContext.set_current(self)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        return (
            f"SessionContext(id={self.context_id}, "
            f"workspace={self.workspace.id}, "
            f"operations={self.stats.operations_count})"
        )


# ============================================================================
# DEPENDENCY INJECTION HELPER
# ============================================================================

def get_current_context() -> SessionContext:
    """
    Dependency injection helper per ottenere context corrente.
    """
    return SessionContext.require_current()


# ============================================================================
# CONTEXT DECORATOR
# ============================================================================

def with_context(func: Callable) -> Callable:
    """
    Decorator per funzioni che richiedono un SessionContext.
    """
    import functools
    import inspect

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Ottieni context corrente
        ctx = SessionContext.require_current()

        # Inietta context come primo argomento
        if inspect.iscoroutinefunction(func):
            return await func(ctx, *args, **kwargs)
        else:
            return func(ctx, *args, **kwargs)

    return wrapper