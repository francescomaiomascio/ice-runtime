from __future__ import annotations
from ice_core.logging.bridge import get_logger
"""
Session Manager - Orchestratore di Sessioni e Workspace AI-Enhanced

AGGIORNATO per supportare:
- 🤖 AI Workspace configuration
- 🕸️ Knowledge Graph workspace types
- 🔤 Multi-model LLM management
- 📚 RAG-enhanced session activation
"""
import os
from ..base import BackendConfig, BackendType, StorageMode


from engine.constants import GLOBAL_KNOWLEDGE_WORKSPACE_ID
from typing import Any, Optional, Callable, Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import logging
import asyncio
import shutil


def _default_workspace_storage_dir() -> Path:
    return Path(os.getenv("ICE_STUDIO_WORKSPACE_STORAGE_DIR", Path.home() / ".ice_studio" / "workspaces"))

from .workspace import (
    Workspace,
    WorkspaceMetadata,
    WorkspaceState,
    WorkspaceLifecycleHook,
    AIWorkspaceSettings,
)
from .context import SessionContext, SessionConfig
from .lifecycle import SessionLifecycle, LifecycleError
from ..base import BackendConfig
from ..exceptions import (
    SessionError,
    WorkspaceError,
    WorkspaceNotFoundError,
    WorkspaceAlreadyExistsError,
)


logger = get_logger(__name__)


# ============================================================================
# AI WORKSPACE CONFIGURATION - NUOVO
# ============================================================================

class AIWorkspaceConfig:
    """
    Configurazione specifica per workspace AI-enhanced.
    
    Definisce:
    - Tipi di workspace AI (code_analysis, log_analysis, multi_agent, etc.)
    - Configurazione modelli LLM predefiniti
    - Setup Knowledge Graph
    - Configurazione RAG
    """
    
    # Tipi di workspace AI predefiniti
    WORKSPACE_TYPES: Dict[str, Dict[str, Any]] = {
        "code_analysis": {
            "description": "AI-powered code analysis and refactoring",
            "default_llm_models": ["gpt-4", "claude-3"],
            "required_repositories": ["code_sources", "code_elements", "code_issues"],
            "ai_features": ["code_analysis", "refactoring", "pattern_detection", "rag"],
            "knowledge_graph_enabled": True,
        },
        "log_analysis": {
            "description": "AI-powered log analysis and anomaly detection",
            "default_llm_models": ["gpt-4", "claude-3"],
            "required_repositories": ["log_sources", "log_events", "log_patterns"],
            "ai_features": ["log_analysis", "anomaly_detection", "pattern_matching"],
            "knowledge_graph_enabled": True,
        },
        "multi_agent": {
            "description": "Multi-agent AI orchestration workspace",
            "default_llm_models": ["gpt-4", "claude-3", "local/llama"],
            "required_repositories": ["agent_workflows", "llm_interactions", "rag_sessions"],
            "ai_features": ["multi_agent", "workflow_orchestration", "rag"],
            "knowledge_graph_enabled": True,
        },
        "knowledge_base": {
            "description": "Knowledge Graph and semantic search workspace",
            "default_llm_models": ["gpt-4", "text-embedding-ada-002"],
            "required_repositories": [
                "knowledge_entities",
                "knowledge_relationships",
                "knowledge_embeddings",
            ],
            "ai_features": ["semantic_search", "knowledge_graph", "rag"],
            "knowledge_graph_enabled": True,
        },
    }
    
    @classmethod
    def get_workspace_type_config(cls, workspace_type: str) -> Dict[str, Any]:
        """
        Ottiene configurazione per tipo workspace.
        
        Raises:
            ValueError: Se tipo non supportato
        """
        if workspace_type not in cls.WORKSPACE_TYPES:
            raise ValueError(f"Tipo workspace non supportato: {workspace_type}")
        
        return cls.WORKSPACE_TYPES[workspace_type]
    
    @classmethod
    def get_default_ai_settings(cls, workspace_type: str) -> Dict[str, Any]:
        """
        Ottiene settings AI di default per tipo workspace.
        
        Returns:
            Dict con settings AI (workspace_type, ai_config, required_repositories)
        """
        config = cls.get_workspace_type_config(workspace_type)
        
        return {
            "workspace_type": workspace_type,
            "ai_config": {
                "workspace_type": workspace_type,
                "enabled_features": config["ai_features"],
                "knowledge_graph_enabled": config["knowledge_graph_enabled"],
                "rag_enabled": "rag" in config["ai_features"],
                "default_models": config["default_llm_models"],
            },
            "required_repositories": config["required_repositories"],
        }


# ============================================================================
# SESSION REGISTRY - AGGIORNATO
# ============================================================================

class SessionRegistry:
    """
    Registry per tracciare workspace e sessioni AI-enhanced.
    
    Mantiene:
    - Workspace AI attivi (in memoria)
    - Path dei workspace scoperti su disco
    - AI-specific state tracking
    - Performance metrics AI
    - Agent session tracking
    """
    
    def __init__(self) -> None:
        self._workspaces: dict[str, Workspace] = {}
        self._workspace_paths: dict[str, Path] = {}
        self._active_contexts: dict[str, SessionContext] = {}
        
        # AI-specific tracking - NUOVO
        self._ai_workspace_types: dict[str, str] = {}  # workspace_id -> type
        self._ai_performance_metrics: dict[str, Dict[str, Any]] = {}
    
    # ------------------------------------------------------------------------
    # WORKSPACE REGISTRATION
    # ------------------------------------------------------------------------
    
    def register_workspace(self, workspace: Workspace, workspace_type: str = "generic") -> None:
        """Registra un workspace attivo con tipo AI."""
        self._workspaces[workspace.id] = workspace
        self._workspace_paths[workspace.id] = workspace.base_path
        self._ai_workspace_types[workspace.id] = workspace_type or "generic"
        
        logger.debug(f"Workspace AI registrato: {workspace.id} (type: {workspace_type})")
    
    def register_workspace_path(
        self,
        workspace_id: str,
        path: Path,
        workspace_type: str = "generic",
    ) -> None:
        """
        Registra solo il path di un workspace (senza caricarlo in memoria).
        Utile durante la discovery iniziale.
        """
        self._workspace_paths[workspace_id] = path
        # Non sovrascrivere un tipo già noto
        if workspace_id not in self._ai_workspace_types:
            self._ai_workspace_types[workspace_id] = workspace_type or "generic"
    
    def unregister_workspace(self, workspace_id: str) -> None:
        """Rimuove workspace dal registry (in memoria + metriche)."""
        self._workspaces.pop(workspace_id, None)
        self._ai_workspace_types.pop(workspace_id, None)
        self._ai_performance_metrics.pop(workspace_id, None)
        # Manteniamo il path solo se non stiamo cancellando da disco:
        # il caller, se cancella da disco, potrà fare pulizia del path.
        logger.debug(f"Workspace rimosso dal registry: {workspace_id}")
    
    def drop_workspace_path(self, workspace_id: str) -> None:
        """Rimuove solo il path registrato (usato dopo delete da disco)."""
        self._workspace_paths.pop(workspace_id, None)
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Ottiene workspace per ID (solo se già caricato)."""
        return self._workspaces.get(workspace_id)
    
    def get_workspace_type(self, workspace_id: str) -> str:
        """Ottiene tipo workspace AI (default: generic)."""
        return self._ai_workspace_types.get(workspace_id, "generic")
    
    def has_workspace(self, workspace_id: str) -> bool:
        """Verifica se workspace è registrato (oggetto in memoria)."""
        return workspace_id in self._workspaces
    
    def list_workspace_ids(self) -> list[str]:
        """
        Lista ID workspace conosciuti (in memoria o solo da path).
        
        Unifica:
        - workspace caricati in _workspaces
        - workspace scoperti solo su disco in _workspace_paths
        """
        ids = set(self._workspace_paths.keys()) | set(self._workspaces.keys())
        return sorted(ids)
    
    def list_workspaces_by_type(self, workspace_type: str) -> List[str]:
        """Lista ID workspace per tipo specifico."""
        return [
            ws_id for ws_id, ws_type in self._ai_workspace_types.items()
            if ws_type == workspace_type
        ]
    
    def get_workspace_path(self, workspace_id: str) -> Optional[Path]:
        """Ottiene path di un workspace se noto."""
        return self._workspace_paths.get(workspace_id)
    
    # ------------------------------------------------------------------------
    # CONTEXT REGISTRATION
    # ------------------------------------------------------------------------
    
    def register_context(self, context: SessionContext) -> None:
        """Registra un context attivo."""
        self._active_contexts[context.context_id] = context
        logger.debug(f"Context AI registrato: {context.context_id}")
    
    def unregister_context(self, context_id: str) -> None:
        """Rimuove context dal registry."""
        self._active_contexts.pop(context_id, None)
        logger.debug(f"Context rimosso: {context_id}")
    
    def get_active_contexts(self) -> list[SessionContext]:
        """Lista tutti i context attivi."""
        return list(self._active_contexts.values())
    
    def get_context_count(self) -> int:
        """Conta context attivi."""
        return len(self._active_contexts)
    
    # ------------------------------------------------------------------------
    # AI METRICS
    # ------------------------------------------------------------------------
    
    def record_ai_metric(self, workspace_id: str, metric_name: str, value: Any) -> None:
        """Registra metrica AI per workspace."""
        if workspace_id not in self._ai_performance_metrics:
            self._ai_performance_metrics[workspace_id] = {}
        
        self._ai_performance_metrics[workspace_id][metric_name] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_ai_metrics(self, workspace_id: str) -> Dict[str, Any]:
        """Ottiene metriche AI per workspace."""
        return self._ai_performance_metrics.get(workspace_id, {})
    
    def get_ai_workspace_stats(self) -> Dict[str, Any]:
        """Ottiene statistiche workspace AI."""
        type_counts: Dict[str, int] = {}
        for ws_type in self._ai_workspace_types.values():
            type_counts[ws_type] = type_counts.get(ws_type, 0) + 1
        
        return {
            "total_workspaces": len(self.list_workspace_ids()),
            "workspace_types": type_counts,
            "active_contexts": len(self._active_contexts),
            "ai_metrics_tracked": len(self._ai_performance_metrics),
        }


# ============================================================================
# SESSION MANAGER - AGGIORNATO
# ============================================================================

class SessionManager:
    """
    Manager centrale per gestione sessioni e workspace AI-enhanced.
    
    Responsabilità AI-Enhanced:
    - AI workspace type management
    - LLM model configuration
    - Knowledge Graph setup
    - RAG session optimization
    - Multi-agent workspace orchestration
    
    Singleton pattern - usa get_instance() per accesso globale.
    
    Architecture AI-Enhanced:
        SessionManager
        ├── AI Workspace Registry (type-specific tracking)
        ├── Workspaces (AI-enhanced)
        │   ├── Backends (SQL, Vector, LLM)
        │   └── AI Components (Agents, KG, RAG)
        └── Contexts (AI session state)
    
    Usage:
        manager = SessionManager(base_path="./data")
        await manager.initialize()
        
        # Create AI workspace
        ws = await manager.create_ai_workspace(
            name="Code Analysis AI",
            workspace_type="code_analysis"
        )
        
        # Activate with AI config
        async with manager.activate_workspace(ws.id) as ctx:
            # AI repositories available
            code_repo = ctx.get_repository("code_sources")
            llm_repo = ctx.get_llm_repository()
    """
    
    _instance: Optional["SessionManager"] = None
    
    def __init__(
        self,
        base_path: Path | str,
        default_backend_configs: Optional[dict[str, BackendConfig]] = None,
        lifecycle_hooks: Optional[list[WorkspaceLifecycleHook]] = None,
        ai_config: Optional[Dict[str, Any]] = None,  # NUOVO
    ):
        """
        Args:
            base_path: Path base per tutti i workspace
            default_backend_configs: Config backend default per nuovi workspace
            lifecycle_hooks: Hooks da applicare a tutti i workspace
            ai_config: Configurazione AI globale (model defaults, etc.)
        """
        self.base_path = Path(base_path)
        self.default_backend_configs = default_backend_configs or self._get_default_backends()
        self.lifecycle_hooks = lifecycle_hooks or []
        self.ai_config = ai_config or self._get_default_ai_config()  # NUOVO
        
        self.storage_root = _default_workspace_storage_dir()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        # Registry AI-enhanced
        self._registry = SessionRegistry()

        # Session tracking
        self._sessions: dict[str, "Session"] = {}
        
        # State
        self._initialized = False
        self._shutdown_requested = False
        
        logger.debug(f"SessionManager AI creato: base_path={self.base_path}")

    def _get_context_for_workspace(self, workspace_id: str) -> Optional[SessionContext]:
        """Restituisce il context attivo per un workspace, se presente."""
        for ctx in self._registry.get_active_contexts():
            if ctx.workspace_id == workspace_id:
                return ctx
        return None

    def _workspace_storage_path(self, workspace_id: str) -> Path:
        """Percorso fisico dedicato al workspace sotto la storage root."""
        return self.storage_root / workspace_id

    @property
    def current_workspace_id(self) -> Optional[str]:
        """Workspace attivo nel contesto corrente (se presente)."""
        ctx = SessionContext.current()
        if ctx:
            return ctx.workspace.id
        return None

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Ritorna il workspace (se già caricato in memoria)."""
        return self._registry.get_workspace(workspace_id)

    def get_workspace_path(self, workspace_id: str) -> Optional[Path]:
        """Ritorna il path registrato per un workspace."""
        return self._registry.get_workspace_path(workspace_id)

    def get_workspace_type(self, workspace_id: str) -> str:
        """Ritorna il tipo di workspace registrato."""
        return self._registry.get_workspace_type(workspace_id)
    
    # ------------------------------------------------------------------------
    # SINGLETON HELPERS
    # ------------------------------------------------------------------------
    
    @classmethod
    def get_instance(cls) -> "SessionManager":
        """
        Restituisce l'istanza globale del SessionManager.
        
        Raises:
            SessionError: se l'istanza non è stata inizializzata.
        """
        if cls._instance is None:
            raise SessionError(
                "SessionManager non inizializzato. "
                "Crea un'istanza e chiama 'await initialize()' prima di usare get_instance()."
            )
        return cls._instance
    
    @classmethod
    def set_instance(cls, instance: Optional["SessionManager"]) -> None:
        """Imposta (o resetta) l'istanza globale del SessionManager."""
        cls._instance = instance
    
    # ========================================================================
    # AI WORKSPACE CREATION - NUOVO
    # ========================================================================
    
    async def create_ai_workspace(
        self,
        name: str,
        workspace_type: str,
        workspace_id: Optional[str] = None,
        description: str = "",
        backends: Optional[dict[str, BackendConfig]] = None,
        tags: Optional[list[str]] = None,
        ai_settings: Optional[Dict[str, Any]] = None,
    ) -> Workspace:
        """
        Crea un nuovo workspace AI-enhanced.
        
        Args:
            name: Nome workspace
            workspace_type: Tipo workspace AI (code_analysis, log_analysis, etc.)
            workspace_id: ID custom (auto-generato se None)
            description: Descrizione
            backends: Config backend (usa default se None)
            tags: Tag per categorizzazione
            ai_settings: Settings AI custom (sovrascrive defaults)
        
        Returns:
            Workspace AI creato e inizializzato
        """
        # Verifica tipo workspace supportato
        if workspace_type not in AIWorkspaceConfig.WORKSPACE_TYPES:
            raise ValueError(
                f"Tipo workspace non supportato: {workspace_type}. "
                f"Tipi supportati: {list(AIWorkspaceConfig.WORKSPACE_TYPES.keys())}"
            )
        
        # Genera ID se non fornito
        if workspace_id is None:
            workspace_id = self._generate_ai_workspace_id(name, workspace_type)
        
        # Verifica se esiste in memoria
        if self._registry.has_workspace(workspace_id):
            raise WorkspaceAlreadyExistsError(
                workspace_id=workspace_id,
                reason="Workspace già registrato in memoria",
            )
        
        # Verifica se esiste su disco
        existing_path = self._registry.get_workspace_path(workspace_id)
        if existing_path and existing_path.exists():
            raise WorkspaceAlreadyExistsError(
                workspace_id=workspace_id,
                reason=f"Workspace directory già esistente: {existing_path}",
            )
        
        logger.info(f"Creazione workspace AI: {workspace_id} ({name}, type: {workspace_type})")
        
        # Configurazione AI di default + override custom
        ai_defaults = AIWorkspaceConfig.get_default_ai_settings(workspace_type)
        final_ai_settings = {**ai_defaults, **(ai_settings or {})}
        final_ai_config_dict = final_ai_settings.get("ai_config", {})
        
        # Instanzia AIWorkspaceSettings
        ai_workspace_settings = AIWorkspaceSettings.from_dict(
            {
                "workspace_type": final_ai_config_dict.get("workspace_type", workspace_type),
                "enabled_features": final_ai_config_dict.get("enabled_features", []),
                "knowledge_graph_enabled": final_ai_config_dict.get("knowledge_graph_enabled", False),
                "rag_enabled": final_ai_config_dict.get("rag_enabled", False),
                "default_models": final_ai_config_dict.get("default_models", ["gpt-4"]),
                "agent_settings": final_ai_config_dict.get("agent_settings", {}),
            }
        )
        
        # Crea metadata con configurazione AI
        metadata = WorkspaceMetadata(
            id=workspace_id,
            name=name,
            description=description,
            tags=(tags or []) + [f"ai_{workspace_type}"],  # Auto-tag AI
            settings=final_ai_settings,
            ai_config=ai_workspace_settings.to_dict(),
        )

        root_dir = (final_ai_settings or {}).get("root_dir")
        if root_dir:
            metadata.settings["root_dir"] = root_dir
        metadata.root_dir = root_dir or metadata.root_dir
        
        # Path workspace
        workspace_path = self._workspace_storage_path(workspace_id)
        workspace_path.mkdir(parents=True, exist_ok=True)
        metadata.workspace_path = str(workspace_path)
        metadata.settings["workspace_path"] = metadata.workspace_path
        metadata.workspace_path = str(workspace_path)
        metadata.settings["workspace_path"] = metadata.workspace_path
        
        # Configurazione backend con supporto AI
        backend_configs = backends or self._get_ai_backend_configs(workspace_type)
        
        # Crea workspace
        workspace = Workspace(
            id=workspace_id,
            name=name,
            base_path=workspace_path,
            backends=backend_configs,
            metadata=metadata,
            lifecycle_hooks=self.lifecycle_hooks.copy(),
            ai_config=ai_workspace_settings,
        )
        
        # Inizializza (crea directory, backends, AI components)
        await workspace.initialize()
        
        # Registra con tipo AI
        self._registry.register_workspace(workspace, workspace_type)
        self._registry.record_ai_metric(workspace_id, "creation_time", datetime.now().isoformat())
        self._registry.record_ai_metric(workspace_id, "workspace_type", workspace_type)
        
        logger.info(f"Workspace AI creato: {workspace_id} (type: {workspace_type})")
        
        return workspace
    
    async def create_workspace(
        self,
        name: str,
        workspace_id: Optional[str] = None,
        description: str = "",
        backends: Optional[dict[str, BackendConfig]] = None,
        tags: Optional[list[str]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> Workspace:
        """
        Crea un nuovo workspace (compatibilità backward).
        
        Args:
            name: Nome workspace
            workspace_id: ID custom (auto-generato se None)
            description: Descrizione
            backends: Config backend (usa default se None)
            tags: Tag per categorizzazione
            settings: Settings custom (può includere ai_config)
        
        Returns:
            Workspace creato e inizializzato
        """
        # Se settings contiene AI config, usa create_ai_workspace
        if settings and settings.get("ai_config"):
            workspace_type = settings.get("workspace_type", "generic")
            return await self.create_ai_workspace(
                name=name,
                workspace_type=workspace_type,
                workspace_id=workspace_id,
                description=description,
                backends=backends,
                tags=tags,
                ai_settings=settings,
            )
        
        # Altrimenti crea workspace generico
        return await self._create_generic_workspace(
            name=name,
            workspace_id=workspace_id,
            description=description,
            backends=backends,
            tags=tags,
            settings=settings,
        )
    
    async def _create_generic_workspace(
        self,
        name: str,
        workspace_id: Optional[str] = None,
        description: str = "",
        backends: Optional[dict[str, BackendConfig]] = None,
        tags: Optional[list[str]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> Workspace:
        """Crea workspace generico (implementazione originale estesa)."""
        # Genera ID se non fornito
        if workspace_id is None:
            workspace_id = self._generate_workspace_id(name)
        
        # Verifica se esiste in memoria
        if self._registry.has_workspace(workspace_id):
            raise WorkspaceAlreadyExistsError(
                workspace_id=workspace_id,
                reason="Workspace già registrato",
            )
        
        # Verifica se esiste su disco
        existing_path = self._registry.get_workspace_path(workspace_id)
        if existing_path and existing_path.exists():
            raise WorkspaceAlreadyExistsError(
                workspace_id=workspace_id,
                reason=f"Workspace directory già esistente: {existing_path}",
            )
        
        logger.info(f"Creazione workspace: {workspace_id} ({name})")
        
        # Crea metadata
        metadata = WorkspaceMetadata(
            id=workspace_id,
            name=name,
            description=description,
            tags=tags or [],
            settings=settings or {},
        )

        root_dir = (settings or {}).get("root_dir")
        if root_dir:
            metadata.settings["root_dir"] = root_dir
        metadata.root_dir = root_dir or metadata.root_dir
        
        # Path workspace
        workspace_path = self._workspace_storage_path(workspace_id)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Usa backend di default o forniti
        backend_configs = backends or self.default_backend_configs.copy()
        
        # Crea workspace generico (ai_config default)
        workspace = Workspace(
            id=workspace_id,
            name=name,
            base_path=workspace_path,
            backends=backend_configs,
            metadata=metadata,
            lifecycle_hooks=self.lifecycle_hooks.copy(),
        )
        
        # Inizializza
        await workspace.initialize()
        
        # Registra come generic
        self._registry.register_workspace(workspace, "generic")
        self._registry.record_ai_metric(workspace_id, "creation_time", datetime.now().isoformat())
        self._registry.record_ai_metric(workspace_id, "workspace_type", "generic")
        
        logger.info(f"Workspace creato: {workspace_id}")
        
        return workspace
    
    # ========================================================================
    # AI WORKSPACE MANAGEMENT - NUOVO
    # ========================================================================
    
    async def list_ai_workspaces(
        self,
        workspace_type: Optional[str] = None,
        include_inactive: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Lista workspace AI con filtering per tipo.
        
        Args:
            workspace_type: Filtra per tipo specifico (None = tutti)
            include_inactive: Se True, include anche workspace non caricati
        
        Returns:
            Lista di dict con info workspace AI
        """
        all_workspaces = await self.list_workspaces(include_inactive=include_inactive)
        
        if workspace_type is None:
            return all_workspaces
        
        # Filtra per tipo
        return [
            ws for ws in all_workspaces
            if ws.get("workspace_type") == workspace_type
        ]
    
    async def get_ai_workspace_info(self, workspace_id: str) -> dict[str, Any]:
        """
        Ottiene info dettagliate su workspace AI.
        
        Args:
            workspace_id: ID workspace
        
        Returns:
            Dict con info workspace AI-enhanced
        """
        base_info = await self.get_workspace_info(workspace_id)
        
        # Aggiungi info AI
        ai_info = {
            "workspace_type": self._registry.get_workspace_type(workspace_id),
            "ai_metrics": self._registry.get_ai_metrics(workspace_id),
            "ai_config": base_info.get("ai_config", {}),
        }
        
        # base_info già contiene id, name, state, ecc.
        base_info.update(ai_info)
        return base_info
    
    def get_ai_workspace_stats(self) -> Dict[str, Any]:
        """
        Ottiene statistiche workspace AI.
        
        Returns:
            Dict con stats AI-specifiche
        """
        return self._registry.get_ai_workspace_stats()
    
    # ========================================================================
    # SESSION ACTIVATION AI-ENHANCED - AGGIORNATO
    # ========================================================================
    
    async def activate_ai_workspace(
        self,
        workspace_id: str,
        ai_session_config: Optional[Dict[str, Any]] = None,
        auto_load: bool = True,
    ) -> SessionContext:
        """
        Attiva un workspace AI con configurazione ottimizzata.
        
        Args:
            workspace_id: ID workspace da attivare
            ai_session_config: Configurazione sessione AI-optimized
            auto_load: Se True, carica workspace se non già caricato
        
        Returns:
            SessionContext AI-optimized
        """
        # Ottieni o carica workspace
        if auto_load:
            workspace = await self.get_or_load_workspace(workspace_id)
        else:
            workspace = self._registry.get_workspace(workspace_id)
            if not workspace:
                raise WorkspaceNotFoundError(
                    workspace_id=workspace_id,
                    reason="Workspace non caricato (usa auto_load=True)",
                )
        
        # Se esiste già un context attivo per questo workspace, riusalo
        existing_ctx = self._get_context_for_workspace(workspace_id)
        if existing_ctx:
            SessionContext.set_current(existing_ctx)
            return existing_ctx

        # Configurazione sessione AI-optimized
        workspace_type = self._registry.get_workspace_type(workspace_id)
        session_config = self._create_ai_session_config(workspace_type, ai_session_config)
        
        # Verifica stato
        if not workspace.is_active:
            if workspace.state == WorkspaceState.SUSPENDED:
                await workspace.resume()
            else:
                raise WorkspaceError(
                    workspace_id=workspace_id,
                    message=f"Workspace in stato non valido: {workspace.state.value}",
                )
        
        # Crea context AI-optimized
        context = SessionContext.create(
            workspace=workspace,
            config=session_config,
            set_as_current=True,
            workspace_id=workspace_id,
        )
        
        # Registra context
        self._registry.register_context(context)
        
        # Setup cleanup callback
        context.on_close(lambda ctx: self._registry.unregister_context(ctx.context_id))
        
        # Record AI activation metric
        metrics = self._registry.get_ai_metrics(workspace_id)
        prev_activations = metrics.get("activation_count", {}).get("value", 0)
        self._registry.record_ai_metric(workspace_id, "last_activation", datetime.now().isoformat())
        self._registry.record_ai_metric(workspace_id, "activation_count", prev_activations + 1)
        
        logger.info(
            f"Workspace AI attivato: {workspace_id} "
            f"(type: {workspace_type}, context: {context.context_id})"
        )
        
        return context
    
    async def activate_workspace(
        self,
        workspace_id: str,
        config: Optional[SessionConfig] = None,
        auto_load: bool = True,
    ) -> SessionContext:
        """
        Attiva un workspace (compatibilità backward).
        
        Se il workspace è AI-enhanced, usa automaticamente configurazione ottimizzata.
        """
        workspace_type = self._registry.get_workspace_type(workspace_id)
        
        if workspace_type != "generic":
            # Usa attivazione AI-optimized
            ai_config = config.to_dict() if config else {}
            return await self.activate_ai_workspace(workspace_id, ai_config, auto_load)
        
        # Altrimenti usa attivazione standard
        return await self._activate_generic_workspace(workspace_id, config, auto_load)

    async def deactivate_workspace(self, workspace_id: str) -> bool:
        """
        Deattiva un workspace senza eliminarlo.
        Chiude context, chiude workspace, salva stato, e rimuove dal registry attivo.
        """
        logger.info(f"[SessionManager] Deactivation requested for workspace {workspace_id}")

        # 1. Trova eventuale context attivo
        ctx = self._get_context_for_workspace(workspace_id)
        if ctx:
            try:
                logger.debug(f"[SessionManager] Closing context {ctx.context_id}")
                await ctx.close()
            except Exception as e:
                logger.error(f"[SessionManager] Error closing context {ctx.context_id}: {e}")

            # deregistra context
            try:
                self._registry.unregister_context(ctx.context_id)
            except Exception:
                pass

        # 2. Trova il workspace stesso
        ws = self._registry.get_workspace(workspace_id)
        if not ws:
            logger.warning(f"[SessionManager] Workspace {workspace_id} not loaded, nothing to deactivate")
            return True

        # 3. Chiudi workspace attivo (teardown)
        if not ws.is_closed:
            try:
                logger.debug(f"[SessionManager] Closing workspace {workspace_id}")
                await ws.close()
            except Exception as e:
                logger.error(f"[SessionManager] Error closing workspace {workspace_id}: {e}")

        # 4. Salva eventuali cambiamenti persistenti
        try:
            ws.save_metadata()
        except Exception as e:
            logger.error(f"[SessionManager] Error saving metadata for workspace {workspace_id}: {e}")

        # 5. Rimuovi workspace dal registry attivo,
        #    ma NON cancellare la directory su disco
        self._registry.unregister_workspace(workspace_id)

        logger.info(f"[SessionManager] Workspace {workspace_id} deactivated successfully")
        return True
    
    async def _activate_generic_workspace(
        self,
        workspace_id: str,
        config: Optional[SessionConfig] = None,
        auto_load: bool = True,
    ) -> SessionContext:
        """Attiva workspace generico (implementazione originale)."""
        # Ottieni o carica workspace
        if auto_load:
            workspace = await self.get_or_load_workspace(workspace_id)
        else:
            workspace = self._registry.get_workspace(workspace_id)
            if not workspace:
                raise WorkspaceNotFoundError(
                    workspace_id=workspace_id,
                    reason="Workspace non caricato (usa auto_load=True)",
                )

        existing_ctx = self._get_context_for_workspace(workspace_id)
        if existing_ctx:
            SessionContext.set_current(existing_ctx)
            return existing_ctx

        # Verifica stato
        if not workspace.is_active:
            if workspace.state == WorkspaceState.SUSPENDED:
                await workspace.resume()
            else:
                raise WorkspaceError(
                    workspace_id=workspace_id,
                    message=f"Workspace in stato non valido: {workspace.state.value}",
                )
        
        # Crea context
        context = SessionContext.create(
            workspace=workspace,
            config=config,
            set_as_current=True,
            workspace_id=workspace_id,
        )
        
        # Registra context
        self._registry.register_context(context)
        
        # Setup cleanup callback
        context.on_close(lambda ctx: self._registry.unregister_context(ctx.context_id))
        
        logger.info(f"Workspace attivato: {workspace_id} (context={context.context_id})")
        
        return context
    
    # ========================================================================
    # HELPERS AI-ENHANCED - NUOVO
    # ========================================================================
    
    def _generate_ai_workspace_id(self, name: str, workspace_type: str) -> str:
        """
        Genera ID workspace AI da nome e tipo.
        """
        import re
        import uuid
        
        # Slug da nome
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        slug = slug[:20]  # Max 20 chars
        
        # Tipo abbreviato
        type_abbr = workspace_type[:3] if len(workspace_type) > 3 else workspace_type
        
        # Aggiungi short UUID per unicità
        short_id = uuid.uuid4().hex[:6]
        
        return f"ai_{type_abbr}_{slug}_{short_id}"
    
    def _get_ai_backend_configs(self, workspace_type: str) -> dict[str, BackendConfig]:
        """
        Ottiene configurazione backend ottimizzata per tipo workspace AI.

        Per tutti i workspace AI garantiamo sempre la presenza di:
        - backend relazionale (SQL)
        - backend vettoriale (Chroma)
        - backend di cache LLM ("llm_cache")
        """
        base_configs = self._get_default_backends()

        # Garantisce un backend di cache LLM dedicato per workspace AI
        if "llm_cache" not in base_configs:
            # File dedicato per la cache LLM del workspace
            base_configs["llm_cache"] = BackendConfig.sqlite("llm_cache.db")

        # Ottimizzazioni specifiche per tipo di workspace AI
        optimizations: Dict[str, Dict[str, BackendConfig]] = {
            "code_analysis": {
                "relational": BackendConfig.sqlite("code_analysis.db"),
            },
            "log_analysis": {
                "relational": BackendConfig.sqlite("log_analysis.db"),
            },
            "multi_agent": {
                "relational": BackendConfig.sqlite("agent_workflows.db"),
            },
            "knowledge_base": {
                "relational": BackendConfig.sqlite("knowledge_graph.db"),
                "vector": BackendConfig.memory_cache(),
            },
        }

        if workspace_type in optimizations:
            base_configs.update(optimizations[workspace_type])

        return base_configs
    
    def _create_ai_session_config(
        self,
        workspace_type: str,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> SessionConfig:
        """
        Crea configurazione sessione ottimizzata per tipo workspace AI.
        """
        # Configurazioni ottimizzate per tipo
        optimized_configs: Dict[str, Dict[str, Any]] = {
            "code_analysis": {
                "enable_ai_features": True,
                "enable_rag": True,
                "enable_knowledge_graph": True,
                "max_llm_context": 16000,  # Codice richiede più context
                "rag_top_k": 8,
            },
            "log_analysis": {
                "enable_ai_features": True,
                "enable_rag": True,
                "enable_knowledge_graph": True,
                "max_llm_context": 8000,
                "rag_top_k": 5,
            },
            "multi_agent": {
                "enable_ai_features": True,
                "enable_rag": True,
                "enable_knowledge_graph": True,
                "max_llm_context": 32000,
                "rag_top_k": 10,
            },
            "knowledge_base": {
                "enable_ai_features": True,
                "enable_rag": True,
                "enable_knowledge_graph": True,
                "max_llm_context": 8000,
                "rag_top_k": 15,
            },
        }
        
        # Configurazione di base
        base_config = SessionConfig()
        
        # Applica ottimizzazioni per tipo
        if workspace_type in optimized_configs:
            for key, value in optimized_configs[workspace_type].items():
                # Aggiorna in settings avanzate
                base_config.set(key, value)
        
        # Applica configurazione custom
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
                else:
                    base_config.set(key, value)
        
        return base_config
    
    def _get_default_ai_config(self) -> Dict[str, Any]:
        """Ottiene configurazione AI di default."""
        return {
            "default_llm_provider": "openai",
            "embedding_model": "text-embedding-ada-002",
            "max_concurrent_llm_requests": 5,
            "enable_agent_orchestration": True,
            "knowledge_graph_auto_update": True,
        }

    # ========================================================================
    # SESSION MANAGEMENT (Lightweight)
    # ========================================================================

    def create_session(
        self,
        workspace: Workspace,
        config: Optional[SessionConfig] = None,
        metadata: Optional[dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> "Session":
        """
        Crea una nuova sessione e registra context + lifecycle.
        """
        context = SessionContext.create(
            workspace=workspace,
            config=config,
            metadata=metadata,
            expires_at=expires_at,
            set_as_current=False,
            workspace_id=workspace.id,
        )
        lifecycle = SessionLifecycle(metadata=metadata)

        session = Session(id=context.context_id, context=context, lifecycle=lifecycle)
        self._sessions[session.id] = session
        self._registry.register_context(context)
        return session

    def get_session(self, session_id: str) -> Optional["Session"]:
        """Ottiene una sessione dal registry locale."""
        return self._sessions.get(session_id)

    def close_session(self, session_id: str) -> None:
        """Chiude una sessione attiva e aggiorna il lifecycle."""
        session = self._sessions.get(session_id)
        if not session:
            return

        lifecycle = session.lifecycle
        # Forza attivazione se ancora in CREATED per permettere il finish
        if lifecycle.state == "CREATED":
            try:
                lifecycle.transition("ACTIVE")
            except LifecycleError:
                pass
        if lifecycle.state in {"ACTIVE", "PAUSED"}:
            lifecycle.transition("FINISHED")

        # Chiudi il context (async)
        coro = session.context.close()
        if asyncio.iscoroutine(coro):
            # NOTA: in contesti async veri è meglio usare 'await'
            asyncio.run(coro)

        self._registry.unregister_context(session.context.context_id)

    def delete_session(self, session_id: str) -> None:
        """Elimina una sessione dal manager (context + registry)."""
        session = self._sessions.get(session_id)
        if not session:
            return

        # Assicura chiusura
        self.close_session(session_id)
        self._sessions.pop(session_id, None)

    def cleanup_expired_sessions(self) -> None:
        """Rimuove sessioni scadute e non attive."""
        expired_ids: list[str] = []
        for session_id, session in list(self._sessions.items()):
            if session.context.is_expired():
                expired_ids.append(session_id)

        for session_id in expired_ids:
            self.close_session(session_id)
            self._sessions.pop(session_id, None)
    
    # ========================================================================
    # DISCOVERY & WORKSPACE MANAGEMENT COMPLETO
    # ========================================================================
    
    async def _discover_workspaces(self) -> None:
        """
        Scopre workspace esistenti su disco e registra path + tipi.
        
        - Scansiona la cartella base
        - Cerca 'workspace.json'
        - Carica metadata via Workspace.load_metadata
        - Deduce workspace_type da ai_config/settings
        - Registra path e tipo nel registry (senza caricare i workspace)
        """
        if not self.storage_root.exists():
            logger.debug("Base path non esiste ancora, nessun workspace da scoprire.")
            return
        
        for entry in self.storage_root.iterdir():
            if not entry.is_dir():
                continue
            
            try:
                metadata = Workspace.load_metadata(entry)
                workspace_id = metadata.id
                
                # Deduci tipo dal metadata
                ai_cfg = metadata.ai_config or metadata.settings.get("ai_config", {})
                workspace_type = (
                    ai_cfg.get("workspace_type")
                    or metadata.settings.get("workspace_type")
                    or "generic"
                )
                
                # Registra path + tipo
                self._registry.register_workspace_path(
                    workspace_id=workspace_id,
                    path=entry,
                    workspace_type=workspace_type,
                )
                
                # Metriche di discovery
                self._registry.record_ai_metric(workspace_id, "discovered_at", datetime.now().isoformat())
                
                logger.debug(
                    f"Workspace scoperto su disco: {workspace_id} "
                    f"(type={workspace_type}, path={entry})"
                )
            
            except WorkspaceNotFoundError:
                # Nessun workspace.json, skip silenzioso
                continue
            except Exception as e:
                logger.warning(f"Errore durante discovery workspace in {entry}: {e}")
    
    async def load_workspace(self, workspace_id: str) -> Workspace:
        """
        Carica un workspace esistente (AI-aware) da disco e lo inizializza.
        
        Se è già caricato, restituisce l'istanza esistente.
        """
        # Verifica se già caricato
        workspace = self._registry.get_workspace(workspace_id)
        if workspace:
            logger.debug(
                f"Workspace già caricato: {workspace_id} "
                f"(type: {self._registry.get_workspace_type(workspace_id)})"
            )
            return workspace
        
        # Assicurati che la discovery abbia popolato i path
        await self._discover_workspaces()
        
        # Determina path del workspace
        workspace_path = self._registry.get_workspace_path(workspace_id)
        if not workspace_path:
            workspace_path = self._workspace_storage_path(workspace_id)
        
        if not workspace_path.exists():
            raise WorkspaceNotFoundError(
                workspace_id=workspace_id,
                reason=f"Directory workspace non trovata: {workspace_path}",
            )
        
        # Carica metadata
        try:
            metadata = WorkspaceMetadata.from_dict(
                Workspace.load_metadata(workspace_path).to_dict()
            )
        except Exception as e:
            logger.error(f"Errore caricamento metadata per workspace {workspace_id}: {e}")
            raise WorkspaceError(
                workspace_id=workspace_id,
                message="Impossibile caricare metadata workspace",
                cause=e,
            ) from e
        
        # Deduci configurazione AI
        ai_cfg_dict = metadata.ai_config or metadata.settings.get("ai_config", {})
        workspace_type = (
            ai_cfg_dict.get("workspace_type")
            or metadata.settings.get("workspace_type")
            or self._registry.get_workspace_type(workspace_id)
            or "generic"
        )
        
        ai_settings: Optional[AIWorkspaceSettings] = None
        if ai_cfg_dict:
            ai_settings = AIWorkspaceSettings.from_dict(
                {
                    "workspace_type": ai_cfg_dict.get("workspace_type", workspace_type),
                    "enabled_features": ai_cfg_dict.get("enabled_features", []),
                    "knowledge_graph_enabled": ai_cfg_dict.get("knowledge_graph_enabled", False),
                    "rag_enabled": ai_cfg_dict.get("rag_enabled", False),
                    "default_models": ai_cfg_dict.get("default_models", ["gpt-4"]),
                    "agent_settings": ai_cfg_dict.get("agent_settings", {}),
                }
            )
        
        # Backend configs: AI-ottimizzati o generici
        if workspace_type != "generic":
            backend_configs = self._get_ai_backend_configs(workspace_type)
        else:
            backend_configs = self.default_backend_configs.copy()
        
        # Instanzia Workspace
        workspace = Workspace(
            id=metadata.id,
            name=metadata.name,
            base_path=workspace_path,
            backends=backend_configs,
            metadata=metadata,
            lifecycle_hooks=self.lifecycle_hooks.copy(),
            ai_config=ai_settings,
        )
        
        # Inizializza (riconnette backends, inizializza schema se necessario)
        await workspace.initialize()
        
        # Registra in registry
        self._registry.register_workspace(workspace, workspace_type)
        self._registry.record_ai_metric(workspace_id, "loaded_from_disk", True)
        
        logger.info(f"Workspace caricato da disco: {workspace_id} (type={workspace_type})")
        
        return workspace
    
    async def get_or_load_workspace(self, workspace_id: str) -> Workspace:
        """
        Restituisce il workspace se già caricato, altrimenti lo carica da disco.
        """
        ws = self._registry.get_workspace(workspace_id)
        if ws:
            return ws
        return await self.load_workspace(workspace_id)
    
    async def list_workspaces(
        self,
        include_inactive: bool = True,
        include_hidden: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Lista tutti i workspace conosciuti (caricati o solo scoperti).
        
        Returns:
            Lista di dict con info workspace:
                - id, name, state, base_path
                - workspace_type
                - metadata, ai_config
                - loaded (bool)
        """
        await self._discover_workspaces()
        
        results: list[dict[str, Any]] = []
        for workspace_id in self._registry.list_workspace_ids():
            loaded_ws = self._registry.get_workspace(workspace_id)
            path = self._registry.get_workspace_path(workspace_id) or self._workspace_storage_path(workspace_id)
            ws_type = self._registry.get_workspace_type(workspace_id)

            if not include_hidden and workspace_id == GLOBAL_KNOWLEDGE_WORKSPACE_ID:
                continue
            
            if loaded_ws:
                info = loaded_ws.get_info()
                info["loaded"] = True
                info["workspace_type"] = ws_type
                results.append(info)
                continue
            
            if not include_inactive:
                continue
            
            # workspace non caricato: costruisci info minimale da metadata
            try:
                metadata = Workspace.load_metadata(path)
                ai_cfg = metadata.ai_config or metadata.settings.get("ai_config", {})
                ai_cfg_ws_type = (
                    ai_cfg.get("workspace_type")
                    or metadata.settings.get("workspace_type")
                    or ws_type
                    or "generic"
                )
                
                results.append(
                    {
                        "id": metadata.id,
                        "name": metadata.name,
                        "state": "inactive",
                        "base_path": str(path),
                        "workspace_type": ai_cfg_ws_type,
                        "metadata": metadata.to_dict(),
                        "ai_config": ai_cfg,
                        "backends": [],
                        "repositories": [],
                        "ai_components": [],
                        "ai_metrics": self._registry.get_ai_metrics(workspace_id),
                        "loaded": False,
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Impossibile leggere metadata per workspace {workspace_id} in {path}: {e}"
                )
        
        return results
    
    async def get_workspace_info(self, workspace_id: str) -> dict[str, Any]:
        """
        Ottiene informazioni dettagliate su un workspace, anche se non caricato.
        """
        # Se è caricato, usa direttamente get_info()
        ws = self._registry.get_workspace(workspace_id)
        if ws:
            info = ws.get_info()
            info["loaded"] = True
            info["workspace_type"] = self._registry.get_workspace_type(workspace_id)
            return info
        
        await self._discover_workspaces()
        
        path = self._registry.get_workspace_path(workspace_id) or self._workspace_storage_path(workspace_id)
        if not path.exists():
            raise WorkspaceNotFoundError(
                workspace_id=workspace_id,
                reason=f"Workspace non trovato su disco: {path}",
            )
        
        try:
            metadata = Workspace.load_metadata(path)
        except Exception as e:
            raise WorkspaceError(
                workspace_id=workspace_id,
                message="Impossibile leggere metadata workspace",
                cause=e,
            ) from e
        
        ai_cfg = metadata.ai_config or metadata.settings.get("ai_config", {})
        ws_type = (
            ai_cfg.get("workspace_type")
            or metadata.settings.get("workspace_type")
            or self._registry.get_workspace_type(workspace_id)
            or "generic"
        )
        
        return {
            "id": metadata.id,
            "name": metadata.name,
            "state": "inactive",
            "base_path": str(path),
            "workspace_type": ws_type,
            "metadata": metadata.to_dict(),
            "ai_config": ai_cfg,
            "backends": [],
            "repositories": [],
            "ai_components": [],
            "ai_metrics": self._registry.get_ai_metrics(workspace_id),
            "loaded": False,
        }
    
    async def delete_workspace(self, workspace_id: str, delete_from_disk: bool = True) -> bool:
        """
        Elimina un workspace:
        - chiude se caricato
        - rimuove dal registry
        - opzionalmente cancella directory su disco
        """
        ws = self._registry.get_workspace(workspace_id)
        
        # Chiudi workspace se attivo
        if ws and not ws.is_closed:
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Errore chiusura workspace {workspace_id} durante delete: {e}")

        # Chiudi context associati
        contexts = [
            ctx for ctx in self._registry.get_active_contexts()
            if ctx.workspace_id == workspace_id
        ]
        for ctx in contexts:
            try:
                await ctx.close()
            except Exception as e:
                logger.error(f"Errore chiusura context {ctx.context_id}: {e}")
            self._registry.unregister_context(ctx.context_id)

        # Rimuovi dal registry (in memoria)
        workspace_path = self._registry.get_workspace_path(workspace_id)
        self._registry.unregister_workspace(workspace_id)
        
        # Gestione su disco
        path = workspace_path or self._workspace_storage_path(workspace_id)
        if delete_from_disk and path.exists():
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Workspace directory rimossa: {path}")
            except Exception as e:
                logger.error(f"Errore rimozione directory workspace {path}: {e}")
        
        # Rimuovi path registrato
        self._registry.drop_workspace_path(workspace_id)

        # Rimuovi eventuali sessioni riferite al workspace
        for session_id, session in list(self._sessions.items()):
            if session.context.workspace_id == workspace_id:
                self._sessions.pop(session_id, None)

        logger.info(f"Workspace {workspace_id} eliminato")
        return True
    
    def get_stats(self) -> dict[str, Any]:
        """
        Ottiene statistiche globali AI-enhanced.
        """
        base_stats = {
            "initialized": self._initialized,
            "base_path": str(self.base_path),
            "ai_config": self.ai_config,
        }
        
        ai_stats = self._registry.get_ai_workspace_stats()
        
        return {**base_stats, **ai_stats}

    async def set_workspace_panel(self, workspace_id: str, panel: str) -> None:
        """
        Aggiorna il pannello UI attivo per un workspace se il context è attivo.
        """
        ctx = self._get_context_for_workspace(workspace_id)
        if ctx:
            ctx.set_panel_context(panel)
            logger.info(
                "workspace.panel.updated",
                extra={"workspace_id": workspace_id, "panel_context": panel},
            )
    
    # ========================================================================
    # METODI ESISTENTI (INIZIALIZZAZIONE / SHUTDOWN)
    # ========================================================================
    
    async def initialize(self) -> None:
        """Inizializza il manager AI-enhanced."""
        if self._initialized:
            logger.warning("SessionManager già inizializzato")
            return
        
        try:
            logger.info(f"Inizializzazione SessionManager AI: {self.base_path}")
            
            # Crea directory base
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Scopri workspace esistenti
            await self._discover_workspaces()
            
            self._initialized = True
            
            # Set come singleton
            SessionManager.set_instance(self)
            
            stats = self._registry.get_ai_workspace_stats()
            logger.info(
                "SessionManager AI inizializzato: "
                f"{stats['total_workspaces']} workspace trovati, "
                f"tipi: {stats['workspace_types']}"
            )
        
        except Exception as e:
            raise SessionError(
                f"Errore inizializzazione SessionManager AI: {e}"
            ) from e
    
    async def shutdown(self) -> None:
        """
        Shutdown del manager.
        
        - Chiude tutti i context attivi
        - Chiude tutti i workspace
        - Cleanup
        """
        if self._shutdown_requested:
            logger.warning("Shutdown già in corso")
            return
        
        self._shutdown_requested = True
        
        logger.info("Shutdown SessionManager...")
        
        # Chiudi context attivi
        contexts = self._registry.get_active_contexts()
        for context in contexts:
            try:
                await context.close()
            except Exception as e:
                logger.error(f"Errore chiusura context {context.context_id}: {e}")
        
        # Chiudi workspace
        for workspace_id in list(self._registry.list_workspace_ids()):
            workspace = self._registry.get_workspace(workspace_id)
            if workspace:
                try:
                    await workspace.close()
                except Exception as e:
                    logger.error(f"Errore chiusura workspace {workspace_id}: {e}")
        
        self._initialized = False
        
        # Clear singleton
        if SessionManager._instance == self:
            SessionManager.set_instance(None)
        
        logger.info("SessionManager shut down completato")
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def _generate_workspace_id(self, name: str) -> str:
        """Genera ID workspace (compatibilità)."""
        import re
        import uuid
        
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        slug = slug[:30]
        short_id = uuid.uuid4().hex[:8]
        
        return f"ws_{slug}_{short_id}"
    
    def _get_default_backends(self) -> dict[str, BackendConfig]:
        """
        Ottiene configurazione backend di default AI-enhanced.

        Usa env var ICE_STUDIO_STORAGE_BACKEND per selezionare:
        - sqlite (default)
        - mysql
        - mariadb
        - postgres
        """
        backend_name = os.getenv("ICE_STUDIO_STORAGE_BACKEND", "sqlite").lower()

        # Backend relazionale
        if backend_name == "mysql":
            relational = BackendConfig(
                backend_type=BackendType.MYSQL,
                connection_string="mysql://{host}/{db}".format(
                    host=os.getenv("MYSQL_HOST", "localhost"),
                    db=os.getenv("MYSQL_DB", "cortex"),
                ),
                mode=StorageMode.PERSISTENT,
                options={
                    "host": os.getenv("MYSQL_HOST", "localhost"),
                    "port": int(os.getenv("MYSQL_PORT", "3306")),
                    "user": os.getenv("MYSQL_USER", "root"),
                    "password": os.getenv("MYSQL_PASSWORD", ""),
                    "database": os.getenv("MYSQL_DB", "cortex"),
                },
            )
        elif backend_name == "mariadb":
            relational = BackendConfig(
                backend_type=BackendType.MARIADB,
                connection_string="mariadb://{host}/{db}".format(
                    host=os.getenv("MARIADB_HOST", "localhost"),
                    db=os.getenv("MARIADB_DB", "cortex"),
                ),
                mode=StorageMode.PERSISTENT,
                options={
                    "host": os.getenv("MARIADB_HOST", "localhost"),
                    "port": int(os.getenv("MARIADB_PORT", "3306")),
                    "user": os.getenv("MARIADB_USER", "root"),
                    "password": os.getenv("MARIADB_PASSWORD", ""),
                    "database": os.getenv("MARIADB_DB", "cortex"),
                },
            )
        elif backend_name == "postgres":
            relational = BackendConfig(
                backend_type=BackendType.POSTGRES,
                connection_string="postgres://{host}/{db}".format(
                    host=os.getenv("POSTGRES_HOST", "localhost"),
                    db=os.getenv("POSTGRES_DB", "cortex"),
                ),
                mode=StorageMode.PERSISTENT,
                options={
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", ""),
                    "database": os.getenv("POSTGRES_DB", "cortex"),
                },
            )
        else:
            # Default: SQLite file sotto il workspace
            relational = BackendConfig.sqlite("data.db")

        # Backend vettoriale FAISS
        vector = BackendConfig.memory_cache()

        return {
            "relational": relational,
            "vector": vector,
            "llm_cache": BackendConfig.sqlite("llm_cache.db"),
        }

    
    def __repr__(self) -> str:
        ai_stats = self._registry.get_ai_workspace_stats()
        return (
            f"SessionManagerAI(base_path={self.base_path}, "
            f"workspaces={ai_stats['total_workspaces']}, "
            f"types={ai_stats['workspace_types']}, "
            f"initialized={self._initialized})"
        )
    
    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================
    
    async def __aenter__(self) -> "SessionManager":
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.shutdown()


# ============================================================================
# SESSION RECORD
# ============================================================================

@dataclass
class Session:
    """Sessione gestita dal SessionManager."""
    id: str
    context: SessionContext
    lifecycle: SessionLifecycle