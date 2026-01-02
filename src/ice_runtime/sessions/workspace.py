from __future__ import annotations

"""
Workspace - Namespace Isolato per Sessioni AI-Enhanced

Un Workspace rappresenta un ambiente di lavoro isolato con:
- Backend storage dedicati (SQL + Vector + AI-specific)
- Repository configurati (inclusi AI repositories)
- Metadata e configurazione AI
- Lifecycle hooks AI-aware
- Supporto per Knowledge Graph, RAG e multi-LLM
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from engine.storage.base import BackendConfig, BackendFactory, StorageBackend, BackendType
from engine.storage.backends.vector.base import VectorBackend
from engine.storage.exceptions import WorkspaceError, WorkspaceNotFoundError
from engine.storage.repositories.rag_repository import RAGRepository
from engine.storage.repositories.knowledge_repository import KnowledgeRepositoryManager, create_knowledge_repository
from engine.embeddings.adapter import get_default_embedding_adapter

# LLM adapter unificato (per llm_main / llm_coder)
from engine.llm.adapter import UnifiedLLMAdapter
from engine.llm.types import LLMProvider
from ice_core.logging.bridge import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMS AI-ENHANCED
# ============================================================================

class WorkspaceState(str, Enum):
    """Stati di un workspace."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    ERROR = "error"


class AIWorkspaceType(str, Enum):
    """Tipi di workspace AI supportati."""
    GENERIC = "generic"
    CODE_ANALYSIS = "code_analysis"
    LOG_ANALYSIS = "log_analysis"
    MULTI_AGENT = "multi_agent"
    KNOWLEDGE_BASE = "knowledge_base"


# ============================================================================
# WORKSPACE METADATA AI-ENHANCED
# ============================================================================

@dataclass
class WorkspaceMetadata:
    """Metadata di un workspace AI-enhanced."""
    id: str
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    ai_config: Dict[str, Any] = field(default_factory=dict)
    root_dir: str = ""
    workspace_path: str = ""

    def update(self) -> None:
        """Aggiorna timestamp."""
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Serializza a dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": list(self.tags),
            "settings": dict(self.settings),
            "ai_config": dict(self.ai_config),
            "root_dir": self.root_dir,
            "workspace_path": self.workspace_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceMetadata":
        """Deserializza da dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []) or [],
            settings=data.get("settings", {}) or {},
            ai_config=data.get("ai_config", {}) or {},
            root_dir=data.get("root_dir", "") or "",
            workspace_path=data.get("workspace_path", "") or "",
        )


def _ensure_minimal_metadata_dict(
    metadata: dict[str, Any],
    *,
    workspace_id: str,
    workspace_name: str,
    update_timestamp: bool = False,
) -> dict[str, Any]:
    now = datetime.now().isoformat()
    normalized = dict(metadata)
    normalized.setdefault("id", workspace_id)
    normalized.setdefault("name", workspace_name)
    normalized.setdefault("type", AIWorkspaceType.GENERIC.value)
    normalized.setdefault("project_root", None)

    rag = normalized.get("rag")
    if not isinstance(rag, dict):
        rag = {}
    rag.setdefault("enabled", False)
    normalized["rag"] = rag

    kg = normalized.get("kg")
    if not isinstance(kg, dict):
        kg = {}
    kg.setdefault("enabled", False)
    normalized["kg"] = kg

    llm_main = normalized.get("llm_main")
    if not isinstance(llm_main, dict) or not llm_main:
        llm_main = {"provider": "mock", "model": "mock-main"}
    normalized["llm_main"] = llm_main

    llm_coder = normalized.get("llm_coder")
    if not isinstance(llm_coder, dict) or not llm_coder:
        llm_coder = {"provider": "mock", "model": "mock-coder"}
    normalized["llm_coder"] = llm_coder

    embeddings = normalized.get("embeddings")
    if not isinstance(embeddings, dict) or not embeddings:
        embeddings = {"provider": "mock", "model": "mock-embed", "dimension": 384}
    else:
        embeddings.setdefault("provider", "mock")
        embeddings.setdefault("model", "mock-embed")
        embeddings.setdefault("dimension", 384)
    normalized["embeddings"] = embeddings

    normalized.setdefault("created_at", now)
    if update_timestamp:
        normalized["updated_at"] = now
    else:
        normalized.setdefault("updated_at", now)

    return normalized


# ============================================================================
# AI WORKSPACE CONFIGURATION
# ============================================================================

@dataclass
class AIWorkspaceSettings:
    """
    Configurazione specifica per workspace AI.

    Oltre ai campi generici, include tre blocchi principali:
    - llm_config:       LLM "main" scelto dall'utente (pannello LLM)
    - coder_llm_config: LLM dedicato al codice (es. Qwen2.5 coder)
    - embedding_config: motore di embeddings (pannello Embeddings)
    """

    workspace_type: str = AIWorkspaceType.GENERIC.value
    enabled_features: List[str] = field(default_factory=list)
    knowledge_graph_enabled: bool = False
    rag_enabled: bool = False
    default_models: List[str] = field(default_factory=lambda: ["gpt-4"])
    agent_settings: Dict[str, Any] = field(default_factory=dict)

    # Runtime AI configs
    llm_config: Dict[str, Any] = field(default_factory=dict)
    coder_llm_config: Dict[str, Any] = field(default_factory=dict)
    embedding_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializza a dict."""
        return {
            "workspace_type": self.workspace_type,
            "enabled_features": list(self.enabled_features),
            "knowledge_graph_enabled": self.knowledge_graph_enabled,
            "rag_enabled": self.rag_enabled,
            "default_models": list(self.default_models),
            "agent_settings": dict(self.agent_settings),
            "llm_config": dict(self.llm_config),
            "coder_llm_config": dict(self.coder_llm_config),
            "embedding_config": dict(self.embedding_config),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIWorkspaceSettings":
        """Deserializza da dict."""
        return cls(
            workspace_type=data.get("workspace_type", AIWorkspaceType.GENERIC.value),
            enabled_features=data.get("enabled_features", []) or [],
            knowledge_graph_enabled=data.get("knowledge_graph_enabled", False),
            rag_enabled=data.get("rag_enabled", False),
            default_models=data.get("default_models", ["gpt-4"]) or ["gpt-4"],
            agent_settings=data.get("agent_settings", {}) or {},
            llm_config=data.get("llm_config", {}) or {},
            coder_llm_config=data.get("coder_llm_config", {}) or {},
            embedding_config=data.get("embedding_config", {}) or {},
        )


@dataclass
class WorkspaceAIConfig:
    """
    Runtime configuration for workspace AI components.
    """
    enable_rag: bool = False
    enable_kg: bool = False
    llm_main: Optional[dict[str, Any]] = None
    llm_coder: Optional[dict[str, Any]] = None
    embeddings: Optional[dict[str, Any]] = None
    vector_backend: Optional[dict[str, Any]] = None
    kg: Optional[dict[str, Any]] = None
    rag: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enable_rag": self.enable_rag,
            "enable_kg": self.enable_kg,
            "llm_main": self.llm_main or {},
            "llm_coder": self.llm_coder or {},
            "embeddings": self.embeddings or {},
            "vector_backend": self.vector_backend or {},
            "kg": self.kg or {},
            "rag": self.rag or {},
        }

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "WorkspaceAIConfig":
        data = data or {}
        rag = data.get("rag") or {}
        enable_rag = rag.get("enabled", data.get("enable_rag", False))
        rag = data.get("rag") or {}
        enable_rag = rag.get("enabled", data.get("enable_rag", False))
        return cls(
            enable_rag=enable_rag,
            enable_kg=data.get("enable_kg", False),
            llm_main=data.get("llm_main"),
            llm_coder=data.get("llm_coder"),
            embeddings=data.get("embeddings"),
            vector_backend=data.get("vector_backend"),
            rag=rag,
            kg=data.get("kg"),
        )


# ============================================================================
# LIFECYCLE HOOKS AI-AWARE
# ============================================================================

class WorkspaceLifecycleHook(Protocol):
    """Protocol per hooks del lifecycle AI-aware."""

    def on_initialize(self, workspace: "Workspace") -> None: ...
    def on_activate(self, workspace: "Workspace") -> None: ...
    def on_suspend(self, workspace: "Workspace") -> None: ...
    def on_close(self, workspace: "Workspace") -> None: ...
    def on_error(self, workspace: "Workspace", error: Exception) -> None: ...


# ============================================================================
# WORKSPACE AI-ENHANCED
# ============================================================================

class Workspace:
    """
    Workspace isolato per gestione sessioni AI-enhanced.

    Architecture AI-Enhanced:
        Workspace AI
        ├── Backends (SQL, Vector, LLM Cache, Knowledge Graph)
        ├── Repositories (Events, Embeddings, LLM, RAG, Knowledge)
        ├── AI Components (Agents, KG Manager, RAG Engine, LLM adapters)
        ├── Metadata & AI Config
        └── Lifecycle Hooks AI-aware
    """

    # ----------------------------------------------------------------------
    # COSTRUZIONE
    # ----------------------------------------------------------------------
    def __init__(
        self,
        id: str,
        name: str,
        base_path: Path | str,
        backends: dict[str, BackendConfig],
        metadata: Optional[WorkspaceMetadata] = None,
        lifecycle_hooks: Optional[list[WorkspaceLifecycleHook]] = None,
        ai_config: Optional[AIWorkspaceSettings] = None,
    ):
        self.id = id
        self.name = name
        self.base_path = Path(base_path).resolve()
        self._backend_configs: dict[str, BackendConfig] = dict(backends)

        # Metadata
        self.metadata: WorkspaceMetadata = metadata or WorkspaceMetadata(id=id, name=name)

        # Ricostruisci AI settings partendo da metadata se non forniti
        # Ricostruisci AI settings partendo da metadata se non forniti
        if ai_config is None:
            ai_cfg_dict = self.metadata.ai_config or self.metadata.settings.get("ai_config", {}) or {}
            if not ai_cfg_dict and "workspace_type" in self.metadata.settings:
                ai_cfg_dict["workspace_type"] = self.metadata.settings["workspace_type"]

            self.ai_config = (
                AIWorkspaceSettings.from_dict(ai_cfg_dict)
                if ai_cfg_dict
                else AIWorkspaceSettings()
            )
        else:
            self.ai_config = ai_config

        self.runtime_ai_config = self._build_runtime_ai_config()
        self.paths = self._build_workspace_paths()

        # ---------------------------------------------------------
        #  AUTO-ENABLE features for AI workspaces
        # ---------------------------------------------------------
        # Stato
        self._state: WorkspaceState = WorkspaceState.INITIALIZING

        # Backends concreti
        self._backends: dict[str, StorageBackend] = {}

        # Repository cache (LLM cache, log repos, ecc.)
        self._repositories: dict[str, Any] = {}

        # Componenti AI (LLM main, coder, embedder, RAG, KG, ecc.)
        self._ai_components: Dict[str, Any] = {}

        # Lifecycle hooks
        self._lifecycle_hooks: list[WorkspaceLifecycleHook] = list(lifecycle_hooks or [])

        # Metriche AI locali
        self._ai_metrics: Dict[str, Any] = {
            "initialization_time": None,
            "last_ai_operation": None,
            "feature_usage": {},
        }

        logger.debug(
            "Workspace AI creato: %s (%s), type=%s, base_path=%s",
            self.id,
            self.name,
            self.ai_config.workspace_type,
            self.base_path,
        )

    # ----------------------------------------------------------------------
    # COSTRUTTORI DI COMODO
    # ----------------------------------------------------------------------
    @classmethod
    def load_from_disk(
        cls,
        base_path: Path | str,
        backend_configs: dict[str, BackendConfig],
        lifecycle_hooks: Optional[list[WorkspaceLifecycleHook]] = None,
    ) -> "Workspace":
        """
        Costruttore per caricare un workspace esistente da disco.

        Usato tipicamente da SessionManager.load_workspace().
        """
        base_path = Path(base_path)
        metadata = cls.load_metadata(base_path)

        ai_cfg_dict = metadata.ai_config or metadata.settings.get("ai_config", {}) or {}
        if not ai_cfg_dict and "workspace_type" in metadata.settings:
            ai_cfg_dict["workspace_type"] = metadata.settings["workspace_type"]

        ai_settings = AIWorkspaceSettings.from_dict(ai_cfg_dict) if ai_cfg_dict else AIWorkspaceSettings()

        return cls(
            id=metadata.id,
            name=metadata.name,
            base_path=base_path,
            backends=backend_configs,
            metadata=metadata,
            lifecycle_hooks=lifecycle_hooks,
            ai_config=ai_settings,
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def state(self) -> WorkspaceState:
        """Stato corrente workspace."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Verifica se workspace è attivo."""
        return self._state == WorkspaceState.ACTIVE

    @property
    def is_closed(self) -> bool:
        """Verifica se workspace è chiuso."""
        return self._state == WorkspaceState.CLOSED

    @property
    def is_ai_workspace(self) -> bool:
        """Verifica se è un workspace AI-enhanced."""
        return self.ai_config.workspace_type != AIWorkspaceType.GENERIC.value

    @property
    def workspace_type(self) -> str:
        """Tipo workspace AI (stringa)."""
        return self.ai_config.workspace_type

    @property
    def ai_features_enabled(self) -> List[str]:
        """Lista feature AI abilitate."""
        return list(self.ai_config.enabled_features)

    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    async def initialize(self) -> None:
        """Inizializza workspace AI-enhanced."""
        if self._state != WorkspaceState.INITIALIZING:
            logger.warning("Workspace %s già inizializzato (state=%s)", self.id, self._state.value)
            return

        try:
            logger.info(
                "Inizializzazione workspace AI: %s (name=%s, type=%s)",
                self.id,
                self.name,
                self.workspace_type,
            )

            # Directory workspace
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Backends
            await self._initialize_backends()
            self._auto_initialize_vector_backend()

            # Componenti AI (RAG, KG, LLMs, Embeddings, ...)
            await self._initialize_ai_components()
            # ---------------------------------------------------------
            # FIX: registra il workspace nel relational backend
            # ---------------------------------------------------------
                        # ---------------------------------------------------------
            # FIX: registra il workspace nel relational backend
            #      → per ora solo SQLite, per evitare problemi di dialetto
            # ---------------------------------------------------------
            try:
                rb = self.get_backend("relational")
                backend_type = getattr(rb.config, "backend_type", None)

                if backend_type == BackendType.SQLITE:
                    rb.execute(
                        """
                        INSERT OR IGNORE INTO workspaces
                        (workspace_id, name, description, workspace_type, ai_config,
                         knowledge_graph_enabled, rag_enabled,
                         created_at, updated_at, last_accessed,
                         is_active, metadata, owner_user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.id,
                            self.name,
                            self.metadata.description or "",
                            self.workspace_type,
                            json.dumps(self.ai_config.to_dict()),
                            1 if self.ai_config.knowledge_graph_enabled else 0,
                            1 if self.ai_config.rag_enabled else 0,
                            datetime.now().isoformat(),
                            datetime.now().isoformat(),
                            None,
                            1,
                            json.dumps(self.metadata.to_dict()),
                            None,
                        ),
                    )
                    logger.debug("Workspace registrato nel relational DB (SQLite): %s", self.id)
                else:
                    # Per MySQL/Postgres, per ora non registriamo per evitare mismatch di sintassi
                    logger.debug(
                        "Workspace registry su tabella workspaces non ancora implementato per backend=%s",
                        backend_type,
                    )

            except Exception as e:
                logger.error("Errore inserimento workspace nella tabella workspaces: %s", e)
                # Non blocchiamo l'inizializzazione del workspace per un problema di registry


            # Metadata
            self._save_metadata()
            self._validate_workspace_artifacts()

            # Metriche
            self._ai_metrics["initialization_time"] = datetime.now().isoformat()

            # Stato → ACTIVE
            self._state = WorkspaceState.ACTIVE

            # Hooks
            self._trigger_hooks("on_initialize")
            self._trigger_hooks("on_activate")

            logger.info("Workspace AI %s inizializzato e attivo", self.id)

        except Exception as e:
            self._state = WorkspaceState.ERROR
            self._trigger_hooks("on_error", error=e)
            raise WorkspaceError(
                workspace_id=self.id,
                message=f"Errore inizializzazione AI workspace: {e}",
                cause=e,
            ) from e

    async def suspend(self) -> None:
        """Sospende workspace (disconnette backend)."""
        if self._state != WorkspaceState.ACTIVE:
            logger.warning("Workspace %s non attivo (state=%s)", self.id, self._state.value)
            return

        logger.info("Sospensione workspace AI: %s", self.id)

        for name, backend in self._backends.items():
            if hasattr(backend, "is_connected") and backend.is_connected():
                backend.disconnect()
                logger.debug("Backend disconnesso: %s", name)

        await self._suspend_ai_components()
        self._state = WorkspaceState.SUSPENDED
        self._trigger_hooks("on_suspend")

    async def resume(self) -> None:
        """Riprende workspace sospeso (riconnette backend)."""
        if self._state != WorkspaceState.SUSPENDED:
            logger.warning("Workspace %s non sospeso (state=%s)", self.id, self._state.value)
            return

        logger.info("Ripresa workspace AI: %s", self.id)

        for name, backend in self._backends.items():
            if not hasattr(backend, "is_connected") or not backend.is_connected():
                backend.connect()
                logger.debug("Backend riconnesso: %s", name)

        await self._resume_ai_components()
        self._state = WorkspaceState.ACTIVE
        self._trigger_hooks("on_activate")

    async def close(self) -> None:
        """Chiude workspace definitivamente."""
        if self._state == WorkspaceState.CLOSED:
            logger.warning("Workspace %s già chiuso", self.id)
            return

        logger.info("Chiusura workspace AI: %s", self.id)

        # Hooks pre-close
        self._trigger_hooks("on_close")

        # Componenti AI
        await self._close_ai_components()

        # Backends
        for name, backend in self._backends.items():
            try:
                if hasattr(backend, "is_connected") and backend.is_connected():
                    backend.disconnect()
                    logger.debug("Backend disconnesso: %s", name)
            except Exception as e:
                logger.warning("Errore disconnessione backend %s: %s", name, e)

        # Cache
        self._repositories.clear()
        self._backends.clear()
        self._ai_components.clear()

        # Metadata finale
        self.metadata.update()
        self._save_metadata()

        self._state = WorkspaceState.CLOSED
        logger.info("Workspace AI %s chiuso", self.id)

    # =========================================================================
    # AI COMPONENTS
    # =========================================================================

    async def _initialize_ai_components(self) -> None:
        """Inizializza componenti AI specifici per tipo workspace."""
        logger.debug("Inizializzazione componenti AI per workspace: %s", self.id)
        cfg = self.runtime_ai_config

        # --------------------------------------------------------
        # 1. KNOWLEDGE GRAPH (opzionale)
        # --------------------------------------------------------
        if cfg.enable_kg:
            await self._initialize_knowledge_graph()
        else:
            logger.debug(">>> KG DISABLED (workspace runtime config)")

        # --------------------------------------------------------
        # 2. LLM MAIN / CODER (indipendenti dal RAG)
        # --------------------------------------------------------
        await self._initialize_llm_main(cfg.llm_main)
        await self._initialize_llm_coder(cfg.llm_coder)

        # --------------------------------------------------------
        # 3. EMBEDDINGS (MUST COME BEFORE RAG)
        # --------------------------------------------------------
        # RAG dipende da embeddings: vanno inizializzati prima.
        await self._initialize_embedding_engine(cfg.embeddings)

        # --------------------------------------------------------
        # 4. RAG ENGINE (usa embeddings + vector)
        # --------------------------------------------------------
        if cfg.enable_rag:
            logger.debug(">>> RAG ENABLED → chiamo _initialize_rag_engine()")
            await self._initialize_rag_engine()
        else:
            logger.debug(">>> RAG DISABLED (workspace runtime config)")

        logger.debug("Componenti AI inizializzati per workspace: %s", self.id)

    # ---------------------- Knowledge Graph ---------------------- #

    async def _initialize_knowledge_graph(self) -> None:
        """Inizializza Knowledge Graph."""
        try:
            backend = self.get_backend("relational")
            kg_repo = KnowledgeRepositoryManager(backend)
            kg_repo.workspace_id = self.id

            self._ai_components["knowledge_graph"] = kg_repo
            logger.debug("Knowledge Graph inizializzato per workspace: %s", self.id)

        except Exception as e:
            logger.warning(
                "Errore inizializzazione Knowledge Graph (%s): %s",
                self.id,
                e,
            )

    # ------------------------- RAG Engine ------------------------ #
    async def _initialize_rag_engine(self) -> None:
        """Inizializza RAG engine."""
        logger.debug(">>> ENTER _initialize_rag_engine() per workspace %s", self.id)
        rag_cfg = self.runtime_ai_config.rag or {}
        if not rag_cfg.get("enabled"):
            logger.debug(">>> RAG DISABLED (workspace runtime config)")
            self._ai_components.pop("rag", None)
            return

        self._ai_components.pop("rag", None)

        try:
            from engine.storage.repositories.rag_repository import RAGRepository

            # VECTOR
            try:
                vb = self.get_backend("vector")
            except Exception:
                logger.debug("Vector backend assente, inizializzo fallback…")
                self._auto_initialize_vector_backend()
                vb = self.get_backend("vector")

            # RELATIONAL
            rb = self.get_backend("relational")

            emb_cfg = rag_cfg.get("embeddings") or {
                "provider": rag_cfg.get("provider", "mock"),
                "model": rag_cfg.get("model"),
            }
            embedding_adapter = get_default_embedding_adapter(emb_cfg)

            logger.debug(
                ">>> Backends OK: vector=%s relational=%s embeddings=%s",
                type(vb).__name__, type(rb).__name__, embedding_adapter.__class__.__name__
            )

            rag_repo = RAGRepository(
                relational_backend=rb,
                vector_backend=vb,
                embeddings=embedding_adapter,
                workspace_id=self.id,
            )

            self._ai_components["rag"] = rag_repo
            logger.debug(">>> RAG ENGINE CREATED for workspace %s", self.id)

        except Exception as e:
            logger.error(">>> ERRORE NEL RAG ENGINE: %s", e)


    # --------------------------- LLMs ---------------------------- #

    def _build_llm_adapter_from_config(self, cfg: Dict[str, Any]) -> Optional[UnifiedLLMAdapter]:
        """
        Crea un UnifiedLLMAdapter a partire da una config di tipo:

        {
            "provider": "openai" | "openai_compat" | "ollama" | "llama_cpp" | "mock",
            "model": "gpt-4.1-mini",
            "base_url": "...",
            "api_key": "...",
            "organization": "...",
            "mode": "cloud" | "local",
            "default_params": {...},
            "local": { "binary_path": "...", "model_path": "...", "host": "...", "port": 8000, ... }
        }

        Nota: qui usiamo solo i campi necessari all'adapter; il resto (local, mode, default_params)
        rimane a disposizione di altri componenti (es. process manager).
        """
        if not cfg:
            return None

        provider_str = (cfg.get("provider") or "mock").lower()
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            logger.warning(
                "Workspace %s: provider LLM sconosciuto '%s', fallback a MOCK",
                self.id,
                provider_str,
            )
            provider = LLMProvider.MOCK

        model = cfg.get("model")
        adapter_cfg = {
            "model": model,
            "base_url": cfg.get("base_url"),
            "api_key": cfg.get("api_key"),
            "organization": cfg.get("organization"),
        }

        try:
            adapter = UnifiedLLMAdapter(
                provider=provider,
                default_model=model,
                config=adapter_cfg,
            )
            return adapter
        except Exception as e:
            logger.error(
                "Workspace %s: errore creazione UnifiedLLMAdapter (provider=%s, model=%s): %s",
                self.id,
                provider.value,
                model,
                e,
            )
            return None

    async def _initialize_llm_main(self, override: Optional[dict[str, Any]] = None) -> None:
        """
        Inizializza l'LLM principale del workspace (llm_main).

        La config è letta da:
        - self.ai_config.llm_config
        - in fallback, self.metadata.ai_config.get("llm") o agent_settings["llm"]
        - se tutto è vuoto, default sicuro: provider=mock
        """
        cfg = dict(override or self.runtime_ai_config.llm_main or {})

        if not cfg:
            # fallback compatibilità da metadata / agent_settings
            cfg = (
                self.metadata.ai_config.get("llm", {})
                or self.ai_config.agent_settings.get("llm", {})
                or {}
            )

        if not cfg:
            # default ENV-aware ma con fallback sicuro a mock
            provider_env = os.getenv("ICE_STUDIO_LLM_PROVIDER", "mock")
            model_env = os.getenv("ICE_STUDIO_LLM_MODEL", "mock-main")

            cfg = {
                "provider": provider_env,
                "model": model_env,
            }
            if provider_env == "openai":
                cfg["base_url"] = os.getenv("OPENAI_BASE_URL")
                cfg["api_key"] = os.getenv("OPENAI_API_KEY")
                cfg["organization"] = os.getenv("OPENAI_ORG")
        else:
            # Override con ENV se presenti (priorità a configurazione runtime)
            provider_env = os.getenv("ICE_STUDIO_LLM_PROVIDER")
            model_env = os.getenv("ICE_STUDIO_LLM_MODEL")
            base_url_env = os.getenv("OPENAI_BASE_URL")
            api_key_env = os.getenv("OPENAI_API_KEY")
            org_env = os.getenv("OPENAI_ORG")
            if provider_env:
                cfg["provider"] = provider_env
            if model_env:
                cfg["model"] = model_env
            if base_url_env:
                cfg["base_url"] = base_url_env
            if api_key_env:
                cfg["api_key"] = api_key_env
            if org_env:
                cfg["organization"] = org_env

        adapter = self._build_llm_adapter_from_config(cfg)

        # Se fallisce (es. manca OPENAI_API_KEY), fallback HARD a mock
        if adapter is None and (cfg.get("provider") or "").lower() != "mock":
            logger.warning(
                "Workspace %s: fallback llm_main → provider=mock",
                self.id,
            )
            adapter = self._build_llm_adapter_from_config(
                {"provider": "mock", "model": "mock-main"}
            )

        if adapter is not None:
            self._ai_components["llm_main"] = adapter
            logger.debug(
                "Workspace %s: llm_main inizializzato (provider=%s)",
                self.id,
                (cfg.get("provider") or "mock"),
            )
        else:
            logger.debug(
                "Workspace %s: nessun llm_main configurato (neanche mock)",
                self.id,
            )

    async def _initialize_llm_coder(self, override: Optional[dict[str, Any]] = None) -> None:
        """
        Inizializza l'LLM dedicato al codice (llm_coder).

        Se non è configurato esplicitamente, viene creato un adapter
        di default (mock-coder), così i test non dipendono da Ollama
        o altri servizi esterni.
        """
        cfg = dict(override or self.runtime_ai_config.llm_coder or {})

        if not cfg:
            # fallback compatibilità da metadata / agent_settings
            cfg = (
                self.metadata.ai_config.get("llm_coder", {})
                or self.ai_config.agent_settings.get("llm_coder", {})
                or {}
            )

        if not cfg:
            # default sicuro per test: mock-coder
            cfg = {
                "provider": "mock",
                "model": "mock-coder",
            }
        else:
            # Consenti override da ENV se presenti (usa stessi env dell'LLM principale)
            provider_env = os.getenv("ICE_STUDIO_LLM_PROVIDER")
            model_env = os.getenv("ICE_STUDIO_LLM_MODEL")
            base_url_env = os.getenv("OPENAI_BASE_URL")
            api_key_env = os.getenv("OPENAI_API_KEY")
            org_env = os.getenv("OPENAI_ORG")
            if provider_env:
                cfg["provider"] = provider_env
            if model_env:
                cfg["model"] = model_env
            if base_url_env:
                cfg["base_url"] = base_url_env
            if api_key_env:
                cfg["api_key"] = api_key_env
            if org_env:
                cfg["organization"] = org_env

        adapter = self._build_llm_adapter_from_config(cfg)

        # Se fallisce qualcosa, fallback a mock-coder
        if adapter is None and (cfg.get("provider") or "").lower() != "mock":
            logger.warning(
                "Workspace %s: fallback llm_coder → provider=mock",
                self.id,
            )
            adapter = self._build_llm_adapter_from_config(
                {"provider": "mock", "model": "mock-coder"}
            )

        if adapter is not None:
            self._ai_components["llm_coder"] = adapter
            logger.debug(
                "Workspace %s: llm_coder inizializzato (provider=%s)",
                self.id,
                (cfg.get("provider") or "mock"),
            )
        else:
            logger.debug(
                "Workspace %s: nessun llm_coder configurato (neanche mock)",
                self.id,
            )
    # ----------------------- Embedding Engine -------------------- #

    async def _initialize_embedding_engine(self, override: Optional[dict[str, Any]] = None) -> None:
        """
        Inizializza il motore di embeddings.
        Se nessuna config è presente, crea un mock embedding engine
        così il RAG può funzionare nei test e in locale.
        """
        cfg = dict(override or self.runtime_ai_config.embeddings or {})

        # Fallback compatibilità
        if not cfg:
            cfg = (
                self.metadata.ai_config.get("embeddings", {})
                or self.ai_config.agent_settings.get("embeddings", {})
                or {}
            )

        # Se ancora vuoto → mock embeddings
        if not cfg:
            cfg = {
                "provider": "mock",
                "model": "mock-embed",
                "dimension": 384,
            }
        else:
            provider_env = os.getenv("ICE_STUDIO_EMBEDDINGS_PROVIDER")
            model_env = os.getenv("ICE_STUDIO_EMBEDDINGS_MODEL")
            compat_url_env = os.getenv("ICE_STUDIO_EMBEDDINGS_COMPAT_URL")
            compat_key_env = os.getenv("ICE_STUDIO_EMBEDDINGS_COMPAT_API_KEY")
            if provider_env:
                cfg["provider"] = provider_env
            if model_env:
                cfg["model"] = model_env
            if compat_url_env:
                cfg["base_url"] = compat_url_env
            if compat_key_env:
                cfg["api_key"] = compat_key_env

        self._ai_components["embedding_engine"] = cfg
        logger.debug(
            "Workspace %s: embedding_engine inizializzato (cfg=%s)",
            self.id,
            cfg,
        )

    # ------------------------ Component lifecycle ---------------- #

    async def _suspend_ai_components(self) -> None:
        """Sospende componenti AI (se supportano .suspend)."""
        for name, component in self._ai_components.items():
            if hasattr(component, "suspend"):
                try:
                    await component.suspend()
                    logger.debug("Componente AI sospeso: %s", name)
                except Exception as e:
                    logger.warning("Errore sospensione componente %s: %s", name, e)

    async def _resume_ai_components(self) -> None:
        """Riprende componenti AI (se supportano .resume)."""
        for name, component in self._ai_components.items():
            if hasattr(component, "resume"):
                try:
                    await component.resume()
                    logger.debug("Componente AI ripreso: %s", name)
                except Exception as e:
                    logger.warning("Errore ripresa componente %s: %s", name, e)

    async def _close_ai_components(self) -> None:
        """Chiude componenti AI (se supportano .close)."""
        for name, component in self._ai_components.items():
            if hasattr(component, "close"):
                try:
                    await component.close()
                    logger.debug("Componente AI chiuso: %s", name)
                except Exception as e:
                    logger.warning("Errore chiusura componente %s: %s", name, e)

    # =========================================================================
    # BACKEND MANAGEMENT
    # =========================================================================

    async def _initialize_backends(self) -> None:
        """Inizializza tutti i backend configurati."""
        for name, config in self._backend_configs.items():
            logger.debug("Inizializzazione backend AI: %s", name)

            local_cfg = config

            # SOLO i backend file-based vengono "reindirizzati" sotto il workspace
            file_based_types = {
                BackendType.SQLITE,
                BackendType.DUCKDB,
                BackendType.CHROMADB,
            }

            backend_path = self.paths.get(name)

            if (
                backend_path is not None
                and config.backend_type in file_based_types
                and config.connection_string != ":memory:"
            ):
                self._prepare_backend_path(
                    backend_path,
                    config.backend_type,
                    is_dir=config.backend_type in {BackendType.CHROMADB, BackendType.FAISS},
                )

                local_cfg = BackendConfig(
                    backend_type=config.backend_type,
                    connection_string=str(backend_path),
                    mode=config.mode,
                    options=dict(getattr(config, "options", {})),
                )

            backend = BackendFactory.create(local_cfg)

            backend.connect()
            if hasattr(backend, "initialize_schema"):
                backend.initialize_schema()

            self._backends[name] = backend
            logger.debug("Backend AI %s inizializzato", name)


    def get_backend(self, name: str) -> StorageBackend:
        """Ottiene un backend per nome."""
        if name not in self._backends:
            raise WorkspaceError(
                workspace_id=self.id,
                message=f"Backend '{name}' non trovato",
                details={"available": list(self._backends.keys())},
            )
        return self._backends[name]

    def get_relational_backend(self) -> StorageBackend:
        """Shortcut per il backend relazionale."""
        return self.get_backend("relational")

    def get_vector_backend(self) -> VectorBackend | None:
        """Restituisce il backend vettoriale se presente."""
        try:
            return self.get_backend("vector")
        except WorkspaceError:
            return None

    def get_rag_engine(self) -> Optional["RAGRepository"]:
        """Restituisce il RAGRepository se attivo."""
        return self._ai_components.get("rag")

    def get_knowledge_manager(self) -> Optional[KnowledgeRepositoryManager]:
        """Restituisce il KnowledgeRepositoryManager se abilitato."""
        return self._ai_components.get("knowledge_graph")

    def list_backends(self) -> list[str]:
        """Lista nomi backend disponibili."""
        return list(self._backends.keys())

    async def rag_index(self, texts: list[str], metadata: Optional[dict[str, Any]] = None) -> list[str]:
        """
        Indicizza un set di testi usando il RAGRepository attivo.
        """
        rag_engine = self.get_rag_engine()
        if rag_engine is None:
            raise RuntimeError("RAG non abilitato per questo workspace")

        metadata = metadata or {}
        doc_ids = []
        for text in texts:
            doc_id = await rag_engine.index_text(text, metadata=metadata)
            doc_ids.append(doc_id)
        return doc_ids

    async def rag_query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Esegue una query semantica sul knowledge store del workspace.
        """
        rag_engine = self.get_rag_engine()
        if rag_engine is None:
            raise RuntimeError("RAG non abilitato per questo workspace")

        return rag_engine.search(query, top_k=top_k)

    # =========================================================================
    # REPOSITORY FACTORY
    # =========================================================================

    def get_repository(self, repo_type: str) -> Any:
        """
        Ottiene un repository (con caching) AI-enhanced.

        repo_type comuni:
        - "events", "sources", "patterns"
        - "llm", "rag", "knowledge"
        """
        if repo_type in self._repositories:
            return self._repositories[repo_type]

        repo = self._create_ai_repository(repo_type)
        self._repositories[repo_type] = repo
        return repo

    def get_llm_repository(self) -> Any:
        """Ottiene repository LLM (convenience)."""
        return self.get_repository("llm")

    def get_rag_repository(self) -> Any:
        """Ottiene repository RAG (convenience)."""
        return self.get_repository("rag")

    def get_knowledge_repository(self) -> Any:
        """Ottiene repository Knowledge (convenience)."""
        return self.get_repository("knowledge")

    def _create_ai_repository(self, repo_type: str) -> Any:
        """Factory per creare repository AI-enhanced."""
        from engine.storage.repositories.log_repository import (
            LogEventRepository,
            LogSourceRepository,
            LogPatternRepository,
        )

        repo_map: dict[str, Any] = {
            "events": lambda: LogEventRepository(self.get_backend("relational")),
            "sources": lambda: LogSourceRepository(self.get_backend("relational")),
            "patterns": lambda: LogPatternRepository(self.get_backend("relational")),
        } 
        
        # --- REPOSITORY CODE ANALYSIS ---
        from engine.storage.repositories.code_repository import CodeRepository
        repo_map["code"] = lambda: CodeRepository(self.get_backend("relational"))


        if self.is_ai_workspace:
            # LLM cache repository
            if "llm_cache" in self._backends:
                from engine.storage.repositories.llm_repository import LLMRepository

                repo_map["llm"] = lambda: LLMRepository(self.get_backend("llm_cache"))

            # RAG engine come repository
            if "rag_engine" in self._ai_components:
                repo_map["rag"] = lambda: self._ai_components["rag_engine"]

            # Knowledge Graph come repository
            if "knowledge_graph" in self._ai_components:
                repo_map["knowledge"] = lambda: self._ai_components["knowledge_graph"]

        if repo_type not in repo_map:
            raise WorkspaceError(
                workspace_id=self.id,
                message=f"Repository type '{repo_type}' non supportato",
                details={
                    "available": list(repo_map.keys()),
                    "ai_workspace": self.is_ai_workspace,
                },
            )

        return repo_map[repo_type]()

    # =========================================================================
    # AI COMPONENT ACCESSORS (per SessionContext)
    # =========================================================================

    def get_llm_main(self) -> Optional[UnifiedLLMAdapter]:
        """LLM principale del workspace (reasoning, planner, analyzer, ecc.)."""
        comp = self._ai_components.get("llm_main")
        return comp if isinstance(comp, UnifiedLLMAdapter) else None

    def get_llm_coder(self) -> Optional[UnifiedLLMAdapter]:
        """LLM dedicato al codice (codegen, refactor)."""
        comp = self._ai_components.get("llm_coder")
        return comp if isinstance(comp, UnifiedLLMAdapter) else None

    def get_embedding_engine(self) -> Any:
        """
        Motore di embeddings o la sua config.

        Può essere:
        - dict di configurazione (cloud / local)
        - oggetto client concreto, se in futuro lo crei qui.
        """
        return self._ai_components.get("embedding_engine")

    # =========================================================================
    # METADATA MANAGEMENT
    # =========================================================================

    def _metadata_path(self) -> Path:
        return self.paths["metadata"]

    def _build_workspace_paths(self) -> dict[str, Path]:
        """
        Costruisce i layout fissati dei file di backend per il workspace.
        """
        relational_name = self._relational_filename()
        return {
            "relational": self._workspace_backend_path(relational_name),
            "llm_cache": self._workspace_backend_path("llm_cache.db"),
            "vector": self._workspace_backend_path("vector_index"),
            "metadata": self.base_path / "workspace.json",
            "rag": self._workspace_backend_path("rag"),
            "kg": self._workspace_backend_path("kg"),
        }

    def _relational_filename(self) -> str:
        """
        Determina il nome file del DB relazionale in base al tipo di workspace.
        """
        workspace_type = self.ai_config.workspace_type or AIWorkspaceType.GENERIC.value
        mapping = {
            AIWorkspaceType.MULTI_AGENT.value: "agent_workflows.db",
            AIWorkspaceType.KNOWLEDGE_BASE.value: "knowledge_graph.db",
        }
        if workspace_type == AIWorkspaceType.GENERIC.value:
            return "data.db"
        return mapping.get(workspace_type, f"{workspace_type}.db")

    def _build_runtime_ai_config(self) -> WorkspaceAIConfig:
        """
        Costruisce la config runtime unificata per llm, rag, kg, embeddings ecc.
        """
        runtime_dict: dict[str, Any] = {}
        runtime_dict.update(self.metadata.ai_config.get("workspace_runtime_config", {}))
        runtime_dict.update(self.metadata.settings.get("workspace_runtime_config", {}))

        defaults = {
            "enable_rag": self.ai_config.rag_enabled,
            "enable_kg": self.ai_config.knowledge_graph_enabled,
            "llm_main": self.ai_config.llm_config or None,
            "llm_coder": self.ai_config.coder_llm_config or None,
            "embeddings": self.ai_config.embedding_config or None,
            "vector_backend": {"dimension": 384, "distance_metric": "cosine"},
        }

        for key, value in defaults.items():
            runtime_dict.setdefault(key, value)

        rag_defaults = runtime_dict.get("rag") or {}
        rag_enabled_fallback = runtime_dict.get("enable_rag", self.ai_config.rag_enabled)
        rag_defaults.setdefault("enabled", rag_enabled_fallback)
        runtime_dict["rag"] = rag_defaults
        runtime_dict["enable_rag"] = rag_defaults["enabled"]

        kg_defaults = runtime_dict.get("kg") or {}
        kg_enabled_fallback = runtime_dict.get("enable_kg", self.ai_config.knowledge_graph_enabled)
        kg_defaults.setdefault("enabled", kg_enabled_fallback)
        runtime_dict["kg"] = kg_defaults
        runtime_dict["enable_kg"] = kg_defaults["enabled"]

        return WorkspaceAIConfig.from_dict(runtime_dict)

    def _save_metadata(self) -> None:
        """Salva metadata su disco (inclusa AI config)."""
        metadata_path = self._metadata_path()

        # Sincronizza ai_config nel metadata
        self.metadata.ai_config = self.ai_config.to_dict()
        self.metadata.settings.setdefault("workspace_type", self.ai_config.workspace_type)
        self.metadata.settings.setdefault("ai_config", self.metadata.ai_config)
        self.metadata.root_dir = self.metadata.settings.get("root_dir", self.metadata.root_dir)
        self.metadata.workspace_path = self.metadata.settings.get("workspace_path", str(self.base_path))
        self.metadata.ai_config.setdefault("workspace_runtime_config", {})
        self.metadata.ai_config["workspace_runtime_config"] = self.runtime_ai_config.to_dict()
        self.metadata.update()

        normalized = _ensure_minimal_metadata_dict(
            self.metadata.to_dict(),
            workspace_id=self.id,
            workspace_name=self.name,
            update_timestamp=True,
        )

        self.metadata = WorkspaceMetadata.from_dict(normalized)

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2)

        logger.debug("Metadata AI salvati: %s", metadata_path)

    def save_metadata(self) -> None:
        """Public wrapper to persist workspace metadata."""
        self._save_metadata()

    def _validate_workspace_artifacts(self) -> None:
        """
        Verifica che i file fondamentali del workspace esistano, creando quelli mancanti.
        """
        metadata_path = self._metadata_path()
        if not metadata_path.exists():
            self._save_metadata()

        for backend_name in ("relational", "llm_cache"):
            config = self._backend_configs.get(backend_name)
            backend_type = config.backend_type if config else BackendType.SQLITE
            path = self.paths[backend_name]
            self._prepare_backend_path(path, backend_type)

        self._prepare_backend_path(self.paths["vector"], BackendType.FAISS, is_dir=True)
        self._prepare_backend_path(self.paths["rag"], BackendType.FAISS, is_dir=True)
        self._prepare_backend_path(self.paths["kg"], BackendType.FAISS, is_dir=True)

    async def configure_runtime(self, updates: dict[str, Any]) -> None:
        """
        Aggiorna la runtime config persistente del workspace e (ri)attiva i componenti interessati.
        """
        runtime_cfg = self.metadata.ai_config.setdefault("workspace_runtime_config", {})

        for key, value in updates.items():
            if key == "rag" and isinstance(value, dict):
                runtime_cfg.setdefault("rag", {}).update(value)
            else:
                runtime_cfg[key] = value

        self.runtime_ai_config = self._build_runtime_ai_config()
        self._save_metadata()

        if "rag" in updates or "enable_rag" in updates:
            if not self.runtime_ai_config.enable_rag:
                self._ai_components.pop("rag", None)
                logger.debug("Workspace %s: RAG engine disattivato via configurazione", self.id)
            else:
                await self._initialize_rag_engine()
        if "kg" in updates or "enable_kg" in updates:
            if not self.runtime_ai_config.enable_kg:
                self._ai_components.pop("knowledge_graph", None)
                logger.debug("Workspace %s: Knowledge Graph disattivato via configurazione", self.id)
            else:
                await self._initialize_knowledge_graph()

    def update_metadata(self, data: dict[str, Any]) -> None:
        """Merge dei metadata (settings/ai_config)."""
        self.metadata.settings.update(data or {})
        self.metadata.update()
        self._save_metadata()

    async def reload_ai_components(self, updated_only: bool = True):
        """
        Ricostruisce i componenti AI partendo dai metadata.
        """
        if not updated_only:
            return await self._initialize_ai_components()

        meta = self.metadata

        if getattr(meta, "llm_main", None):
            try:
                if hasattr(self, "_initialize_llm_main"):
                    await self._initialize_llm_main(meta.llm_main)
            except Exception:
                pass

        if getattr(meta, "llm_coder", None):
            try:
                if hasattr(self, "_initialize_llm_coder"):
                    await self._initialize_llm_coder(meta.llm_coder)
            except Exception:
                pass

        if getattr(meta, "embeddings", None):
            try:
                if hasattr(self, "_initialize_embedding_engine"):
                    await self._initialize_embedding_engine(meta.embeddings)
            except Exception:
                pass

        rag_cfg = meta.settings.get("rag") or {}
        if isinstance(rag_cfg, dict) and rag_cfg.get("enabled"):
            try:
                if hasattr(self, "_initialize_rag_engine"):
                    await self._initialize_rag_engine()
            except Exception:
                pass

        kg_cfg = meta.settings.get("kg") or {}
        if isinstance(kg_cfg, dict) and kg_cfg.get("enabled"):
            try:
                if hasattr(self, "_initialize_knowledge_graph"):
                    await self._initialize_knowledge_graph()
            except Exception:
                pass

        return True

    def apply_configuration(self, cfg: dict[str, Any]) -> None:
        """Aggiorna i metadata con i valori forniti, senza salvare né ricaricare."""
        meta = self.metadata

        if "name" in cfg:
            meta.name = cfg["name"]
        if "project_root" in cfg:
            meta.project_root = cfg["project_root"]
        if "type" in cfg:
            meta.settings["workspace_type"] = cfg["type"]

        rag_cfg = cfg.get("rag")
        if isinstance(rag_cfg, dict):
            enabled = rag_cfg.get("enabled")
            if enabled is not None:
                meta.settings.setdefault("rag", {})["enabled"] = bool(enabled)
            meta.settings.setdefault("rag", {}).update(rag_cfg)

        kg_cfg = cfg.get("kg")
        if isinstance(kg_cfg, dict):
            enabled = kg_cfg.get("enabled")
            if enabled is not None:
                meta.settings.setdefault("kg", {})["enabled"] = bool(enabled)
            meta.settings.setdefault("kg", {}).update(kg_cfg)

        llm_main_cfg = cfg.get("llm_main")
        if isinstance(llm_main_cfg, dict):
            meta.llm_main.update(llm_main_cfg)

        llm_coder_cfg = cfg.get("llm_coder")
        if isinstance(llm_coder_cfg, dict):
            meta.llm_coder.update(llm_coder_cfg)

        emb_cfg = cfg.get("embeddings")
        if isinstance(emb_cfg, dict):
            meta.embeddings.update(emb_cfg)

        meta.updated_at = datetime.utcnow()

    @classmethod
    def load_metadata(cls, base_path: Path | str) -> WorkspaceMetadata:
        """Carica metadata da workspace esistente."""
        base_path = Path(base_path)
        metadata_path = base_path / "workspace.json"

        if not metadata_path.exists():
            raise WorkspaceNotFoundError(
                workspace_id=str(base_path),
                reason="Metadata file non trovato",
            )

        with metadata_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        workspace_id = data.get("id") or base_path.name
        workspace_name = data.get("name") or workspace_id
        normalized = _ensure_minimal_metadata_dict(
            data,
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            update_timestamp=False,
        )
        return WorkspaceMetadata.from_dict(normalized)

    def delete(self) -> None:
        """Rimuove la directory del workspace e pulisce lo stato."""
        if self.base_path.exists():
            shutil.rmtree(self.base_path, ignore_errors=True)

        self._repositories.clear()
        self._backends.clear()
        self._ai_components.clear()
        self._state = WorkspaceState.CLOSED

        logger.info("Workspace %s eliminato (directory rimossa)", self.id)

    def validate(self) -> None:
        """Valida che il workspace sia utilizzabile."""
        if not self.base_path.exists() or not self.base_path.is_dir():
            raise WorkspaceError(
                workspace_id=self.id,
                message="Workspace path non esiste",
                details={"path": str(self.base_path)},
            )
        if not os.access(self.base_path, os.W_OK):
            raise WorkspaceError(
                workspace_id=self.id,
                message="Workspace path non scrivibile",
                details={"path": str(self.base_path)},
            )

    # =========================================================================
    # LIFECYCLE HOOKS
    # =========================================================================

    def add_lifecycle_hook(self, hook: WorkspaceLifecycleHook) -> None:
        """Aggiunge un lifecycle hook."""
        self._lifecycle_hooks.append(hook)

    def _trigger_hooks(self, method_name: str, **kwargs) -> None:
        """Trigger hooks per un evento."""
        for hook in list(self._lifecycle_hooks):
            try:
                method = getattr(hook, method_name, None)
                if method:
                    method(self, **kwargs)
            except Exception as e:
                logger.error("Errore in hook %s per workspace %s: %s", method_name, self.id, e)

    # =========================================================================
    # AI METRICS
    # =========================================================================

    def record_ai_operation(self, feature: str) -> None:
        """Registra l'uso di una feature AI (in-mem, usato da manager/registry)."""
        self._ai_metrics["last_ai_operation"] = datetime.now().isoformat()
        usage = self._ai_metrics.setdefault("feature_usage", {})
        usage[feature] = usage.get(feature, 0) + 1

    def get_ai_metrics(self) -> Dict[str, Any]:
        """Ritorna una copia delle metriche AI locali."""
        return dict(self._ai_metrics)

    # =========================================================================
    # INFO
    # =========================================================================

    def get_info(self) -> dict[str, Any]:
        """
        Ottiene informazioni sul workspace AI-enhanced.

        Nota: il formato 'backends' è compatibile con LoggingHook
        (dict name -> {type, stats?}).
        """
        backends_info: dict[str, Any] = {}
        for name, backend in self._backends.items():
            info: dict[str, Any] = {"type": backend.__class__.__name__}
            if hasattr(backend, "get_stats"):
                try:
                    info["stats"] = backend.get_stats()
                except Exception:
                    info["stats"] = {}
            backends_info[name] = info

        return {
            "id": self.id,
            "name": self.name,
            "state": self._state.value,
            "base_path": str(self.base_path),
            "metadata": self.metadata.to_dict(),
            "ai_config": self.ai_config.to_dict(),
            "backends": backends_info,
            "repositories": list(self._repositories.keys()),
            "ai_components": list(self._ai_components.keys()),
            "ai_metrics": self.get_ai_metrics(),
        }
    # =========================================================================
    # VECTOR BACKEND INITIALIZATION (AUTO-DETECT)
    # =========================================================================

    def _auto_initialize_vector_backend(self) -> None:
        """
        Garantisce che un backend vectore esista SEMPRE.
        Nei test pytest, FakeVectorBackend sostituisce FAISS/Chroma.
        """
        # Già presente → OK
        if "vector" in self._backends:
            return

        logger.debug("Workspace %s: inizializzazione backend vector AUTO", self.id)

        # In assenza di config esplicita → fallback automatico
        vector_path = self.paths["vector"]
        self._prepare_backend_path(vector_path, BackendType.FAISS, is_dir=True)
        index_path = vector_path / "index.faiss"

        vector_opts = self.runtime_ai_config.vector_backend or {}
        dimension = vector_opts.get("dimension") or vector_opts.get("dim") or 384
        distance_metric = vector_opts.get("distance_metric") or vector_opts.get("metric") or "cosine"

        cfg = BackendConfig(
            backend_type=BackendType.FAISS,     # non importa cosa usi, pytest lo sostituisce con FakeVectorBackend
            connection_string=str(index_path),
            options={
                "dimension": dimension,
                "distance_metric": distance_metric,
            }
        )

        backend = BackendFactory.create(cfg)
        backend.connect()

        self._backends["vector"] = backend
        logger.debug("Workspace %s: backend vector inizializzato automaticamente (FAISS/FakeVector)", self.id)

    def _workspace_backend_path(self, raw_path: str | Path) -> Path:
        """
        Normalizza un percorso di backend verso la directory workspace.
        Rimuove componenti assoluti o traversali e restituisce workspace.base_path / safe_path.
        """
        candidate = Path(raw_path)
        if candidate.is_absolute():
            safe = Path(candidate.name or candidate.stem or "backend")
        else:
            safe_parts = [part for part in candidate.parts if part not in ("", ".", "..")]
            if not safe_parts:
                safe_parts = [candidate.name or candidate.stem or "backend"]
            safe = Path(*safe_parts)
        return self.base_path.joinpath(safe)

    def _prepare_backend_path(
        self,
        path: Path,
        backend_type: BackendType,
        is_dir: bool = False,
    ) -> None:
        """
        Garantisce che il file o la directory del backend esistano sotto il workspace.
        """
        if is_dir or backend_type in {BackendType.CHROMADB, BackendType.FAISS}:
            path.mkdir(parents=True, exist_ok=True)
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    # =========================================================================
    # CONTEXT MANAGER
    # =========================================================================

    async def __aenter__(self) -> "Workspace":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        return (
            f"WorkspaceAI(id={self.id}, name={self.name}, "
            f"state={self._state.value}, type={self.workspace_type})"
        )
        
    
    
