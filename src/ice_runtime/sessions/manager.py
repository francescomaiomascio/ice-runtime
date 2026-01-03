from __future__ import annotations

"""
Session Manager - Orchestratore di Sessioni e Workspace (Runtime)

Versione RUNTIME-CLEAN:
- Nessuna dipendenza da engine.*
- Responsabilità limitata a:
  - discovery workspace
  - lifecycle workspace/session
  - context management
- AI settings e backend config restano orchestration-level
"""

import os
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Optional, Dict, List

from ice_core.logging.bridge import get_logger

from ..base import BackendConfig, BackendType, StorageMode
from ..exceptions import (
    SessionError,
    WorkspaceError,
    WorkspaceNotFoundError,
    WorkspaceAlreadyExistsError,
)

from .context import SessionContext, SessionConfig
from .lifecycle import SessionLifecycle, LifecycleError
from .workspace import (
    Workspace,
    WorkspaceMetadata,
    WorkspaceState,
    WorkspaceLifecycleHook,
    AIWorkspaceSettings,
)

logger = get_logger(__name__)

# ============================================================================
# RUNTIME CONSTANTS (ex-engine.constants)
# ============================================================================

GLOBAL_KNOWLEDGE_WORKSPACE_ID = "__global__"

# ============================================================================
# HELPERS
# ============================================================================

def _default_workspace_storage_dir() -> Path:
    return Path(
        os.getenv(
            "ICE_STUDIO_WORKSPACE_STORAGE_DIR",
            Path.home() / ".ice_studio" / "workspaces",
        )
    )

# ============================================================================
# SESSION REGISTRY
# ============================================================================

class SessionRegistry:
    """
    Registry interno runtime.
    Traccia workspace caricati, path scoperti e context attivi.
    """

    def __init__(self) -> None:
        self._workspaces: dict[str, Workspace] = {}
        self._workspace_paths: dict[str, Path] = {}
        self._active_contexts: dict[str, SessionContext] = {}
        self._workspace_types: dict[str, str] = {}

    # --- workspace ----------------------------------------------------------

    def register_workspace(self, workspace: Workspace, workspace_type: str = "generic") -> None:
        self._workspaces[workspace.id] = workspace
        self._workspace_paths[workspace.id] = workspace.base_path
        self._workspace_types[workspace.id] = workspace_type

    def unregister_workspace(self, workspace_id: str) -> None:
        self._workspaces.pop(workspace_id, None)
        self._workspace_types.pop(workspace_id, None)

    def register_workspace_path(self, workspace_id: str, path: Path, workspace_type: str = "generic") -> None:
        self._workspace_paths[workspace_id] = path
        self._workspace_types.setdefault(workspace_id, workspace_type)

    def drop_workspace_path(self, workspace_id: str) -> None:
        self._workspace_paths.pop(workspace_id, None)

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        return self._workspaces.get(workspace_id)

    def has_workspace(self, workspace_id: str) -> bool:
        return workspace_id in self._workspaces

    def get_workspace_path(self, workspace_id: str) -> Optional[Path]:
        return self._workspace_paths.get(workspace_id)

    def get_workspace_type(self, workspace_id: str) -> str:
        return self._workspace_types.get(workspace_id, "generic")

    def list_workspace_ids(self) -> list[str]:
        return sorted(set(self._workspaces) | set(self._workspace_paths))

    # --- context ------------------------------------------------------------

    def register_context(self, context: SessionContext) -> None:
        self._active_contexts[context.context_id] = context

    def unregister_context(self, context_id: str) -> None:
        self._active_contexts.pop(context_id, None)

    def get_active_contexts(self) -> list[SessionContext]:
        return list(self._active_contexts.values())

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """
    Runtime SessionManager.

    Gestisce:
    - discovery workspace
    - load / activate / deactivate
    - SessionContext lifecycle
    """

    _instance: Optional["SessionManager"] = None

    def __init__(
        self,
        base_path: Path | str,
        default_backend_configs: Optional[dict[str, BackendConfig]] = None,
        lifecycle_hooks: Optional[list[WorkspaceLifecycleHook]] = None,
    ):
        self.base_path = Path(base_path)
        self.storage_root = _default_workspace_storage_dir()
        self.storage_root.mkdir(parents=True, exist_ok=True)

        self.default_backend_configs = default_backend_configs or self._get_default_backends()
        self.lifecycle_hooks = lifecycle_hooks or []

        self._registry = SessionRegistry()
        self._sessions: dict[str, Session] = {}

        self._initialized = False
        self._shutdown_requested = False

    # ---------------------------------------------------------------------
    # SINGLETON
    # ---------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            raise SessionError("SessionManager non inizializzato")
        return cls._instance

    @classmethod
    def set_instance(cls, instance: Optional["SessionManager"]) -> None:
        cls._instance = instance

    # ---------------------------------------------------------------------
    # INIT / SHUTDOWN
    # ---------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._initialized:
            return

        self.base_path.mkdir(parents=True, exist_ok=True)
        await self._discover_workspaces()

        self._initialized = True
        SessionManager.set_instance(self)

        logger.info("SessionManager inizializzato")

    async def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True

        for ctx in self._registry.get_active_contexts():
            try:
                await ctx.close()
            except Exception:
                pass

        for ws_id in self._registry.list_workspace_ids():
            ws = self._registry.get_workspace(ws_id)
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass

        self._initialized = False
        SessionManager.set_instance(None)

    # ---------------------------------------------------------------------
    # DISCOVERY
    # ---------------------------------------------------------------------

    async def _discover_workspaces(self) -> None:
        if not self.storage_root.exists():
            return

        for entry in self.storage_root.iterdir():
            if not entry.is_dir():
                continue
            try:
                metadata = Workspace.load_metadata(entry)
                ws_type = (
                    metadata.ai_config.get("workspace_type")
                    if metadata.ai_config
                    else metadata.settings.get("workspace_type", "generic")
                )
                self._registry.register_workspace_path(metadata.id, entry, ws_type)
            except Exception:
                continue

    # ---------------------------------------------------------------------
    # WORKSPACE LOAD / ACTIVATE
    # ---------------------------------------------------------------------

    async def load_workspace(self, workspace_id: str) -> Workspace:
        ws = self._registry.get_workspace(workspace_id)
        if ws:
            return ws

        await self._discover_workspaces()
        path = self._registry.get_workspace_path(workspace_id)

        if not path or not path.exists():
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        metadata = Workspace.load_metadata(path)
        ws_type = self._registry.get_workspace_type(workspace_id)

        ai_settings = None
        if metadata.ai_config:
            ai_settings = AIWorkspaceSettings.from_dict(metadata.ai_config)

        backends = (
            self._get_ai_backend_configs(ws_type)
            if ws_type != "generic"
            else self.default_backend_configs.copy()
        )

        ws = Workspace(
            id=metadata.id,
            name=metadata.name,
            base_path=path,
            backends=backends,
            metadata=metadata,
            lifecycle_hooks=self.lifecycle_hooks.copy(),
            ai_config=ai_settings,
        )

        await ws.initialize()
        self._registry.register_workspace(ws, ws_type)
        return ws

    async def activate_workspace(
        self,
        workspace_id: str,
        config: Optional[SessionConfig] = None,
        auto_load: bool = True,
    ) -> SessionContext:
        ws = (
            await self.load_workspace(workspace_id)
            if auto_load
            else self._registry.get_workspace(workspace_id)
        )

        if not ws:
            raise WorkspaceNotFoundError(workspace_id=workspace_id)

        for ctx in self._registry.get_active_contexts():
            if ctx.workspace_id == workspace_id:
                SessionContext.set_current(ctx)
                return ctx

        if not ws.is_active:
            if ws.state == WorkspaceState.SUSPENDED:
                await ws.resume()
            else:
                raise WorkspaceError(
                    workspace_id=workspace_id,
                    message=f"Workspace in stato non valido: {ws.state.value}",
                )

        ctx = SessionContext.create(
            workspace=ws,
            config=config,
            set_as_current=True,
            workspace_id=workspace_id,
        )

        self._registry.register_context(ctx)
        ctx.on_close(lambda c: self._registry.unregister_context(c.context_id))
        return ctx

    # ---------------------------------------------------------------------
    # LIST / INFO
    # ---------------------------------------------------------------------

    async def list_workspaces(self, include_hidden: bool = False) -> list[dict[str, Any]]:
        await self._discover_workspaces()
        result: list[dict[str, Any]] = []

        for ws_id in self._registry.list_workspace_ids():
            if not include_hidden and ws_id == GLOBAL_KNOWLEDGE_WORKSPACE_ID:
                continue

            ws = self._registry.get_workspace(ws_id)
            if ws:
                info = ws.get_info()
                info["loaded"] = True
                result.append(info)
                continue

            path = self._registry.get_workspace_path(ws_id)
            try:
                metadata = Workspace.load_metadata(path)
                result.append(
                    {
                        "id": metadata.id,
                        "name": metadata.name,
                        "state": "inactive",
                        "base_path": str(path),
                        "workspace_type": self._registry.get_workspace_type(ws_id),
                        "metadata": metadata.to_dict(),
                        "loaded": False,
                    }
                )
            except Exception:
                continue

        return result

    # ---------------------------------------------------------------------
    # DEFAULT BACKENDS
    # ---------------------------------------------------------------------

    def _get_default_backends(self) -> dict[str, BackendConfig]:
        relational = BackendConfig.sqlite("data.db")
        vector = BackendConfig.memory_cache()

        return {
            "relational": relational,
            "vector": vector,
            "llm_cache": BackendConfig.sqlite("llm_cache.db"),
        }

    def _get_ai_backend_configs(self, workspace_type: str) -> dict[str, BackendConfig]:
        configs = self._get_default_backends()
        if workspace_type == "knowledge_base":
            configs["vector"] = BackendConfig.memory_cache()
        return configs

# ============================================================================
# SESSION RECORD
# ============================================================================

@dataclass
class Session:
    id: str
    context: SessionContext
    lifecycle: SessionLifecycle
