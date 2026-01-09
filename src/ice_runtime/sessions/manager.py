from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from ice_runtime.sessions.workspace import Workspace
from ice_runtime.sessions.errors import (
    SessionError,
    WorkspaceNotFoundError,
    WorkspaceAlreadyExistsError,
)


class SessionRegistry:
    """
    Registro runtime-side.

    NON persiste.
    NON è globale.
    Vive quanto il Runtime.
    """

    def __init__(self) -> None:
        self._workspaces: Dict[str, Workspace] = {}

    # -------------------------------------------------
    # WORKSPACES
    # -------------------------------------------------

    def register_workspace(self, workspace: Workspace) -> None:
        if workspace.workspace_id in self._workspaces:
            raise WorkspaceAlreadyExistsError(workspace.workspace_id)

        self._workspaces[workspace.workspace_id] = workspace

    def get_workspace(self, workspace_id: str) -> Workspace:
        try:
            return self._workspaces[workspace_id]
        except KeyError:
            raise WorkspaceNotFoundError(workspace_id)

    def list_workspaces(self) -> List[Workspace]:
        return list(self._workspaces.values())


class SessionManager:
    """
    Runtime SessionManager (NON singleton).

    Responsabilità:
    - gestire Workspace
    - fornire binding Run → Workspace
    - NON gestisce Run
    - NON gestisce Engine
    """

    def __init__(self, *, base_dir: Path) -> None:
        self._base_dir = base_dir.resolve()
        self._registry = SessionRegistry()

        self._base_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # WORKSPACE API
    # -------------------------------------------------

    def create_workspace(
        self,
        *,
        workspace_id: str,
        name: str,
        path: Path,
    ) -> Workspace:
        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            base_path=path,
        )

        workspace.initialize()
        self._registry.register_workspace(workspace)
        return workspace

    def get_workspace(self, workspace_id: str) -> Workspace:
        return self._registry.get_workspace(workspace_id)

    def list_workspaces(self) -> List[Workspace]:
        return self._registry.list_workspaces()
