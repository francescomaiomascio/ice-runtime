from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any

from ice_runtime.sessions.project_tree import ProjectTree


class WorkspaceState(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    CLOSED = "closed"
    ERROR = "error"


@dataclass(frozen=True)
class WorkspaceMetadata:
    workspace_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
        }


class Workspace:
    """
    ICE Runtime — Workspace

    Boundary strutturale del filesystem.
    NON contiene logica applicativa.
    """

    def __init__(
        self,
        *,
        workspace_id: str,
        name: str,
        base_path: Path,
    ) -> None:
        self.workspace_id = workspace_id
        self.name = name
        self.base_path = base_path.resolve()

        self.metadata = WorkspaceMetadata(
            workspace_id=workspace_id,
            name=name,
        )

        self.state: WorkspaceState = WorkspaceState.CREATED

        self._project_tree = ProjectTree(root=self.base_path)

    # -------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------

    def initialize(self) -> None:
        if self.state != WorkspaceState.CREATED:
            return

        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.state = WorkspaceState.ACTIVE
        except Exception:
            self.state = WorkspaceState.ERROR
            raise

    def close(self) -> None:
        if self.state == WorkspaceState.CLOSED:
            return

        self.state = WorkspaceState.CLOSED

    # -------------------------------------------------
    # READ-ONLY VIEWS
    # -------------------------------------------------

    def project_tree(self) -> Dict[str, Any]:
        return self._project_tree.build()

    def info(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "state": self.state.value,
            "base_path": str(self.base_path),
            "metadata": self.metadata.to_dict(),
        }
