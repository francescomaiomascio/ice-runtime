from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any


class ProjectTree:
    """
    ICE Runtime — Project Tree (READ-ONLY)

    Vista deterministica del filesystem di un Workspace.

    - Nessuna semantica
    - Nessun logging
    - Nessun side-effect
    - Nessuna cache
    """

    def __init__(
        self,
        *,
        root: Path,
        ignore_hidden: bool = True,
        max_depth: Optional[int] = None,
    ) -> None:
        self._root = root.resolve()
        self._ignore_hidden = ignore_hidden
        self._max_depth = max_depth

        if not self._root.exists():
            raise FileNotFoundError(self._root)

    # -------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------

    def build(self) -> Dict[str, Any]:
        return {
            "root": str(self._root),
            "items": self._walk(self._root, depth=0),
        }

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _walk(self, directory: Path, *, depth: int) -> List[Dict[str, Any]]:
        if self._max_depth is not None and depth > self._max_depth:
            return []

        nodes: List[Dict[str, Any]] = []

        for entry in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if entry.is_symlink():
                continue

            if self._ignore_hidden and entry.name.startswith("."):
                continue

            nodes.append(self._node(entry, depth))

        return nodes

    def _node(self, path: Path, depth: int) -> Dict[str, Any]:
        if path.is_dir():
            return {
                "name": path.name,
                "path": str(path),
                "type": "folder",
                "children": self._walk(path, depth=depth + 1),
            }

        return {
            "name": path.name,
            "path": str(path),
            "type": "file",
        }
