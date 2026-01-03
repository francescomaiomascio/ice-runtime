from __future__ import annotations

"""
Session & Workspace Lifecycle (Runtime)

Questo modulo definisce:
- State machine minimale per sessioni
- Hook lifecycle runtime-safe
- Implementazioni standard (logging, metrics, cleanup, backup)
- Zero dipendenze da engine / storage

Il lifecycle NON persiste stato:
è un coordinatore di eventi runtime.
"""

from typing import Any, Optional, Callable, Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import time
import logging
import tarfile
import shutil

from ice_core.logging.bridge import get_logger

from .workspace import Workspace, WorkspaceLifecycleHook
from ..exceptions import SessionError

logger = get_logger(__name__)

# ============================================================================
# SESSION LIFECYCLE STATE MACHINE
# ============================================================================

class LifecycleError(SessionError):
    """Errore per transizioni lifecycle non valide."""


class SessionLifecycle:
    """
    State machine minimale per sessioni runtime.

    Stati:
    - CREATED   → sessione creata
    - ACTIVE    → in uso
    - PAUSED    → sospesa (workspace ancora vivo)
    - FINISHED  → conclusa
    """

    VALID_STATES = {"CREATED", "ACTIVE", "PAUSED", "FINISHED"}
    TRANSITIONS = {
        "CREATED": {"ACTIVE"},
        "ACTIVE": {"PAUSED", "FINISHED"},
        "PAUSED": {"ACTIVE", "FINISHED"},
        "FINISHED": set(),
    }

    def __init__(self, metadata: Optional[dict[str, Any]] = None):
        self.state = "CREATED"
        self.created_at = datetime.now()
        self.first_activated_at: Optional[datetime] = None
        self.last_updated_at = self.created_at
        self.metadata = metadata.copy() if metadata else {}

    def transition(self, target: str) -> None:
        target = target.upper()
        if target not in self.VALID_STATES:
            raise LifecycleError(f"Stato lifecycle non valido: {target}")

        if target not in self.TRANSITIONS[self.state]:
            raise LifecycleError(f"Transizione non valida: {self.state} → {target}")

        now = datetime.now()
        if target == "ACTIVE" and self.first_activated_at is None:
            self.first_activated_at = now

        self.state = target
        self.last_updated_at = now

    def duration_seconds(self) -> float:
        if not self.first_activated_at:
            return 0.0
        return max(
            0.0,
            (self.last_updated_at - self.first_activated_at).total_seconds(),
        )

# ============================================================================
# LOGGING HOOK
# ============================================================================

class LoggingHook(WorkspaceLifecycleHook):
    """
    Hook di logging puro.
    Nessuna side-effect, solo osservabilità.
    """

    def __init__(self, level: str = "INFO", detailed: bool = False):
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.detailed = detailed
        self.logger = get_logger("workspace.lifecycle")

    def on_initialize(self, workspace: Workspace) -> None:
        self.logger.log(self.level, f"workspace.init {workspace.id}")

    def on_activate(self, workspace: Workspace) -> None:
        self.logger.log(self.level, f"workspace.activate {workspace.id}")

    def on_suspend(self, workspace: Workspace) -> None:
        self.logger.log(self.level, f"workspace.suspend {workspace.id}")

    def on_close(self, workspace: Workspace) -> None:
        self.logger.log(self.level, f"workspace.close {workspace.id}")

    def on_error(self, workspace: Workspace, error: Exception) -> None:
        self.logger.error(
            f"workspace.error {workspace.id}: {error}",
            exc_info=self.detailed,
        )

# ============================================================================
# METRICS HOOK
# ============================================================================

class MetricsHook(WorkspaceLifecycleHook):
    """
    Hook metriche in-memory.
    Nessuna persistenza, solo runtime insight.
    """

    def __init__(self):
        self._metrics: dict[str, dict[str, Any]] = {}

    def _m(self, ws: Workspace) -> dict[str, Any]:
        return self._metrics.setdefault(
            ws.id,
            {
                "events": {"init": 0, "activate": 0, "suspend": 0, "close": 0, "error": 0},
                "timing": {"init_start": None, "init_duration": None},
            },
        )

    def on_initialize(self, workspace: Workspace) -> None:
        m = self._m(workspace)
        m["events"]["init"] += 1
        m["timing"]["init_start"] = time.perf_counter()

    def on_activate(self, workspace: Workspace) -> None:
        m = self._m(workspace)
        m["events"]["activate"] += 1
        if m["timing"]["init_start"]:
            m["timing"]["init_duration"] = (
                time.perf_counter() - m["timing"]["init_start"]
            )
            m["timing"]["init_start"] = None

    def on_suspend(self, workspace: Workspace) -> None:
        self._m(workspace)["events"]["suspend"] += 1

    def on_close(self, workspace: Workspace) -> None:
        self._m(workspace)["events"]["close"] += 1

    def on_error(self, workspace: Workspace, error: Exception) -> None:
        self._m(workspace)["events"]["error"] += 1

    def get_metrics(self, workspace_id: str) -> dict[str, Any]:
        return self._metrics.get(workspace_id, {})

# ============================================================================
# CLEANUP HOOK
# ============================================================================

class CleanupHook(WorkspaceLifecycleHook):
    """
    Cleanup puramente filesystem-based.
    """

    def __init__(self, temp: bool = True, cache: bool = False):
        self.cleanup_temp = temp
        self.cleanup_cache = cache

    def on_close(self, workspace: Workspace) -> None:
        base = workspace.base_path

        if self.cleanup_temp:
            self._rm(base / "temp")

        if self.cleanup_cache:
            self._rm(base / "cache")

    def _rm(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            logger.debug(f"cleanup.removed {path}")
        except Exception as e:
            logger.warning(f"cleanup.failed {path}: {e}")

# ============================================================================
# NOTIFICATION HOOK
# ============================================================================

class NotificationHook(WorkspaceLifecycleHook):
    """
    Hook per callback custom.
    """

    def __init__(self):
        self._callbacks: dict[str, list[Callable]] = {
            "initialize": [],
            "activate": [],
            "suspend": [],
            "close": [],
            "error": [],
        }

    def register(self, event: str, fn: Callable) -> None:
        if event not in self._callbacks:
            raise ValueError(f"Evento lifecycle non valido: {event}")
        self._callbacks[event].append(fn)

    def _emit(self, event: str, workspace: Workspace, **kw) -> None:
        for fn in self._callbacks.get(event, []):
            try:
                fn(workspace, **kw)
            except Exception as e:
                logger.error(f"callback.{event}.error: {e}")

    def on_initialize(self, workspace: Workspace) -> None:
        self._emit("initialize", workspace)

    def on_activate(self, workspace: Workspace) -> None:
        self._emit("activate", workspace)

    def on_suspend(self, workspace: Workspace) -> None:
        self._emit("suspend", workspace)

    def on_close(self, workspace: Workspace) -> None:
        self._emit("close", workspace)

    def on_error(self, workspace: Workspace, error: Exception) -> None:
        self._emit("error", workspace, error=error)

# ============================================================================
# BACKUP HOOK
# ============================================================================

class BackupHook(WorkspaceLifecycleHook):
    """
    Backup filesystem-level del workspace.
    """

    def __init__(
        self,
        backup_dir: Path | str,
        on_close: bool = True,
        keep_last: int = 5,
    ):
        self.backup_dir = Path(backup_dir)
        self.on_close = on_close
        self.keep_last = keep_last
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def on_close(self, workspace: Workspace) -> None:
        if self.on_close:
            self._backup(workspace)

    def _backup(self, workspace: Workspace) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.backup_dir / f"{workspace.id}_{ts}.tar.gz"

        try:
            with tarfile.open(out, "w:gz") as tar:
                tar.add(workspace.base_path, arcname=workspace.id)
            logger.info(f"backup.created {out}")
        except Exception as e:
            logger.error(f"backup.failed {workspace.id}: {e}")
            return

        self._rotate(workspace.id)

    def _rotate(self, workspace_id: str) -> None:
        backups = sorted(
            self.backup_dir.glob(f"{workspace_id}_*.tar.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in backups[self.keep_last:]:
            try:
                old.unlink()
            except Exception:
                pass

# ============================================================================
# DEFAULT HOOK SET
# ============================================================================

def get_default_hooks(
    logging_enabled: bool = True,
    metrics_enabled: bool = True,
    cleanup_enabled: bool = True,
) -> list[WorkspaceLifecycleHook]:
    hooks: list[WorkspaceLifecycleHook] = []

    if logging_enabled:
        hooks.append(LoggingHook())

    if metrics_enabled:
        hooks.append(MetricsHook())

    if cleanup_enabled:
        hooks.append(CleanupHook())

    return hooks
