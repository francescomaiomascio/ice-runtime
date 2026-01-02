from ice_core.logging.bridge import get_logger
"""
Lifecycle Hooks - Implementazioni Concrete

Fornisce implementazioni pronte all'uso dei WorkspaceLifecycleHook:
- LoggingHook: Log dettagliati degli eventi
- MetricsHook: Tracking metriche (tempo, operazioni)
- ValidationHook: Validazione stato workspace
- CleanupHook: Cleanup automatico risorse
- NotificationHook: Notifiche eventi (callbacks custom)
- BackupHook: Backup automatico su close

Usage:
    from storage.session.lifecycle import LoggingHook, MetricsHook, BackupHook
    
    manager = SessionManager(
        base_path="./data",
        lifecycle_hooks=[
            LoggingHook(log_level="INFO"),
            MetricsHook(track_performance=True),
            BackupHook(backup_on_close=True, backup_dir="./backups")
        ]
    )
"""

from typing import Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import logging
import time

from engine.storage.exceptions import SessionError
from .workspace import Workspace, WorkspaceLifecycleHook

logger = get_logger(__name__)

# ============================================================================
# SESSION LIFECYCLE STATE MACHINE
# ============================================================================


class LifecycleError(SessionError):
    """Errore per transizioni di lifecycle non valide."""


class SessionLifecycle:
    """
    State machine semplificata per il lifecycle di una sessione/workspace.

    Stati supportati:
    - CREATED (stato iniziale)
    - ACTIVE
    - PAUSED
    - FINISHED
    """

    VALID_STATES = {"CREATED", "ACTIVE", "PAUSED", "FINISHED"}
    ALLOWED_TRANSITIONS = {
        "CREATED": {"ACTIVE"},
        "ACTIVE": {"PAUSED", "FINISHED"},
        "PAUSED": {"ACTIVE", "FINISHED"},
        "FINISHED": set(),
    }

    def __init__(self, metadata: Optional[dict[str, Any]] = None):
        self.state: str = "CREATED"
        self.created_at: datetime = datetime.now()
        self.first_activation_at: Optional[datetime] = None
        self.last_updated_at: datetime = self.created_at
        self.metadata: dict[str, Any] = metadata.copy() if metadata else {}

    def transition(self, new_state: str) -> None:
        """Esegue una transizione di stato, validando il cambio."""
        new_state = new_state.upper()
        if new_state not in self.VALID_STATES:
            raise LifecycleError(f"Stato non valido: {new_state}")

        if new_state not in self.ALLOWED_TRANSITIONS[self.state]:
            raise LifecycleError(f"Transizione non valida: {self.state} -> {new_state}")

        # Aggiorna timestamps
        now = datetime.now()
        if new_state == "ACTIVE" and self.first_activation_at is None:
            self.first_activation_at = now

        self.state = new_state
        self.last_updated_at = now

    def is_active(self) -> bool:
        return self.state == "ACTIVE"

    def is_finished(self) -> bool:
        return self.state == "FINISHED"

    def duration_seconds(self) -> float:
        """
        Durata dalla prima attivazione fino all'ultimo update.
        """
        if not self.first_activation_at:
            return 0.0
        end_time = self.last_updated_at
        return max(0.0, (end_time - self.first_activation_at).total_seconds())

    def reset(self) -> None:
        """Ripristina lo stato iniziale e pulisce i metadata."""
        self.state = "CREATED"
        self.first_activation_at = None
        self.last_updated_at = datetime.now()
        self.metadata.clear()

# ============================================================================
# LOGGING HOOK
# ============================================================================

class LoggingHook(WorkspaceLifecycleHook):
    """
    Hook per logging dettagliato degli eventi workspace.
    
    Log di:
    - Inizializzazione
    - Attivazione/sospensione
    - Chiusura
    - Errori
    
    Configurabile per livello di dettaglio.
    
    Usage:
        hook = LoggingHook(log_level="INFO", detailed=True)
    """
    
    def __init__(
        self,
        log_level: str = "INFO",
        detailed: bool = False,
        logger_name: Optional[str] = None
    ):
        """
        Args:
            log_level: Livello log (DEBUG, INFO, WARNING, ERROR)
            detailed: Se True, log dettagliati con stats
            logger_name: Nome custom logger (usa default se None)
        """
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.detailed = detailed
        self.logger = get_logger(logger_name or __name__)
    
    def on_initialize(self, workspace: Workspace) -> None:
        """Log inizializzazione workspace."""
        msg = f"Workspace inizializzato: {workspace.id} ({workspace.name})"
        
        if self.detailed:
            backends = ", ".join(workspace.list_backends())
            msg += f" | backends=[{backends}]"
        
        self.logger.log(self.log_level, msg)
    
    def on_activate(self, workspace: Workspace) -> None:
        """Log attivazione workspace."""
        msg = f"Workspace attivato: {workspace.id}"
        
        if self.detailed:
            msg += f" | state={workspace.state.value}"
        
        self.logger.log(self.log_level, msg)
    
    def on_suspend(self, workspace: Workspace) -> None:
        """Log sospensione workspace."""
        msg = f"Workspace sospeso: {workspace.id}"
        self.logger.log(self.log_level, msg)
    
    def on_close(self, workspace: Workspace) -> None:
        """Log chiusura workspace."""
        msg = f"Workspace chiuso: {workspace.id}"
        
        if self.detailed:
            info = workspace.get_info()
            backend_stats = []
            for name, backend_info in info.get("backends", {}).items():
                stats = backend_info.get("stats", {})
                if "total_rows" in stats:
                    backend_stats.append(f"{name}={stats['total_rows']} rows")
            
            if backend_stats:
                msg += f" | stats=[{', '.join(backend_stats)}]"
        
        self.logger.log(self.log_level, msg)
    
    def on_error(self, workspace: Workspace, error: Exception) -> None:
        """Log errori workspace."""
        self.logger.error(
            f"Errore workspace {workspace.id}: {error}",
            exc_info=self.detailed
        )


# ============================================================================
# METRICS HOOK
# ============================================================================

class MetricsHook(WorkspaceLifecycleHook):
    """
    Hook per tracking metriche workspace.
    
    Traccia:
    - Tempo di inizializzazione
    - Uptime workspace
    - Eventi lifecycle
    - Performance metrics
    
    Usage:
        hook = MetricsHook(track_performance=True)
        
        # Accesso metriche
        metrics = hook.get_metrics(workspace_id)
    """
    
    def __init__(
        self,
        track_performance: bool = True,
        track_events: bool = True
    ):
        """
        Args:
            track_performance: Se True, traccia timing
            track_events: Se True, traccia contatori eventi
        """
        self.track_performance = track_performance
        self.track_events = track_events
        
        # Storage metriche (in-memory)
        self._metrics: dict[str, dict[str, Any]] = {}
    
    def _get_workspace_metrics(self, workspace_id: str) -> dict[str, Any]:
        """Ottiene o crea dict metriche per workspace."""
        if workspace_id not in self._metrics:
            self._metrics[workspace_id] = {
                "events": {
                    "initialize": 0,
                    "activate": 0,
                    "suspend": 0,
                    "close": 0,
                    "error": 0,
                },
                "timing": {
                    "initialized_at": None,
                    "last_activated_at": None,
                    "closed_at": None,
                    "initialization_duration": 0.0,
                },
            }
        return self._metrics[workspace_id]
    
    def on_initialize(self, workspace: Workspace) -> None:
        """Traccia inizializzazione."""
        metrics = self._get_workspace_metrics(workspace.id)
        
        if self.track_events:
            metrics["events"]["initialize"] += 1
        
        if self.track_performance:
            metrics["timing"]["initialized_at"] = datetime.now()
            metrics["timing"]["_init_start"] = time.perf_counter()
    
    def on_activate(self, workspace: Workspace) -> None:
        """Traccia attivazione."""
        metrics = self._get_workspace_metrics(workspace.id)
        
        if self.track_events:
            metrics["events"]["activate"] += 1
        
        if self.track_performance:
            metrics["timing"]["last_activated_at"] = datetime.now()
            
            # Calcola tempo di init se disponibile
            if "_init_start" in metrics["timing"]:
                duration = time.perf_counter() - metrics["timing"]["_init_start"]
                metrics["timing"]["initialization_duration"] = duration
                del metrics["timing"]["_init_start"]
    
    def on_suspend(self, workspace: Workspace) -> None:
        """Traccia sospensione."""
        metrics = self._get_workspace_metrics(workspace.id)
        
        if self.track_events:
            metrics["events"]["suspend"] += 1
    
    def on_close(self, workspace: Workspace) -> None:
        """Traccia chiusura."""
        metrics = self._get_workspace_metrics(workspace.id)
        
        if self.track_events:
            metrics["events"]["close"] += 1
        
        if self.track_performance:
            metrics["timing"]["closed_at"] = datetime.now()
    
    def on_error(self, workspace: Workspace, error: Exception) -> None:
        """Traccia errori."""
        metrics = self._get_workspace_metrics(workspace.id)
        
        if self.track_events:
            metrics["events"]["error"] += 1
    
    def get_metrics(self, workspace_id: str) -> dict[str, Any]:
        """
        Ottiene metriche per un workspace.
        
        Args:
            workspace_id: ID workspace
        
        Returns:
            Dict con metriche
        """
        return self._metrics.get(workspace_id, {})
    
    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Ottiene tutte le metriche."""
        return self._metrics.copy()
    
    def clear_metrics(self, workspace_id: Optional[str] = None) -> None:
        """
        Pulisce metriche.
        
        Args:
            workspace_id: ID workspace (None = tutti)
        """
        if workspace_id:
            self._metrics.pop(workspace_id, None)
        else:
            self._metrics.clear()


# ============================================================================
# VALIDATION HOOK
# ============================================================================

class ValidationHook(WorkspaceLifecycleHook):
    """
    Hook per validazione stato workspace.
    
    Verifica:
    - Backends connessi correttamente
    - Schema inizializzato
    - Integrity checks
    
    Usage:
        hook = ValidationHook(strict=True)
    """
    
    def __init__(self, strict: bool = False):
        """
        Args:
            strict: Se True, solleva exception su validation failure
        """
        self.strict = strict
    
    def on_initialize(self, workspace: Workspace) -> None:
        """Valida inizializzazione."""
        # Verifica backend connessi
        for backend_name in workspace.list_backends():
            backend = workspace.get_backend(backend_name)
            
            if not backend.is_connected():
                msg = f"Backend {backend_name} non connesso dopo inizializzazione"
                logger.error(msg)
                
                if self.strict:
                    raise RuntimeError(msg)
    
    def on_activate(self, workspace: Workspace) -> None:
        """Valida attivazione."""
        if not workspace.is_active:
            msg = f"Workspace {workspace.id} non attivo dopo on_activate"
            logger.error(msg)
            
            if self.strict:
                raise RuntimeError(msg)
    
    def on_suspend(self, workspace: Workspace) -> None:
        """Valida sospensione."""
        pass
    
    def on_close(self, workspace: Workspace) -> None:
        """Valida chiusura."""
        # Verifica backend disconnessi
        for backend_name in workspace.list_backends():
            backend = workspace.get_backend(backend_name)
            
            if backend.is_connected():
                msg = f"Backend {backend_name} ancora connesso dopo chiusura"
                logger.warning(msg)
    
    def on_error(self, workspace: Workspace, error: Exception) -> None:
        """Log errori di validazione."""
        logger.error(f"Errore validazione workspace {workspace.id}: {error}")


# ============================================================================
# CLEANUP HOOK
# ============================================================================

class CleanupHook(WorkspaceLifecycleHook):
    """
    Hook per cleanup automatico risorse.
    
    Cleanup:
    - Temp files
    - Cache
    - Expired data
    - Lock files
    
    Usage:
        hook = CleanupHook(cleanup_temp=True, cleanup_cache=True)
    """
    
    def __init__(
        self,
        cleanup_temp: bool = True,
        cleanup_cache: bool = False,
        cleanup_locks: bool = True
    ):
        """
        Args:
            cleanup_temp: Rimuovi file temporanei
            cleanup_cache: Rimuovi cache
            cleanup_locks: Rimuovi lock files
        """
        self.cleanup_temp = cleanup_temp
        self.cleanup_cache = cleanup_cache
        self.cleanup_locks = cleanup_locks
    
    def on_initialize(self, workspace: Workspace) -> None:
        """Cleanup iniziale (lock files vecchi)."""
        if self.cleanup_locks:
            self._cleanup_locks(workspace)
    
    def on_activate(self, workspace: Workspace) -> None:
        """Nessun cleanup su activate."""
        pass
    
    def on_suspend(self, workspace: Workspace) -> None:
        """Cleanup su suspend (cache)."""
        if self.cleanup_cache:
            self._cleanup_cache(workspace)
    
    def on_close(self, workspace: Workspace) -> None:
        """Cleanup completo su close."""
        if self.cleanup_temp:
            self._cleanup_temp(workspace)
        
        if self.cleanup_cache:
            self._cleanup_cache(workspace)
        
        if self.cleanup_locks:
            self._cleanup_locks(workspace)
    
    def on_error(self, workspace: Workspace, error: Exception) -> None:
        """Cleanup su errore."""
        pass
    
    def _cleanup_temp(self, workspace: Workspace) -> None:
        """Rimuove file temporanei."""
        temp_dir = workspace.base_path / "temp"
        if temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"Temp files rimossi: {workspace.id}")
            except Exception as e:
                logger.warning(f"Errore cleanup temp: {e}")
    
    def _cleanup_cache(self, workspace: Workspace) -> None:
        """Rimuove cache."""
        cache_dir = workspace.base_path / "cache"
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                logger.debug(f"Cache rimossa: {workspace.id}")
            except Exception as e:
                logger.warning(f"Errore cleanup cache: {e}")
    
    def _cleanup_locks(self, workspace: Workspace) -> None:
        """Rimuove lock files."""
        for lock_file in workspace.base_path.glob("*.lock"):
            try:
                lock_file.unlink()
                logger.debug(f"Lock rimosso: {lock_file}")
            except Exception as e:
                logger.warning(f"Errore rimozione lock: {e}")


# ============================================================================
# NOTIFICATION HOOK
# ============================================================================

class NotificationHook(WorkspaceLifecycleHook):
    """
    Hook per notifiche custom su eventi workspace.
    
    Permette di registrare callback per ogni evento lifecycle.
    
    Usage:
        def on_workspace_ready(workspace):
            print(f"Workspace {workspace.name} pronto!")
        
        hook = NotificationHook()
        hook.register_callback("initialize", on_workspace_ready)
    """
    
    def __init__(self):
        self._callbacks: dict[str, list[Callable]] = {
            "initialize": [],
            "activate": [],
            "suspend": [],
            "close": [],
            "error": [],
        }
    
    def register_callback(
        self,
        event: str,
        callback: Callable[[Workspace], None]
    ) -> None:
        """
        Registra callback per un evento.
        
        Args:
            event: Nome evento (initialize, activate, suspend, close, error)
            callback: Funzione da chiamare
        """
        if event not in self._callbacks:
            raise ValueError(f"Evento non valido: {event}")
        
        self._callbacks[event].append(callback)
        logger.debug(f"Callback registrato per {event}")
    
    def _trigger_callbacks(self, event: str, workspace: Workspace, **kwargs) -> None:
        """Esegue callbacks per un evento."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(workspace, **kwargs)
            except Exception as e:
                logger.error(f"Errore in callback {event}: {e}")
    
    def on_initialize(self, workspace: Workspace) -> None:
        """Trigger callbacks initialize."""
        self._trigger_callbacks("initialize", workspace)
    
    def on_activate(self, workspace: Workspace) -> None:
        """Trigger callbacks activate."""
        self._trigger_callbacks("activate", workspace)
    
    def on_suspend(self, workspace: Workspace) -> None:
        """Trigger callbacks suspend."""
        self._trigger_callbacks("suspend", workspace)
    
    def on_close(self, workspace: Workspace) -> None:
        """Trigger callbacks close."""
        self._trigger_callbacks("close", workspace)
    
    def on_error(self, workspace: Workspace, error: Exception) -> None:
        """Trigger callbacks error."""
        self._trigger_callbacks("error", workspace, error=error)


# ============================================================================
# BACKUP HOOK
# ============================================================================

class BackupHook(WorkspaceLifecycleHook):
    """
    Hook per backup automatico workspace.
    
    Backup:
    - Su chiusura workspace
    - Periodico (opzionale)
    - Incrementale o completo
    
    Usage:
        hook = BackupHook(
            backup_dir="./backups",
            backup_on_close=True,
            keep_last_n=5
        )
    """
    
    def __init__(
        self,
        backup_dir: Path | str,
        backup_on_close: bool = True,
        backup_on_suspend: bool = False,
        keep_last_n: int = 5,
        compress: bool = True
    ):
        """
        Args:
            backup_dir: Directory backup
            backup_on_close: Backup su chiusura
            backup_on_suspend: Backup su sospensione
            keep_last_n: Numero backup da mantenere (rotazione)
            compress: Se True, comprimi backup
        """
        self.backup_dir = Path(backup_dir)
        self.backup_on_close = backup_on_close
        self.backup_on_suspend = backup_on_suspend
        self.keep_last_n = keep_last_n
        self.compress = compress
        
        # Crea directory backup
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def on_initialize(self, workspace: Workspace) -> None:
        """Nessun backup su initialize."""
        pass
    
    def on_activate(self, workspace: Workspace) -> None:
        """Nessun backup su activate."""
        pass
    
    def on_suspend(self, workspace: Workspace) -> None:
        """Backup su suspend (opzionale)."""
        if self.backup_on_suspend:
            self._create_backup(workspace, reason="suspend")
    
    def on_close(self, workspace: Workspace) -> None:
        """Backup su close."""
        if self.backup_on_close:
            self._create_backup(workspace, reason="close")
    
    def on_error(self, workspace: Workspace, error: Exception) -> None:
        """Backup su errore (safe copy)."""
        try:
            self._create_backup(workspace, reason="error")
        except Exception as e:
            logger.error(f"Errore backup su error: {e}")
    
    def _create_backup(self, workspace: Workspace, reason: str = "manual") -> Path:
        """
        Crea backup del workspace.
        
        Args:
            workspace: Workspace da backuppare
            reason: Motivo backup (per naming)
        
        Returns:
            Path del backup creato
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{workspace.id}_{reason}_{timestamp}"
        
        if self.compress:
            backup_path = self.backup_dir / f"{backup_name}.tar.gz"
            self._create_compressed_backup(workspace, backup_path)
        else:
            backup_path = self.backup_dir / backup_name
            self._create_directory_backup(workspace, backup_path)
        
        logger.info(f"Backup creato: {backup_path}")
        
        # Rotazione backup
        self._rotate_backups(workspace.id)
        
        return backup_path
    
    def _create_directory_backup(self, workspace: Workspace, backup_path: Path) -> None:
        """Crea backup come directory (copia ricorsiva)."""
        import shutil
        shutil.copytree(workspace.base_path, backup_path)
    
    def _create_compressed_backup(self, workspace: Workspace, backup_path: Path) -> None:
        """Crea backup compresso (tar.gz)."""
        import tarfile
        
        with tarfile.open(backup_path, "w:gz") as tar:
            tar.add(workspace.base_path, arcname=workspace.id)
    
    def _rotate_backups(self, workspace_id: str) -> None:
        """Rotazione backup (mantieni ultimi N)."""
        # Trova backup per questo workspace
        pattern = f"{workspace_id}_*"
        backups = sorted(
            self.backup_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Rimuovi backup vecchi
        for old_backup in backups[self.keep_last_n:]:
            try:
                if old_backup.is_dir():
                    import shutil
                    shutil.rmtree(old_backup)
                else:
                    old_backup.unlink()
                
                logger.debug(f"Backup vecchio rimosso: {old_backup}")
            except Exception as e:
                logger.warning(f"Errore rimozione backup: {e}")


# ============================================================================
# HOOK REGISTRY
# ============================================================================

class HookRegistry:
    """
    Registry per gestire hook globali.
    
    Permette di registrare hook da usare per tutti i workspace.
    
    Usage:
        registry = HookRegistry()
        registry.register("logging", LoggingHook())
        registry.register("metrics", MetricsHook())
        
        hooks = registry.get_all_hooks()
    """
    
    def __init__(self):
        self._hooks: dict[str, WorkspaceLifecycleHook] = {}
    
    def register(self, name: str, hook: WorkspaceLifecycleHook) -> None:
        """Registra un hook."""
        self._hooks[name] = hook
        logger.debug(f"Hook registrato: {name}")
    
    def unregister(self, name: str) -> None:
        """Rimuove un hook."""
        self._hooks.pop(name, None)
        logger.debug(f"Hook rimosso: {name}")
    
    def get_hook(self, name: str) -> Optional[WorkspaceLifecycleHook]:
        """Ottiene un hook per nome."""
        return self._hooks.get(name)
    
    def get_all_hooks(self) -> list[WorkspaceLifecycleHook]:
        """Ottiene tutti gli hook registrati."""
        return list(self._hooks.values())
    
    def clear(self) -> None:
        """Rimuove tutti gli hook."""
        self._hooks.clear()


# ============================================================================
# DEFAULT HOOKS
# ============================================================================

def get_default_hooks(
    enable_logging: bool = True,
    enable_metrics: bool = True,
    enable_validation: bool = False,
    enable_cleanup: bool = True,
    enable_backup: bool = False,
    backup_dir: Optional[Path] = None
) -> list[WorkspaceLifecycleHook]:
    """
    Ottiene set di hook di default.
    
    Args:
        enable_logging: Abilita logging hook
        enable_metrics: Abilita metrics hook
        enable_validation: Abilita validation hook
        enable_cleanup: Abilita cleanup hook
        enable_backup: Abilita backup hook
        backup_dir: Directory backup (richiesto se enable_backup=True)
    
    Returns:
        Lista di hook configurati
    """
    hooks = []
    
    if enable_logging:
        hooks.append(LoggingHook(log_level="INFO", detailed=False))
    
    if enable_metrics:
        hooks.append(MetricsHook(track_performance=True, track_events=True))
    
    if enable_validation:
        hooks.append(ValidationHook(strict=False))
    
    if enable_cleanup:
        hooks.append(CleanupHook(cleanup_temp=True, cleanup_cache=False))
    
    if enable_backup:
        if not backup_dir:
            raise ValueError("backup_dir richiesto se enable_backup=True")
        hooks.append(BackupHook(backup_dir=backup_dir, backup_on_close=True))
    
    return hooks