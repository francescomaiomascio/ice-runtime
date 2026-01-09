from __future__ import annotations

"""
ICE Runtime — Core Authority
============================

Il Runtime ICE è l'autorità sovrana di esecuzione.

Responsabilità:
- identità del runtime
- lifecycle globale
- creazione e governo dei Run
- isolamento causale
- accesso controllato allo stato derivato

NON:
- orchestrazione
- agenti
- inferenza
- IO di dominio
"""

from typing import Dict
from pathlib import Path

from ice_runtime.ids.runtime_id import RunID
from ice_runtime.runtime.run_executor import RunExecutor
from ice_runtime.runtime.state_machine import RunStateMachine
from ice_runtime.runtime.state import RunState
from ice_runtime.events.kernel.emitter import EventEmitter
from ice_runtime.events.kernel.store import EventStore
from ice_runtime.runtime.errors import RuntimeError


class Runtime:
    """
    Runtime ICE.

    Un Runtime:
    - è unico per dominio
    - governa più Run
    - mantiene isolamento e causalità
    """

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        runtime_id: str,
        base_dir: Path,
        emitter: EventEmitter,
        event_store: EventStore,
        log_router=None,
    ) -> None:
        self.runtime_id = runtime_id
        self.base_dir = base_dir

        self._emitter = emitter
        self._event_store = event_store
        self._log_router = log_router

        self._runs: Dict[RunID, RunExecutor] = {}
        self._states: Dict[RunID, RunState] = {}

        self._bootstrapped = False
        self._running = False
        self._stopped = False

    def mark_bootstrapped(self) -> None:
        if self._bootstrapped:
            raise RuntimeError("Runtime already bootstrapped")
        self._bootstrapped = True

    def start(self) -> None:
        if not self._bootstrapped:
            raise RuntimeError("Runtime not bootstrapped")
        if self._running:
            return
        if self._stopped:
            raise RuntimeError("Runtime already stopped")

        self._running = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._running = False
        self._stopped = True

    # ------------------------------------------------------------------
    # RUN API
    # ------------------------------------------------------------------

    def create_run(self) -> RunID:
        """
        Crea un nuovo Run.

        Non avvia l'esecuzione.
        """
        self._ensure_running()

        run_id = RunID.generate()

        state_machine = RunStateMachine()
        state = RunState(state_machine=state_machine)

        executor = RunExecutor(
            run_id=run_id,
            emitter=self._emitter,
            state_machine=state_machine,
        )

        self._runs[run_id] = executor
        self._states[run_id] = state

        return run_id

    def execute_run(self, run_id: RunID) -> None:
        """
        Esegue un Run esistente.
        """
        self._ensure_running()

        executor = self._get_executor(run_id)
        executor.execute()

    def get_run_state(self, run_id: RunID) -> RunState:
        """
        Restituisce lo stato DERIVATO di un Run.
        """
        if run_id not in self._states:
            raise RuntimeError(f"Unknown RunID: {run_id}")

        return self._states[run_id]

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _get_executor(self, run_id: RunID) -> RunExecutor:
        if run_id not in self._runs:
            raise RuntimeError(f"Run not found: {run_id}")
        return self._runs[run_id]

    def _ensure_running(self) -> None:
        if not self._running or self._stopped:
            raise RuntimeError("Runtime not running")
