"""
ICE Runtime — Core Authority
============================

Questo modulo definisce il Runtime ICE come entità sovrana.

Il Runtime:
- crea Run
- governa l'esecuzione
- istanzia executor e state machine
- NON esegue inferenza
- NON orchestra agenti
- NON fa IO di dominio

Il Runtime è:
→ autorità
→ boundary
→ punto di verità operativa
"""

from typing import Dict

from ice_runtime.ids.runtime_id import RunID
from ice_runtime.runtime.run_executor import RunExecutor
from ice_runtime.runtime.state_machine import RunStateMachine
from ice_runtime.runtime.state import RunState
from ice_runtime.events.kernel.emitter import EventEmitter
from ice_runtime.runtime.errors import RuntimeError


class Runtime:
    """
    Runtime ICE.

    Un'istanza di Runtime:
    - governa più Run
    - mantiene isolamento tra Run
    - espone solo API legittime
    """

    def __init__(self, *, emitter: EventEmitter) -> None:
        self._emitter = emitter
        self._runs: Dict[RunID, RunExecutor] = {}
        self._states: Dict[RunID, RunState] = {}

    # ------------------------------------------------------------------ #
    # API PUBBLICA
    # ------------------------------------------------------------------ #

    def create_run(self) -> RunID:
        """
        Crea un nuovo Run e ne registra lo stato iniziale.

        NON avvia l'esecuzione.
        """
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

        Ogni Run può essere eseguito UNA SOLA volta.
        """
        executor = self._get_executor(run_id)
        executor.execute()

    def get_run_state(self, run_id: RunID) -> RunState:
        """
        Restituisce lo stato DERIVATO di un Run.
        """
        if run_id not in self._states:
            raise RuntimeError(f"Unknown RunID: {run_id}")

        return self._states[run_id]

    # ------------------------------------------------------------------ #
    # INTERNALS
    # ------------------------------------------------------------------ #

    def _get_executor(self, run_id: RunID) -> RunExecutor:
        if run_id not in self._runs:
            raise RuntimeError(f"Run not found: {run_id}")

        return self._runs[run_id]
