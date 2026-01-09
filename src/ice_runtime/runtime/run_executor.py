from __future__ import annotations

"""
ICE Runtime — Run Executor
==========================

Implementazione canonica dell’esecuzione di UN singolo Run ICE.

Riferimenti:
- RFC-ICE-002 (Execution Model)
- RFC-ICE-005 (Run State Machine)

Questo modulo:
- NON esegue inferenza
- NON conosce agenti
- NON orchestra
- NON fa IO di dominio

Fa UNA cosa sola:
→ applica il lifecycle
→ emette eventi validi
→ garantisce abort deterministico

Se questo file è sbagliato, ICE non esiste.
"""

from ice_runtime.events.kernel.emitter import EventEmitter
from ice_runtime.events.kernel.event import ICEEvent
from ice_runtime.runtime.state_machine import RunStateMachine
from ice_runtime.runtime.errors import RunExecutionError
from ice_runtime.ids.runtime_id import RunID


class RunExecutor:
    """
    Esecutore sovrano di UN singolo Run.

    Proprietà:
    - non riutilizzabile
    - non concorrente
    - deterministico
    """

    def __init__(
        self,
        *,
        run_id: RunID,
        emitter: EventEmitter,
        state_machine: RunStateMachine,
    ) -> None:
        self._run_id = run_id
        self._emitter = emitter
        self._sm = state_machine

    # ------------------------------------------------------------------
    # API PUBBLICA
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """
        Esegue l’intero lifecycle del Run.

        Questa è l’UNICA entrypoint.
        """
        try:
            self._provision()
            self._load_context()
            self._execute_window()
            self._validate()
            self._commit()
            self._finalize()

        except Exception as exc:
            self._abort(reason=str(exc))
            self._finalize()

    # ------------------------------------------------------------------
    # FASI CANONICHE (RFC-ICE-002)
    # ------------------------------------------------------------------

    def _provision(self) -> None:
        self._sm.transition(self._sm.PROVISIONED)

        self._emit("RunProvisioned")
        self._emit("ResourcesAllocated")

    def _load_context(self) -> None:
        self._sm.transition(self._sm.CONTEXT_READY)

        self._emit("ContextResolved")
        self._emit("MemoryMounted")
        self._emit("CapabilitiesBound")

    def _execute_window(self) -> None:
        self._sm.transition(self._sm.EXECUTING)

        # NOTA FONDATIVA:
        # Questa fase NON esegue agenti.
        # Apre una finestra temporale sotto controllo Runtime.
        self._emit("ExecutionWindowOpened")

    def _validate(self) -> None:
        self._sm.transition(self._sm.VALIDATING)

        self._emit("ValidationStarted")
        self._emit("ValidationPassed")

    def _commit(self) -> None:
        self._sm.transition(self._sm.COMMITTED)

        self._emit("RunCommitted")

    # ------------------------------------------------------------------
    # ABORT & FINALIZATION
    # ------------------------------------------------------------------

    def _abort(self, *, reason: str) -> None:
        if self._sm.is_terminal():
            return

        self._sm.abort()

        self._emit("RunAborted")
        self._emit("AbortReasonDeclared", payload={"reason": reason})

    def _finalize(self) -> None:
        if self._sm.is_terminal():
            return

        self._emit("ResourcesReleased")

        self._sm.finalize()

        self._emit("RunTerminated")

    # ------------------------------------------------------------------
    # EVENT EMISSION
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, payload: dict | None = None) -> ICEEvent:
        return self._emitter.emit(
            run_id=self._run_id,
            event_type=event_type,
            origin="runtime",
            payload=payload or {},
        )
