"""
ICE Runtime — Run Executor
==========================

Questo modulo implementa l'esecuzione canonica di UN singolo Run ICE.

È una implementazione DIRETTA del lifecycle definito in:
- RFC-ICE-002 (Execution Model)
- RFC-ICE-005 (Run State Machine)

Il RunExecutor:
- NON esegue inferenza
- NON conosce agenti concreti
- NON fa IO diretto
- NON prende decisioni semantiche

Il suo unico compito è:
→ applicare il modello di esecuzione
→ emettere eventi validi
→ garantire abort deterministico su violazione

Se questo file è sbagliato, ICE non esiste.
"""

from typing import Optional

from ice_runtime.events.kernel.emitter import EventEmitter
from ice_runtime.events.kernel.event import ICEEvent
from ice_runtime.runtime.state_machine import RunStateMachine
from ice_runtime.runtime.errors import RunExecutionError
from ice_runtime.ids.runtime_id import RunID


class RunExecutor:
    """
    Esecutore sovrano di un singolo Run.

    Ogni istanza di RunExecutor governa:
    - UN Run
    - UN lifecycle completo
    - UNA sequenza causale di eventi

    Non è riutilizzabile.
    Non è concorrente.
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
        self._state_machine = state_machine

        self._terminated: bool = False

    # --------------------------------------------------------------------- #
    # API PUBBLICA
    # --------------------------------------------------------------------- #

    def execute(self) -> None:
        """
        Avvia ed esegue l'intero lifecycle del Run.

        Questa è l'UNICA entrypoint pubblica.
        Tutto il resto è implementazione interna.

        Qualsiasi eccezione NON gestita qui
        è una violazione architetturale.
        """
        try:
            self._phase_provisioning()
            self._phase_context_loading()
            self._phase_execution()
            self._phase_validation()
            self._phase_commit()

        except Exception as exc:
            # Qualsiasi errore = Abort valido
            self._abort(reason=str(exc))
            return

        finally:
            # Teardown SEMPRE eseguito
            self._teardown()

    # --------------------------------------------------------------------- #
    # FASI CANONICHE (RFC-ICE-002)
    # --------------------------------------------------------------------- #

    def _phase_provisioning(self) -> None:
        self._ensure_not_terminated()

        self._state_machine.transition("CREATED", "PROVISIONED")

        self._emit_runtime_event(
            event_type="RunProvisioned",
            payload={}
        )

        self._emit_runtime_event(
            event_type="ResourcesAllocated",
            payload={}
        )

    def _phase_context_loading(self) -> None:
        self._ensure_not_terminated()

        self._state_machine.transition("PROVISIONED", "CONTEXT_READY")

        self._emit_runtime_event(
            event_type="ContextResolved",
            payload={}
        )

        self._emit_runtime_event(
            event_type="MemoryMounted",
            payload={}
        )

        self._emit_runtime_event(
            event_type="CapabilitiesBound",
            payload={}
        )

    def _phase_execution(self) -> None:
        self._ensure_not_terminated()

        self._state_machine.transition("CONTEXT_READY", "EXECUTING")

        # NOTA CRITICA:
        # Qui il RunExecutor NON esegue nulla.
        # La fase EXECUTING è una finestra temporale
        # in cui altri componenti possono emettere eventi
        # sotto controllo Runtime.
        #
        # Questa implementazione minima non include agent loop.
        pass

    def _phase_validation(self) -> None:
        self._ensure_not_terminated()

        self._state_machine.transition("EXECUTING", "VALIDATING")

        self._emit_runtime_event(
            event_type="ValidationStarted",
            payload={}
        )

        # NOTA:
        # In questa versione fondativa,
        # la validazione è considerata passante.
        # Le policy verranno innestate qui.
        self._emit_runtime_event(
            event_type="ValidationPassed",
            payload={}
        )

    def _phase_commit(self) -> None:
        self._ensure_not_terminated()

        self._state_machine.transition("VALIDATING", "COMMITTED")

        self._emit_runtime_event(
            event_type="RunCommitted",
            payload={}
        )

    # --------------------------------------------------------------------- #
    # ABORT & TEARDOWN
    # --------------------------------------------------------------------- #

    def _abort(self, *, reason: str) -> None:
        if self._terminated:
            return

        self._state_machine.force_abort()

        self._emit_runtime_event(
            event_type="RunAborted",
            payload={}
        )

        self._emit_runtime_event(
            event_type="AbortReasonDeclared",
            payload={"reason": reason}
        )

        self._terminated = True

    def _teardown(self) -> None:
        if self._terminated:
            return

        self._emit_runtime_event(
            event_type="ResourcesReleased",
            payload={}
        )

        self._emit_runtime_event(
            event_type="RunTerminated",
            payload={}
        )

        self._state_machine.terminate()
        self._terminated = True

    # --------------------------------------------------------------------- #
    # UTILITIES
    # --------------------------------------------------------------------- #

    def _emit_runtime_event(self, *, event_type: str, payload: dict) -> ICEEvent:
        """
        Emette un evento di Runtime attraverso l'unico canale legittimo.
        """
        return self._emitter.emit(
            run_id=self._run_id,
            event_type=event_type,
            origin="runtime",
            payload=payload,
        )

    def _ensure_not_terminated(self) -> None:
        if self._terminated:
            raise RunExecutionError("Run already terminated")
