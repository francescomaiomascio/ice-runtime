from __future__ import annotations

"""
ICE Runtime — Run Context Lifecycle

Coordina la creazione e distruzione di RunContext.

Regole:
- NON mantiene stato
- NON persiste nulla
- NON muta Run o Session
- NON valida policy
- Emette SOLO eventi Runtime
"""

from typing import Optional, Iterable, Dict, Any

from ice_runtime.events.kernel.emitter import EventEmitter
from ice_runtime.events.kernel.event import ICEEvent
from ice_runtime.runtime.state import RunState
from ice_runtime.runtime.state_machine import RunStateMachine
from ice_runtime.sessions.context import RunContext
from ice_runtime.sessions.errors import SessionError


class RunContextLifecycle:
    """
    Lifecycle minimale e deterministico per RunContext.

    Questo componente:
    - costruisce viste
    - notifica il Runtime tramite eventi
    - NON possiede risorse
    """

    def __init__(self, *, emitter: EventEmitter) -> None:
        self._emitter = emitter

    # =====================================================
    # CREATION
    # =====================================================

    def create(
        self,
        *,
        run_id: str,
        agent_id: Optional[str],
        workspace_id: str,
        state: RunState,
        memory_views: Iterable,
        capabilities: Iterable[str],
        metadata: Dict[str, Any],
    ) -> RunContext:
        """
        Costruisce una RunContext read-only.

        Ammessa SOLO quando il Run è operativo.
        """

        canonical_state = state.state

        if canonical_state not in {
            RunStateMachine.CONTEXT_READY,
            RunStateMachine.EXECUTING,
        }:
            raise SessionError(
                f"RunContext cannot be created in state {canonical_state}"
            )

        self._emit_runtime_event(
            run_id=run_id,
            event_type="ContextResolved",
            payload={
                "agent_id": agent_id,
                "workspace_id": workspace_id,
            },
        )

        return RunContext(
            run_id=run_id,
            agent_id=agent_id,
            state=state,
            workspace_id=workspace_id,
            memory_views=memory_views,
            capabilities=capabilities,
            metadata=metadata,
        )

    # =====================================================
    # DESTRUCTION
    # =====================================================

    def destroy(self, *, run_id: str) -> None:
        """
        Notifica la distruzione logica di una RunContext.

        Nessun cleanup fisico.
        """

        self._emit_runtime_event(
            run_id=run_id,
            event_type="RunContextReleased",
            payload={},
        )

    # =====================================================
    # INTERNALS
    # =====================================================

    def _emit_runtime_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        event = ICEEvent(
            event_id=str(id(payload)),  # placeholder, verrà centralizzato
            run_id=run_id,
            event_type=event_type,
            origin="runtime",
            payload=payload,
            timestamp=None,  # verrà normalizzato dall'emitter
        )
        self._emitter.emit(event)
