"""
ICE Runtime — Derived Run State
===============================

Questo modulo espone lo stato DERIVATO di un Run.

Regole fondamentali:
- lo stato NON è sorgente di verità
- lo stato NON è mutabile
- lo stato NON guida il comportamento

Lo stato è:
→ una proiezione read-only
→ derivata da eventi + state machine
→ usabile per debug, UI, audit

RFC:
- RFC-ICE-003 (Event Model)
- RFC-ICE-005 (Run State Machine)
"""

from typing import Optional, Dict, Any

from ice_runtime.runtime.state_machine import RunStateMachine


class RunState:
    """
    Vista read-only dello stato corrente di un Run.

    Questa classe:
    - NON permette transizioni
    - NON espone metodi di mutazione
    - NON ha logica di business
    """

    def __init__(
        self,
        *,
        state_machine: RunStateMachine,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._state_machine = state_machine
        self._metadata = metadata or {}

    # ------------------------------------------------------------------ #
    # STATO CANONICO
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        """
        Stato canonico del Run (RFC-ICE-005).
        """
        return self._state_machine.state

    @property
    def is_terminal(self) -> bool:
        """
        Indica se il Run è in stato finale.
        """
        return self.state in {
            RunStateMachine.COMMITTED,
            RunStateMachine.ABORTED,
            RunStateMachine.TERMINATED_BY_RUNTIME,
            RunStateMachine.TERMINATED,
        }

    # ------------------------------------------------------------------ #
    # METADATA DERIVATA (NON AUTORITATIVA)
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Metadata accessoria (debug / UI / introspezione).

        NON è causale.
        NON influenza il runtime.
        """
        return dict(self._metadata)

    # ------------------------------------------------------------------ #
    # RAPPRESENTAZIONE
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Rappresentazione serializzabile dello stato.
        """
        return {
            "state": self.state,
            "is_terminal": self.is_terminal,
            "metadata": self.metadata,
        }
