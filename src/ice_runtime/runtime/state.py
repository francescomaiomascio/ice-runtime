from __future__ import annotations

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
→ derivata ESCLUSIVAMENTE dalla RunStateMachine
→ usabile per debug, UI, audit, introspezione

RFC:
- RFC-ICE-003 (Event Model)
- RFC-ICE-005 (Run State Machine)
"""

from typing import Dict, Any, Optional

from ice_runtime.runtime.state_machine import RunStateMachine


class RunState:
    """
    Vista read-only dello stato corrente di un Run.

    Questa classe:
    - NON permette transizioni
    - NON espone mutazioni
    - NON contiene logica di business
    - NON è autoritativa

    È una proiezione.
    """

    def __init__(
        self,
        *,
        state_machine: RunStateMachine,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._sm = state_machine
        self._metadata = metadata or {}

    # ------------------------------------------------------------------
    # STATO CANONICO
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """
        Stato canonico del Run (RFC-ICE-005).
        """
        return self._sm.state

    @property
    def is_terminal(self) -> bool:
        """
        True se il Run è in uno stato finale.
        """
        return self._sm.is_terminal()

    # ------------------------------------------------------------------
    # METADATA DERIVATA (NON AUTORITATIVA)
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Metadata accessoria (debug / UI / audit).

        NON è causale.
        NON influenza il Runtime.
        """
        return dict(self._metadata)

    # ------------------------------------------------------------------
    # SERIALIZZAZIONE
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Rappresentazione serializzabile dello stato del Run.
        """
        return {
            "state": self.state,
            "is_terminal": self.is_terminal,
            "metadata": self.metadata,
        }
