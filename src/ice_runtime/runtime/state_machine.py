"""
ICE Runtime — Run State Machine
===============================

Enforcement formale della RFC-ICE-005.

Questo modulo:
- NON introduce semantica nuova
- NON contiene logica di business
- NON emette eventi
- NON gestisce risorse

Il suo unico compito è:
→ validare transizioni di stato
→ impedire stati illegali
→ rendere impossibili shortcut

Se questo modulo è permissivo, ICE è corrotto.
"""

from typing import Optional


class InvalidStateTransition(Exception):
    """Transizione di stato non conforme alla RFC-ICE-005."""


class RunStateMachine:
    """
    State Machine sovrana di un Run.

    Lo stato:
    - NON è mutabile direttamente
    - NON è una sorgente di verità
    - È una proiezione vincolata

    Le transizioni sono l'unico meccanismo ammesso.
    """

    # Stati canonici (CHIUSI)
    CREATED = "CREATED"
    PROVISIONED = "PROVISIONED"
    CONTEXT_READY = "CONTEXT_READY"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"
    TERMINATED_BY_RUNTIME = "TERMINATED_BY_RUNTIME"
    TERMINATED = "TERMINATED"

    # Transizioni ammesse (RFC-ICE-005)
    _TRANSITIONS = {
        CREATED: {PROVISIONED},
        PROVISIONED: {CONTEXT_READY},
        CONTEXT_READY: {EXECUTING},
        EXECUTING: {EXECUTING, VALIDATING, ABORTED},
        VALIDATING: {COMMITTED, ABORTED},
        COMMITTED: {TERMINATED},
        ABORTED: {TERMINATED},
        TERMINATED_BY_RUNTIME: {TERMINATED},
        TERMINATED: set(),
    }

    def __init__(self) -> None:
        self._state: str = self.CREATED

    # ------------------------------------------------------------------ #
    # API PUBBLICA
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        """
        Stato corrente (read-only).
        """
        return self._state

    def transition(self, from_state: str, to_state: str) -> None:
        """
        Esegue una transizione esplicita.

        È responsabilità del chiamante dichiarare
        lo stato di partenza atteso.
        """
        if self._state != from_state:
            raise InvalidStateTransition(
                f"Expected state {from_state}, but current state is {self._state}"
            )

        self._assert_transition_allowed(from_state, to_state)
        self._state = to_state

    def force_abort(self) -> None:
        """
        Abort sovrano del Runtime.

        Ammesso da QUALSIASI stato tranne TERMINATED.
        """
        if self._state == self.TERMINATED:
            return

        self._state = self.ABORTED

    def terminate(self) -> None:
        """
        Chiusura finale del Run.

        Ammessa solo da stati finali.
        """
        if self._state not in {self.COMMITTED, self.ABORTED, self.TERMINATED_BY_RUNTIME}:
            raise InvalidStateTransition(
                f"Cannot terminate Run from state {self._state}"
            )

        self._state = self.TERMINATED

    # ------------------------------------------------------------------ #
    # INTERNALS
    # ------------------------------------------------------------------ #

    def _assert_transition_allowed(self, from_state: str, to_state: str) -> None:
        allowed = self._TRANSITIONS.get(from_state)

        if allowed is None:
            raise InvalidStateTransition(f"Unknown state: {from_state}")

        if to_state not in allowed:
            raise InvalidStateTransition(
                f"Illegal transition: {from_state} → {to_state}"
            )
