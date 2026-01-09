"""
ICE Runtime — Capability Grants
===============================

Rappresentazione formale di una Capability concessa a un Run.

Questo modulo definisce COSA È una capability concessa.

NON:
- applica policy
- valida l'uso
- emette eventi
- conosce il Runtime

È una struttura normativa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


class CapabilityInvariantViolation(Exception):
    """Violazione strutturale di una CapabilityGrant."""
    pass


@dataclass(frozen=True)
class CapabilityGrant:
    """
    Capability concessa a un Run specifico.

    Invarianti:
    - sempre legata a UN Run
    - concessa SOLO dal Runtime
    - temporalmente definita (anche infinita)
    """

    capability_id: str
    capability_type: str

    run_id: str

    scope: str
    constraints: Dict[str, Any] = field(default_factory=dict)

    granted_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    issuer: str = "runtime"
    revocable: bool = True
    revoked: bool = False

    # ------------------------------------------------------------------
    # Post-init: invarianti strutturali
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._enforce_invariants()

    def _enforce_invariants(self) -> None:
        if not self.capability_id:
            raise CapabilityInvariantViolation("capability_id is required")

        if not self.capability_type:
            raise CapabilityInvariantViolation("capability_type is required")

        if not self.run_id:
            raise CapabilityInvariantViolation("run_id is required")

        if not self.scope:
            raise CapabilityInvariantViolation("scope is required")

        if self.issuer != "runtime":
            raise CapabilityInvariantViolation(
                "issuer must be 'runtime'"
            )

        if self.expires_at is not None:
            if self.expires_at < self.granted_at:
                raise CapabilityInvariantViolation(
                    "expires_at must be >= granted_at"
                )

    # ------------------------------------------------------------------
    # Derivazioni pure (NO side effects)
    # ------------------------------------------------------------------

    def is_expired(self, *, now: Optional[datetime] = None) -> bool:
        """
        Ritorna True se la capability è scaduta temporalmente.

        NON:
        - revoca
        - muta stato
        """
        if self.expires_at is None:
            return False

        current_time = now or datetime.utcnow()
        return current_time >= self.expires_at

    def allows_scope(self, requested_scope: str) -> bool:
        """
        Verifica se lo scope richiesto è incluso nello scope concesso.

        Implementazione minimale:
        - scope identico
        - oppure prefisso valido
        """
        return (
            requested_scope == self.scope
            or requested_scope.startswith(f"{self.scope}:")
        )
