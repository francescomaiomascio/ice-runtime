"""
ICE Runtime — Capability Grants
===============================

Rappresentazione formale di una Capability concessa a un Run.

Questo modulo:
- NON applica policy
- NON valida l'uso
- NON emette eventi
- NON conosce il Runtime

Definisce esclusivamente:
- cosa significa "capability concessa"
- quali campi sono obbligatori
- quali invarianti devono sempre valere
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CapabilityGrant:
    """
    Capability concessa a un Run specifico.

    Una CapabilityGrant:
    - è temporanea
    - è revocabile
    - NON è posseduta da un Agente
    - è sempre legata a un Run
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

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """
        Verifica se la capability è scaduta nel tempo.

        Nota:
        - NON revoca
        - NON emette eventi
        """
        if self.expires_at is None:
            return False

        current_time = now or datetime.utcnow()
        return current_time >= self.expires_at

    def assert_valid(self) -> None:
        """
        Verifica invarianti strutturali della capability.

        NON controlla:
        - policy
        - stato del Run
        - autorizzazioni

        Lancia AssertionError se la struttura è invalida.
        """
        assert self.capability_id, "capability_id is required"
        assert self.capability_type, "capability_type is required"
        assert self.run_id, "run_id is required"
        assert self.scope, "scope is required"
        assert self.issuer == "runtime", "issuer must be runtime"

        if self.expires_at is not None:
            assert self.expires_at >= self.granted_at, (
                "expires_at must be >= granted_at"
            )
