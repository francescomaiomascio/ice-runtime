"""
ICE Runtime — Capability Registry
=================================

Registro canonico dei tipi di Capability supportati dal Runtime.

Questo modulo:
- DEFINISCE quali capability ESISTONO
- NON concede capability
- NON valida l'uso
- NON conosce Run, Agenti o Sessioni
- NON emette eventi

È una sorgente dichiarativa e normativa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class CapabilityRegistryInvariantViolation(Exception):
    """Violazione strutturale del Capability Registry."""
    pass


@dataclass(frozen=True)
class CapabilityType:
    """
    Definizione formale di un tipo di Capability.

    Descrive COSA può essere concesso,
    non una concessione specifica.
    """

    capability_type: str
    scope: str

    description: Optional[str] = None

    revocable: bool = True
    default_ttl_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.capability_type:
            raise CapabilityRegistryInvariantViolation(
                "capability_type is required"
            )

        if not self.scope:
            raise CapabilityRegistryInvariantViolation(
                "scope is required"
            )

        if self.default_ttl_seconds is not None:
            if self.default_ttl_seconds <= 0:
                raise CapabilityRegistryInvariantViolation(
                    "default_ttl_seconds must be > 0"
                )


class CapabilityRegistry:
    """
    Registro in-memory dei tipi di Capability noti al Runtime.

    Invarianti:
    - inizializzato all'avvio del Runtime
    - read-only durante l'esecuzione
    - nessuna mutazione dinamica
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityType] = {}

    # ------------------------------------------------------------------
    # Registrazione (bootstrap-time)
    # ------------------------------------------------------------------

    def register(self, capability: CapabilityType) -> None:
        """
        Registra un nuovo tipo di Capability.

        NON può essere sovrascritto.
        """
        if capability.capability_type in self._capabilities:
            raise CapabilityRegistryInvariantViolation(
                f"Capability already registered: {capability.capability_type}"
            )

        self._capabilities[capability.capability_type] = capability

    # ------------------------------------------------------------------
    # Accesso read-only
    # ------------------------------------------------------------------

    def get(self, capability_type: str) -> CapabilityType:
        """
        Recupera una CapabilityType.

        Solleva KeyError se non esiste.
        """
        return self._capabilities[capability_type]

    def exists(self, capability_type: str) -> bool:
        """
        Verifica se una capability è definita.
        """
        return capability_type in self._capabilities

    def list_all(self) -> Dict[str, CapabilityType]:
        """
        Ritorna una copia difensiva del registry.
        """
        return dict(self._capabilities)
