"""
ICE Runtime — Capability Registry
=================================

Registro canonico dei tipi di Capability supportati dal Runtime.

Questo modulo:
- DEFINISCE quali capability esistono
- NON concede capability
- NON valida l'uso
- NON conosce Run, Agenti o Sessioni
- NON emette eventi

È una sorgente dichiarativa.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CapabilityType:
    """
    Definizione formale di un tipo di Capability.

    Questa struttura descrive COSA può essere concesso,
    non una concessione specifica.
    """

    capability_type: str
    scope: str

    description: Optional[str] = None

    revocable: bool = True
    default_ttl_seconds: Optional[int] = None


class CapabilityRegistry:
    """
    Registro in-memory dei tipi di Capability noti al Runtime.

    Il registry:
    - è inizializzato all'avvio del Runtime
    - è read-only durante l'esecuzione
    - NON cambia dinamicamente
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, CapabilityType] = {}

    def register(self, capability: CapabilityType) -> None:
        """
        Registra un nuovo tipo di capability.

        Non può essere sovrascritta.
        """
        assert capability.capability_type, "capability_type is required"

        if capability.capability_type in self._capabilities:
            raise ValueError(
                f"Capability '{capability.capability_type}' already registered"
            )

        self._capabilities[capability.capability_type] = capability

    def get(self, capability_type: str) -> CapabilityType:
        """
        Recupera una capability per tipo.

        Lancia KeyError se non esiste.
        """
        return self._capabilities[capability_type]

    def exists(self, capability_type: str) -> bool:
        """
        Verifica se una capability è definita.
        """
        return capability_type in self._capabilities

    def list_all(self) -> Dict[str, CapabilityType]:
        """
        Restituisce TUTTE le capability registrate.

        È una copia difensiva.
        """
        return dict(self._capabilities)
