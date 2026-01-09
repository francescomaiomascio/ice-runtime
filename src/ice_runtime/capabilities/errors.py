"""
ICE Runtime — Capability Errors
===============================

Gerarchia canonica degli errori del Capability System.

Regole:
- Ogni CapabilityError è FATALE per il Run
- Gli errori NON applicano logica
- Gli errori NON emettono eventi
- La semantica è interpretata dal Runtime / Enforcement
"""


class CapabilityError(Exception):
    """
    Errore base del sistema Capability.

    Qualsiasi CapabilityError implica:
    - uso illegittimo di una capability
    - violazione di un vincolo runtime-side
    """
    pass


# ----------------------------------------------------------------------
# Errori di definizione / registry
# ----------------------------------------------------------------------

class CapabilityNotRegisteredError(CapabilityError):
    """
    CapabilityType non definita nel registry.
    """
    def __init__(self, capability_type: str) -> None:
        super().__init__(
            f"Capability type not registered: {capability_type}"
        )
        self.capability_type = capability_type


class CapabilityNotGrantedError(CapabilityError):
    """
    Tentativo di usare una capability non concessa al Run.
    """
    def __init__(self, capability_id: str) -> None:
        super().__init__(
            f"Capability not granted: {capability_id}"
        )
        self.capability_id = capability_id


# ----------------------------------------------------------------------
# Errori di enforcement
# ----------------------------------------------------------------------

class CapabilityExpiredError(CapabilityError):
    """
    Tentativo di usare una capability scaduta.
    """
    def __init__(self, capability_id: str) -> None:
        super().__init__(
            f"Capability expired: {capability_id}"
        )
        self.capability_id = capability_id


class CapabilityRevokedError(CapabilityError):
    """
    Tentativo di usare una capability revocata.
    """
    def __init__(self, capability_id: str) -> None:
        super().__init__(
            f"Capability revoked: {capability_id}"
        )
        self.capability_id = capability_id


class CapabilityScopeViolationError(CapabilityError):
    """
    Uso della capability fuori dallo scope concesso.
    """
    def __init__(self, *, capability_id: str, requested_scope: str) -> None:
        super().__init__(
            f"Capability scope violation: {capability_id} → {requested_scope}"
        )
        self.capability_id = capability_id
        self.requested_scope = requested_scope


class CapabilityRunMismatchError(CapabilityError):
    """
    Capability concessa a un Run diverso da quello corrente.
    """
    def __init__(self, *, expected: str, actual: str) -> None:
        super().__init__(
            f"Capability run mismatch: expected {expected}, got {actual}"
        )
        self.expected = expected
        self.actual = actual


class CapabilityUsageNotAllowedError(CapabilityError):
    """
    Uso della capability in uno stato del Run non ammesso.
    """
    def __init__(self, *, capability_id: str, run_state: str) -> None:
        super().__init__(
            f"Capability usage not allowed: {capability_id} in state {run_state}"
        )
        self.capability_id = capability_id
        self.run_state = run_state
