"""
ICE Runtime — Capability Errors
===============================

Gerarchia canonica degli errori del Capability System.

Regole:
- Ogni violazione di capability è FATALE per il Run
- Gli errori NON applicano logica
- Gli errori NON emettono eventi
- La semantica è interpretata dal Runtime / Enforcement
"""


class CapabilityError(Exception):
    """
    Base class per tutti gli errori di capability.

    Qualsiasi CapabilityError implica:
    - violazione architetturale
    - abort del Run (a discrezione del Runtime)
    """
    pass


class CapabilityNotRegistered(CapabilityError):
    """
    Tentativo di uso o richiesta di una capability inesistente.
    """
    def __init__(self, capability_name: str) -> None:
        super().__init__(f"Capability not registered: {capability_name}")
        self.capability_name = capability_name


class CapabilityNotGranted(CapabilityError):
    """
    Tentativo di usare una capability non concessa al Run.
    """
    def __init__(self, capability_name: str) -> None:
        super().__init__(f"Capability not granted: {capability_name}")
        self.capability_name = capability_name


class CapabilityExpired(CapabilityError):
    """
    Tentativo di usare una capability scaduta.
    """
    def __init__(self, capability_name: str) -> None:
        super().__init__(f"Capability expired: {capability_name}")
        self.capability_name = capability_name


class CapabilityRevoked(CapabilityError):
    """
    Tentativo di usare una capability revocata dal Runtime.
    """
    def __init__(self, capability_name: str) -> None:
        super().__init__(f"Capability revoked: {capability_name}")
        self.capability_name = capability_name


class CapabilityConstraintViolation(CapabilityError):
    """
    Violazione dei vincoli semantici o quantitativi
    associati a una capability.
    """
    def __init__(self, capability_name: str, reason: str) -> None:
        super().__init__(
            f"Capability constraint violation: {capability_name} — {reason}"
        )
        self.capability_name = capability_name
        self.reason = reason


class CapabilityUsageNotAllowed(CapabilityError):
    """
    Uso di una capability in uno stato del Run non ammesso
    (es. VALIDATING, TERMINATED).
    """
    def __init__(self, capability_name: str, run_state: str) -> None:
        super().__init__(
            f"Capability usage not allowed: {capability_name} in state {run_state}"
        )
        self.capability_name = capability_name
        self.run_state = run_state
