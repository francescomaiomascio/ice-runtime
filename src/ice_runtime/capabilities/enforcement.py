"""
ICE Runtime — Capability Enforcement
====================================

Questo modulo è l'ENFORCER operativo delle capability.

Responsabilità ESCLUSIVE:
- verificare che una capability concessa possa essere usata ORA
- applicare vincoli temporali, di scope e di Run
- NEGARE l'uso in caso di qualsiasi violazione

Questo modulo:
- NON concede capability
- NON registra capability
- NON muta stato
- NON emette eventi (lo fa il Runtime)

Qui si FALLISCE o si PASSA.
"""

from datetime import datetime
from typing import Optional

from ice_runtime.capabilities.grants import CapabilityGrant
from ice_runtime.capabilities.errors import (
    CapabilityExpiredError,
    CapabilityRevokedError,
    CapabilityScopeViolationError,
    CapabilityRunMismatchError,
)


class CapabilityEnforcer:
    """
    Enforcer puro e stateless.

    Ogni chiamata è indipendente.
    """

    @staticmethod
    def enforce(
        *,
        grant: CapabilityGrant,
        current_run_id: str,
        requested_scope: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> None:
        """
        Verifica che una capability possa essere usata.

        Se la verifica FALLISCE → eccezione.
        Se PASSA → ritorna None.

        PARAMETRI:
        - grant: CapabilityGrant concessa dal Runtime
        - current_run_id: Run attualmente in esecuzione
        - requested_scope: scope richiesto dall'azione (opzionale)
        - now: timestamp corrente (iniettato dal Runtime)
        """

        # -------------------------
        # 1. Run ownership
        # -------------------------
        if grant.run_id != current_run_id:
            raise CapabilityRunMismatchError(
                expected=grant.run_id,
                actual=current_run_id,
            )

        # -------------------------
        # 2. Revoca esplicita
        # -------------------------
        if grant.revoked:
            raise CapabilityRevokedError(grant.capability_id)

        # -------------------------
        # 3. Scadenza temporale
        # -------------------------
        if grant.expires_at is not None:
            now_ts = now or datetime.utcnow()
            if now_ts >= grant.expires_at:
                raise CapabilityExpiredError(grant.capability_id)

        # -------------------------
        # 4. Scope enforcement
        # -------------------------
        if requested_scope is not None:
            if not grant.allows_scope(requested_scope):
                raise CapabilityScopeViolationError(
                    capability_id=grant.capability_id,
                    requested_scope=requested_scope,
                )

        # -------------------------
        # 5. SUCCESSO
        # -------------------------
        return None
