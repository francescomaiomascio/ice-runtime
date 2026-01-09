"""
ICE Runtime — Memory Promotion
==============================

Questo modulo governa l'UNICO meccanismo con cui una memoria nasce.

Principio fondante:
- La memoria NON viene scritta
- La memoria NON viene mutata
- La memoria viene PROMOSSA da eventi validati

Questo modulo:
- verifica se un insieme di eventi è promotabile
- costruisce un MemoryRecord immutabile
- NON persiste nulla
- NON emette eventi

L'emissione di `MemoryPromoted` è responsabilità del Runtime.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from ice_runtime.events.kernel.event import ICEEvent
from ice_runtime.events.kernel.taxonomy import EventCategory, category_of
from ice_runtime.memory.errors import (
    MemoryPromotionError,
    NonPromotableEventError,
)
from ice_runtime.memory.registry import MemoryRecord
from ice_runtime.ids.runtime_id import MemoryID


# =====================================================
# PROMOTION REQUEST (STRUCT)
# =====================================================

class MemoryPromotionRequest:
    """
    Richiesta formale di promozione a memoria.

    È una STRUCT.
    Non contiene logica.
    """

    def __init__(
        self,
        *,
        source_events: Sequence[ICEEvent],
        memory_type: str,
        confidence: float,
        lifecycle_policy: dict,
        access_policy: dict,
        schema_version: str,
        created_at: datetime | None = None,
    ) -> None:
        self.source_events = tuple(source_events)
        self.memory_type = memory_type
        self.confidence = confidence
        self.lifecycle_policy = lifecycle_policy
        self.access_policy = access_policy
        self.schema_version = schema_version
        self.created_at = created_at or datetime.utcnow()


# =====================================================
# PROMOTION SERVICE (PURE)
# =====================================================

class MemoryPromotionService:
    """
    Servizio puro di promozione memoria.

    NON:
    - persiste
    - emette eventi
    - muta stato globale

    Costruisce SOLO un MemoryRecord valido.
    """

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    @staticmethod
    def validate_events(events: Iterable[ICEEvent]) -> None:
        """
        Verifica che TUTTI gli eventi siano promotabili.
        """

        events = list(events)
        if not events:
            raise MemoryPromotionError("no source events provided")

        for event in events:
            category = category_of(event.event_type)

            if category is not EventCategory.DOMAIN:
                raise NonPromotableEventError(
                    event_id=event.event_id,
                    reason="only DOMAIN events can be promoted to memory",
                )

    # -------------------------------------------------
    # PROMOTION
    # -------------------------------------------------

    @staticmethod
    def promote(request: MemoryPromotionRequest) -> MemoryRecord:
        """
        Costruisce una MemoryRecord immutabile.

        Se questa funzione ritorna, la memoria:
        - è strutturalmente valida
        - è causalmente tracciabile
        - è pronta per essere registrata dal Runtime
        """

        MemoryPromotionService.validate_events(request.source_events)

        memory_id = MemoryID.generate()

        return MemoryRecord(
            memory_id=memory_id,
            memory_type=request.memory_type,
            payload={},  # payload semantico deciso a monte
            confidence=request.confidence,
            lifecycle_policy=request.lifecycle_policy,
            access_policy=request.access_policy,
            source_events=[e.event_id for e in request.source_events],
            schema_version=request.schema_version,
        )
