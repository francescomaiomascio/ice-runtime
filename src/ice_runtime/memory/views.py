"""
ICE Runtime — Memory Views
=========================

Espone VISTE DERIVATE e FILTRATE delle memorie agli agenti.

Principi:
- read-only
- policy-driven (decisioni già prese a monte)
- event-safe
- non omnisciente

Questo modulo:
- NON decide accessi
- NON applica enforcement
- NON muta stato
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, List

from ice_runtime.memory.registry import MemoryRegistry, MemoryRecord
from ice_runtime.memory.lifecycle import MemoryLifecycleManager


# =====================================================
# MEMORY VIEW (IMMUTABLE)
# =====================================================

@dataclass(frozen=True)
class MemoryView:
    """
    Vista immutabile di una memoria.

    NON È una memoria.
    NON È persistente.
    NON È autoritativa.
    """

    memory_id: str
    memory_type: str
    payload: Dict[str, Any]
    confidence: float
    source_events: List[str]


# =====================================================
# VIEW BUILDER (PURE)
# =====================================================

class MemoryViewBuilder:
    """
    Costruisce viste di memoria per UN Run / UN Agente.

    Assunzione forte:
    - accesso già autorizzato dal Runtime
    - lifecycle già verificato
    """

    def __init__(
        self,
        *,
        registry: MemoryRegistry,
        lifecycle: MemoryLifecycleManager,
    ) -> None:
        self._registry = registry
        self._lifecycle = lifecycle

    # -------------------------------------------------
    # PUBLIC API (Runtime-only)
    # -------------------------------------------------

    def build_views(
        self,
        *,
        memory_ids: Iterable[str],
        max_items: int | None = None,
    ) -> List[MemoryView]:
        """
        Costruisce una lista di MemoryView ordinate come richiesto.

        - Nessun enforcement
        - Nessuna eccezione di sicurezza
        - Le memorie non attive vengono ignorate
        """

        views: List[MemoryView] = []

        for memory_id in memory_ids:
            if max_items is not None and len(views) >= max_items:
                break

            if not self._lifecycle.is_active(memory_id):
                continue

            record = self._registry.get(memory_id)
            views.append(self._build_single(record))

        return views

    # -------------------------------------------------
    # INTERNAL
    # -------------------------------------------------

    @staticmethod
    def _build_single(record: MemoryRecord) -> MemoryView:
        """
        Costruisce una vista read-only da una MemoryRecord.
        """

        return MemoryView(
            memory_id=record.memory_id,
            memory_type=record.memory_type,
            payload=dict(record.payload),   # copia difensiva
            confidence=record.confidence,
            source_events=list(record.source_events),
        )
