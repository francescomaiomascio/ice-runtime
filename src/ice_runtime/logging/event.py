from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LogEvent:
    """
    Evento di logging strutturato e IMMUTABILE.
    """
    ts: datetime
    level: str
    domain: str
    owner: str
    scope: str
    msg: str
    data: Optional[Dict[str, Any]]
    runtime_id: Optional[str] = None

    @staticmethod
    def now() -> datetime:
        return datetime.utcnow()

    def with_runtime_id(self, runtime_id: str) -> "LogEvent":
        """
        Ritorna una copia dell'evento con runtime_id valorizzato.
        """
        return LogEvent(
            ts=self.ts,
            level=self.level,
            domain=self.domain,
            owner=self.owner,
            scope=self.scope,
            msg=self.msg,
            data=self.data,
            runtime_id=runtime_id,
        )
