from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ice_runtime.logging import get_logger


class HttpLogger:
    """
    Logger HTTP runtime-level.
    Usabile da backend, ws, api, ecc.
    """

    def __init__(
        self,
        *,
        owner: str = "http",
        scope: str = "request",
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id
        self.logger = get_logger(
            domain="icenet",
            owner=owner,
            scope=scope,
        )

    def log(
        self,
        *,
        method: Optional[str] = None,
        path: Optional[str] = None,
        status: Optional[int] = None,
        duration_ms: Optional[int] = None,
        transport: str = "rest",
        service: str = "runtime",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "layer": "http",
            "service": service,
            "transport": transport,
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": duration_ms,
        }

        if self.session_id:
            payload["session_id"] = self.session_id

        if extra:
            payload.update(extra)

        self.logger.info("http_request", data=payload)

    def close(self) -> None:
        return None
