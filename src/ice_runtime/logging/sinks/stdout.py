# src/ice_runtime/logging/sinks/stdout.py
from __future__ import annotations

from ice_runtime.logging.event import LogEvent


class StdoutSink:
    """
    Sink minimale: stampa eventi di log su stdout.
    Usato dal runtime standalone.
    """

    def emit(self, event: LogEvent) -> None:
        ts = event.timestamp.isoformat()
        level = event.level
        logger = event.logger
        msg = event.message

        print(f"[{ts}] [{level}] [{logger}] {msg}")
