# src/ice_runtime/runtime/runtime.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ice_runtime.runtime.state import RuntimeState
from ice_runtime.ids.runtime_id import RuntimeId
from ice_runtime.logging.bridge import get_logger


class Runtime:
    """
    ICE Runtime core object.
    Owns lifecycle, state, and global context.
    """

    def __init__(self, *, base_dir: Path):
        self.id = RuntimeId.generate()
        self.base_dir = base_dir
        self.state = RuntimeState.CREATED
        self.logger = get_logger("runtime")

    def start(self) -> None:
        if self.state is not RuntimeState.CREATED:
            raise RuntimeError("Runtime already started or invalid state")

        self.logger.info("runtime.start", extra={"runtime_id": str(self.id)})
        self.state = RuntimeState.RUNNING

    def stop(self) -> None:
        if self.state not in (RuntimeState.RUNNING, RuntimeState.FAILED):
            return

        self.logger.info("runtime.stop", extra={"runtime_id": str(self.id)})
        self.state = RuntimeState.STOPPED

    def status(self) -> dict:
        return {
            "runtime_id": str(self.id),
            "state": self.state.name,
            "base_dir": str(self.base_dir),
        }
