# src/ice_runtime/bootstrap/bootstrap.py
from __future__ import annotations

import os
from pathlib import Path

from ice_runtime.runtime.runtime import Runtime
from ice_runtime.logging.runtime import (
    RuntimeContext,
    init_runtime_context,
)
from ice_runtime.logging.router import LogRouter
from ice_runtime.logging.api import init as init_logging
from ice_runtime.logging.sinks.stdout import StdoutSink



def bootstrap_runtime(base_dir: Path | None = None) -> Runtime:
    base_dir = base_dir or Path.cwd()

    # Logging context
    ctx = RuntimeContext(
        runtime_id="runtime",
        base_dir=base_dir,
    )

    router = LogRouter(ctx, sinks=[StdoutSink()])

    init_logging(router)

    runtime = Runtime(base_dir=base_dir)
    runtime.start()
    return runtime
