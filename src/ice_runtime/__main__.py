# src/ice_runtime/__main__.py
from __future__ import annotations

import time
from pathlib import Path

from ice_runtime.bootstrap.bootstrap import bootstrap_runtime


def main() -> None:
    runtime = bootstrap_runtime(Path.cwd())
    print("ICE Runtime started:", runtime.status())

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        runtime.stop()
        print("ICE Runtime stopped")


if __name__ == "__main__":
    main()
