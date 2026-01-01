# src/ice_runtime/runtime/state.py
from enum import Enum, auto


class RuntimeState(Enum):
    CREATED = auto()
    BOOTSTRAPPING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
