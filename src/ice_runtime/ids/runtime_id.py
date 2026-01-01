# src/ice_runtime/ids/runtime_id.py
from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeId:
    value: str

    @staticmethod
    def generate() -> "RuntimeId":
        return RuntimeId(f"ice-rt-{secrets.token_hex(4)}")

    def __str__(self) -> str:
        return self.value
