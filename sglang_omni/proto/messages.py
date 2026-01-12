# SPDX-License-Identifier: Apache-2.0
"""Control plane messages."""

from dataclasses import dataclass
from typing import Any

from .data import SHMMetadata


@dataclass
class DataReadyMessage:
    """Notify next stage that data is ready in SHM."""

    request_id: str
    from_stage: str
    to_stage: str
    shm_metadata: SHMMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "data_ready",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "shm_metadata": self.shm_metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataReadyMessage":
        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            to_stage=d["to_stage"],
            shm_metadata=SHMMetadata.from_dict(d["shm_metadata"]),
        )


@dataclass
class AbortMessage:
    """Broadcast abort signal to all stages."""

    request_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "abort", "request_id": self.request_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AbortMessage":
        return cls(request_id=d["request_id"])


@dataclass
class CompleteMessage:
    """Notify coordinator that a request completed (or failed)."""

    request_id: str
    from_stage: str
    success: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "complete",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompleteMessage":
        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            success=d["success"],
            result=d.get("result"),
            error=d.get("error"),
        )


@dataclass
class SubmitMessage:
    """Submit a new request to the entry stage."""

    request_id: str
    data: Any

    def to_dict(self) -> dict[str, Any]:
        return {"type": "submit", "request_id": self.request_id, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmitMessage":
        return cls(request_id=d["request_id"], data=d["data"])


@dataclass
class ShutdownMessage:
    """Signal graceful shutdown to a stage."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "shutdown"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShutdownMessage":
        return cls()


def parse_message(
    d: dict[str, Any],
) -> DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage | ShutdownMessage:
    """Parse a dict into the appropriate message type."""
    msg_type = d.get("type")
    if msg_type == "data_ready":
        return DataReadyMessage.from_dict(d)
    elif msg_type == "abort":
        return AbortMessage.from_dict(d)
    elif msg_type == "complete":
        return CompleteMessage.from_dict(d)
    elif msg_type == "submit":
        return SubmitMessage.from_dict(d)
    elif msg_type == "shutdown":
        return ShutdownMessage.from_dict(d)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
