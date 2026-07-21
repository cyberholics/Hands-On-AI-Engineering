"""
Run logging.

Every agent step and every tool call is recorded with a type, timestamp,
duration and metadata. The Streamlit app renders these as a table, which is
what turns the system from a black box into something you can debug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    seq: int
    event_type: str
    timestamp: float
    elapsed_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {
            "#": self.seq,
            "event": self.event_type,
            "time": time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "ms": self.elapsed_ms,
        }
        for key, value in self.metadata.items():
            row[key] = value
        return row


class RunLogger:
    """Collects an ordered event trace for a single run."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._started = time.time()
        self._last = self._started

    def log(self, event_type: str, **metadata: Any) -> Event:
        now = time.time()
        event = Event(
            seq=len(self.events) + 1,
            event_type=event_type,
            timestamp=now,
            elapsed_ms=int((now - self._last) * 1000),
            metadata=metadata,
        )
        self.events.append(event)
        self._last = now
        return event

    @property
    def total_ms(self) -> int:
        return int((time.time() - self._started) * 1000)

    def rows(self) -> list[dict]:
        return [e.to_row() for e in self.events]

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return {
            "events": len(self.events),
            "total_ms": self.total_ms,
            "by_type": counts,
        }

    def reset(self) -> None:
        self.events.clear()
        self._started = time.time()
        self._last = self._started


class timed:
    """Context manager that logs one event with an accurate duration."""

    def __init__(self, logger: RunLogger, event_type: str, **metadata: Any) -> None:
        self.logger = logger
        self.event_type = event_type
        self.metadata = metadata
        self._start = 0.0

    def __enter__(self) -> "timed":
        self._start = time.time()
        return self

    def add(self, **metadata: Any) -> None:
        """Attach metadata discovered during the block."""
        self.metadata.update(metadata)

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration = int((time.time() - self._start) * 1000)
        if exc is not None:
            self.metadata["error"] = str(exc)[:200]
        event = self.logger.log(self.event_type, **self.metadata)
        event.elapsed_ms = duration
        return False
