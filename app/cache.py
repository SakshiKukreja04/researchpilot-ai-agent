"""Small bounded in-memory TTL cache for successful research tool results."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    """Thread-safe, bounded cache that stores successful results only."""

    def __init__(self, max_entries: int = 128, ttl_seconds: float = 300.0) -> None:
        self.max_entries = max(0, max_entries)
        self.ttl_seconds = max(0.0, ttl_seconds)
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = Lock()

    def get(self, key: str) -> T | None:
        if self.max_entries == 0 or self.ttl_seconds == 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.value

    def set(self, key: str, value: T) -> None:
        if self.max_entries == 0 or self.ttl_seconds == 0:
            return
        with self._lock:
            if len(self._entries) >= self.max_entries and key not in self._entries:
                oldest_key = min(
                    self._entries,
                    key=lambda item: self._entries[item].expires_at,
                )
                self._entries.pop(oldest_key, None)
            self._entries[key] = _Entry(
                value=value,
                expires_at=time.monotonic() + self.ttl_seconds,
            )

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
