"""Operational metrics."""

import threading
import time
from sys import float_info
from typing import TypedDict


class MetricsSnapshot(TypedDict):
    uptime_seconds: float
    total_queries: int
    total_ingestions: int
    total_deletions: int
    avg_query_time_ms: float


class MetricsCollector:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._total_queries = 0
        self._total_ingestions = 0
        self._total_deletions = 0
        self._average_query_time_ms = 0.0

    def record_query(self, elapsed_ms: float) -> None:
        if isinstance(elapsed_ms, bool) or not isinstance(elapsed_ms, (int, float)):
            raise TypeError("Query duration must be a number")
        if not 0 <= elapsed_ms <= float_info.max:
            raise ValueError("Query duration must be finite and nonnegative")
        with self._lock:
            query_count = self._total_queries + 1
            average = (
                self._average_query_time_ms
                + (elapsed_ms - self._average_query_time_ms) / query_count
            )
            self._average_query_time_ms = average
            self._total_queries = query_count

    def record_ingest(self) -> None:
        with self._lock:
            self._total_ingestions += 1

    def record_delete(self) -> None:
        with self._lock:
            self._total_deletions += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            return {
                "uptime_seconds": time.monotonic() - self._start_time,
                "total_queries": self._total_queries,
                "total_ingestions": self._total_ingestions,
                "total_deletions": self._total_deletions,
                "avg_query_time_ms": self._average_query_time_ms,
            }


_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = MetricsCollector()
        return _collector
