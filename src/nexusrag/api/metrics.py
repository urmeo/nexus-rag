"""Thread-safe operational metrics collector."""

import threading
import time


class MetricsCollector:
    """Collects operational metrics with thread-safe counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._total_queries = 0
        self._total_ingestions = 0
        self._total_deletions = 0
        self._total_query_time_ms = 0.0

    def record_query(self, elapsed_ms: float) -> None:
        """Record a completed query with its duration."""
        with self._lock:
            self._total_queries += 1
            self._total_query_time_ms += elapsed_ms

    def record_ingest(self) -> None:
        """Record a successful ingestion."""
        with self._lock:
            self._total_ingestions += 1

    def record_delete(self) -> None:
        """Record a successful deletion."""
        with self._lock:
            self._total_deletions += 1

    def snapshot(self) -> dict[str, int | float]:
        """Return a point-in-time snapshot of all metrics."""
        with self._lock:
            avg = (
                self._total_query_time_ms / self._total_queries if self._total_queries > 0 else 0.0
            )
            return {
                "uptime_seconds": time.monotonic() - self._start_time,
                "total_queries": self._total_queries,
                "total_ingestions": self._total_ingestions,
                "total_deletions": self._total_deletions,
                "avg_query_time_ms": avg,
            }


_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Return the module-level singleton MetricsCollector."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector
