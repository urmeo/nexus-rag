"""Tests for MetricsCollector."""

import threading

from nexusrag.api.metrics import MetricsCollector


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    def test_initial_state(self) -> None:
        """Test that a new collector starts with zero counters."""
        collector = MetricsCollector()
        snap = collector.snapshot()

        assert snap["total_queries"] == 0
        assert snap["total_ingestions"] == 0
        assert snap["total_deletions"] == 0
        assert snap["avg_query_time_ms"] == 0.0
        assert snap["uptime_seconds"] >= 0

    def test_record_query(self) -> None:
        """Test recording queries updates count and avg time."""
        collector = MetricsCollector()
        collector.record_query(100.0)
        collector.record_query(200.0)

        snap = collector.snapshot()
        assert snap["total_queries"] == 2
        assert snap["avg_query_time_ms"] == 150.0

    def test_record_ingest(self) -> None:
        """Test recording ingestions updates count."""
        collector = MetricsCollector()
        collector.record_ingest()
        collector.record_ingest()
        collector.record_ingest()

        snap = collector.snapshot()
        assert snap["total_ingestions"] == 3

    def test_record_delete(self) -> None:
        """Test recording deletions updates count."""
        collector = MetricsCollector()
        collector.record_delete()

        snap = collector.snapshot()
        assert snap["total_deletions"] == 1

    def test_avg_query_time_with_no_queries(self) -> None:
        """Test avg query time is 0 when no queries recorded."""
        collector = MetricsCollector()
        assert collector.snapshot()["avg_query_time_ms"] == 0.0

    def test_thread_safety(self) -> None:
        """Test that concurrent recording does not lose counts."""
        collector = MetricsCollector()
        iterations = 1000

        def record_queries() -> None:
            for _ in range(iterations):
                collector.record_query(1.0)

        def record_ingests() -> None:
            for _ in range(iterations):
                collector.record_ingest()

        t1 = threading.Thread(target=record_queries)
        t2 = threading.Thread(target=record_ingests)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        snap = collector.snapshot()
        assert snap["total_queries"] == iterations
        assert snap["total_ingestions"] == iterations
