"""Tests for the corrective-generation eval logic (no model loads)."""

from nexusrag.eval.generation import _format_sources, _mean, _reformulate


class TestReformulate:
    def test_drops_stopwords(self) -> None:
        out = _reformulate("What is the role of BRCA1 in cancer?")
        assert "what" not in out and "the" not in out
        assert "brca1" in out and "cancer" in out

    def test_empty_or_all_stopwords_falls_back(self) -> None:
        assert _reformulate("the of a") == "the of a"


class TestFormatSources:
    def test_numbered_and_truncated(self) -> None:
        long = "x" * 600
        out = _format_sources(["first passage", long])
        assert out.startswith("[1] first passage")
        assert "[2]" in out
        assert len(out.splitlines()) == 2
        # second source truncated to 400 chars
        assert out.splitlines()[1] == "[2] " + "x" * 400


class TestMean:
    def test_mean(self) -> None:
        assert abs(_mean([0.2, 0.4, 0.6]) - 0.4) < 1e-9

    def test_empty(self) -> None:
        assert _mean([]) == 0.0
