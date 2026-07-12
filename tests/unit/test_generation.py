from scinexusrag.eval.generation import _format_sources, _mean


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
