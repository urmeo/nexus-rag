"""Tests for vendored dataset loaders and sentence splitting."""

from nexusrag.eval import datasets as D
from nexusrag.eval import faithfulness as F
from nexusrag.generation.grounding import split_sentences


class TestVendoredIR:
    def test_load_scifact_sample(self) -> None:
        ds = D.load("scifact", prefer_vendored=True)
        assert ds.corpus and ds.queries and ds.qrels
        qid = next(iter(ds.qrels))
        assert ds.qrels[qid]  # has at least one relevant doc
        assert all(d in ds.corpus for rel in ds.qrels.values() for d in rel)

    def test_doc_text_joins_title_body(self) -> None:
        ds = D.load("scifact", prefer_vendored=True)
        doc_id = next(iter(ds.corpus))
        text = ds.doc_text(doc_id)
        assert isinstance(text, str) and len(text) > 0


class TestClaims:
    def test_load_claims_have_gold(self) -> None:
        claims = F.load_claims("dev", prefer_vendored=True)
        assert claims
        c = claims[0]
        assert c.gold and c.candidates
        assert c.label in {"SUPPORT", "CONTRADICT"}

    def test_lexical_overlap_is_jaccard(self) -> None:
        assert F.lexical_overlap("kinase inhibits tumor", "tumor kinase growth") == 0.5
        assert F.lexical_overlap("abc def", "xyz wuv") == 0.0

    def test_f1_at_perfect_separation(self) -> None:
        assert F._f1_at([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 0.5) == 1.0

    def test_bootstrap_auroc_brackets_perfect(self) -> None:
        lo, hi = F._bootstrap_auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], [0, 0, 1, 1], n_boot=200)
        assert 0.5 <= lo <= 1.0 and hi == 1.0


class TestSentenceSplit:
    def test_split_basic(self) -> None:
        out = split_sentences("First claim. Second one! Third? Yes.")
        assert len(out) == 4

    def test_split_empty(self) -> None:
        assert split_sentences("") == []

    def test_split_before_digit(self) -> None:
        # A sentence starting with a digit is a boundary (shared rule).
        assert len(split_sentences("We ran the test. 3 trials passed cleanly.")) == 2

    def test_grounding_and_chunker_share_one_boundary(self) -> None:
        # Both modules must resolve to the single canonical boundary.
        from nexusrag.generation import grounding
        from nexusrag.ingestion import chunker
        from nexusrag.utils.text import SENTENCE_BOUNDARY

        assert grounding.split_sentences.__module__ == "nexusrag.utils.text"
        assert chunker.SENTENCE_BOUNDARY is SENTENCE_BOUNDARY
