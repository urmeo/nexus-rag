"""Tests for vendored dataset loaders and sentence splitting."""

from nexusrag.agents.grounding import split_sentences
from nexusrag.eval import datasets as D
from nexusrag.eval import faithfulness as F


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

    def test_prf_perfect(self) -> None:
        rows = [
            {"evidence_strength": 0.9, "is_gold": True},
            {"evidence_strength": 0.1, "is_gold": False},
        ]
        p, r, f1 = F._prf(rows, 0.5)
        assert p == 1.0 and r == 1.0 and f1 == 1.0


class TestSentenceSplit:
    def test_split_basic(self) -> None:
        out = split_sentences("First claim. Second one! Third? Yes.")
        assert len(out) == 4

    def test_split_empty(self) -> None:
        assert split_sentences("") == []
