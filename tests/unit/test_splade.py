"""SPLADE loader pins and evaluation provenance without model downloads."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scipy import sparse

from scinexusrag.eval import datasets, run, systems
from scinexusrag.ingestion import Chunk
from scinexusrag.retrieval.dense import RetrievalResult
from scinexusrag.retrieval.splade import DEFAULT_MODEL, SpladeRetriever

PIN = "49cf4c7b0db5b870a401ddf5e2669993ef3699c7"


@pytest.mark.parametrize("field", ["batch_size", "max_length"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_encoding_parameters_fail_before_loading(monkeypatch, field, value):
    encode = Mock()
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)

    with pytest.raises(ValueError, match="positive"):
        SpladeRetriever([], **{field: value})
    encode.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, -99])
def test_nonpositive_depth_skips_query_encoding(monkeypatch, top_k):
    encode = Mock(return_value=sparse.csr_matrix([[1.0]]))
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    retriever = SpladeRetriever([Chunk(id="doc", content="Evidence", document_id="doc")])

    assert retriever.retrieve("query", top_k=top_k) == []
    assert encode.call_count == 1


@pytest.mark.parametrize("top_k,expected", [(1, [1]), (2, [1, 2]), (99, [1, 2, 3, 0, 4])])
def test_tied_scores_keep_corpus_order_at_depth_boundary(monkeypatch, top_k, expected):
    encode = Mock(
        side_effect=[
            sparse.csr_matrix([[0.3], [0.7], [0.7], [0.7], [0.1]]),
            sparse.csr_matrix([[2.0]]),
        ]
    )
    monkeypatch.setattr(SpladeRetriever, "_encode", encode)
    chunks = [Chunk(id=str(i), content=str(i), document_id=str(i)) for i in range(5)]
    retriever = SpladeRetriever(chunks)
    results = retriever.retrieve("query", top_k=top_k)

    assert [result.chunk.id for result in results] == [str(i) for i in expected]
    assert results[0].score == pytest.approx(1.4)


@pytest.mark.parametrize(
    "model_name,revision,expected",
    [
        (DEFAULT_MODEL, None, PIN),
        (DEFAULT_MODEL, "explicit-pin", "explicit-pin"),
        ("custom/sparse-model", "custom-pin", "custom-pin"),
        ("custom/sparse-model", None, None),
    ],
)
def test_tokenizer_and_model_share_selected_revision(monkeypatch, model_name, revision, expected):
    tokenizer = Mock()
    model = Mock()
    model.to.return_value = model
    tokenizer_loader = Mock(return_value=tokenizer)
    model_loader = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
            AutoModelForMaskedLM=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    monkeypatch.setattr(SpladeRetriever, "_encode", lambda *_: sparse.csr_matrix((0, 0)))
    retriever = SpladeRetriever([], model_name=model_name, revision=revision)
    retriever._load()
    retriever._load()

    assert retriever.revision == expected
    tokenizer_loader.assert_called_once_with(model_name, revision=expected)
    model_loader.assert_called_once_with(model_name, revision=expected)
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()


@pytest.mark.parametrize(
    "model_name,revision,expected_model,expected_revision",
    [
        (None, None, DEFAULT_MODEL, PIN),
        ("custom/sparse-model", "custom-pin", "custom/sparse-model", "custom-pin"),
        ("custom/sparse-model", None, "custom/sparse-model", None),
    ],
)
def test_enabled_eval_records_revision_passed_to_retriever(
    monkeypatch, model_name, revision, expected_model, expected_revision
):
    dataset = datasets.IRDataset(
        name="fixture",
        corpus={"doc": {"title": "", "text": "Scientific evidence."}},
        queries={"query": "Which evidence?"},
        qrels={"query": {"doc": 1}},
    )
    monkeypatch.setattr(datasets, "load", lambda *a, **k: dataset)
    monkeypatch.setattr(systems, "ExactDenseRetriever", lambda *a: object())
    constructed = []

    def fake_splade(chunks: list[Chunk], **kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(retrieve=lambda *a, **k: [RetrievalResult(chunks[0], 1.0, "splade")])

    monkeypatch.setattr("scinexusrag.retrieval.splade.SpladeRetriever", fake_splade)

    def splade_only(*args, **kwargs):
        built = systems.build_systems(*args, **kwargs)
        return {"SPLADE": built["SPLADE"]}

    monkeypatch.setattr(run, "build_systems", splade_only)
    output = run.evaluate(
        use_sample=True,
        include_splade=True,
        splade_model=model_name,
        splade_revision=revision,
    )

    assert constructed == [
        {"model_name": expected_model, "revision": expected_revision, "device": "cpu"}
    ]
    assert output["provenance"]["models"]["splade"] == {
        "model": expected_model,
        "revision": expected_revision,
    }
    assert output["systems"]["SPLADE"]["means"]["nDCG@10"] == 1.0


def test_cli_forwards_splade_model_and_revision(monkeypatch, tmp_path):
    evaluate = Mock(return_value={"dataset": "fixture"})
    monkeypatch.setattr(run, "evaluate", evaluate)
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scinexusrag-eval",
            "--splade",
            "--splade-model",
            "custom/sparse-model",
            "--splade-revision",
            "custom-pin",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )
    run.main()

    assert evaluate.call_args.kwargs["include_splade"] is True
    assert evaluate.call_args.kwargs["splade_model"] == "custom/sparse-model"
    assert evaluate.call_args.kwargs["splade_revision"] == "custom-pin"
