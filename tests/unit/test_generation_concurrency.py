import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Barrier, Event, Lock, get_ident
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from scinexusrag.generation.grounding import GroundingVerifier
from scinexusrag.generation.llm import LLMClient


class _ObservedLock:
    def __init__(self):
        self.lock = Lock()
        self.attempts = Queue()

    def __enter__(self):
        self.attempts.put(get_ident())
        if not self.lock.acquire(timeout=5):
            raise TimeoutError("Resource lock was not released")
        return self

    def __exit__(self, *_args):
        self.lock.release()


@pytest.mark.parametrize("stage", ["construction", "labels"])
def test_concurrent_first_requests_share_model_and_resolved_labels(monkeypatch, stage):
    verifier = GroundingVerifier(model_name="test-nli", revision="a" * 40)
    lock = _ObservedLock()
    verifier._model_lock = lock
    start = Barrier(4)
    initializing = Event()
    release = Event()

    def pause():
        initializing.set()
        assert release.wait(timeout=5)

    class Config:
        @property
        def id2label(self):
            if stage == "labels":
                pause()
            return {0: "neutral", 1: "contradiction", 2: "entailment"}

    model = SimpleNamespace(model=SimpleNamespace(config=Config()))

    def construct(*_args, **_kwargs):
        if stage == "construction":
            pause()
        return model

    factory = Mock(side_effect=construct)
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    def load():
        start.wait(timeout=5)
        loaded = verifier.model
        return loaded, verifier.entail_idx, verifier.contra_idx

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(load) for _ in range(4)]
        try:
            assert initializing.wait(timeout=5)
            assert len({lock.attempts.get(timeout=5) for _ in range(4)}) == 4
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        assert all(future.result(timeout=5) == (model, 2, 1) for future in futures)

    factory.assert_called_once_with("test-nli", device=None, revision="a" * 40)


def test_failed_grounding_model_load_can_retry(monkeypatch):
    verifier = GroundingVerifier(model_name="test-nli")
    model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(id2label={})))
    factory = Mock(side_effect=[RuntimeError("Load failed"), model])
    module = ModuleType("sentence_transformers")
    module.CrossEncoder = factory
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(RuntimeError, match="Load failed"):
        _ = verifier.model
    assert verifier._model is None

    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: verifier.model).result(timeout=5) is model
    assert verifier.model is model
    assert factory.call_count == 2


def test_concurrent_first_requests_share_http_client(monkeypatch):
    llm = LLMClient()
    lock = _ObservedLock()
    llm._client_lock = lock
    start = Barrier(4)
    constructing = Event()
    release = Event()
    client = Mock()

    def construct(**_kwargs):
        constructing.set()
        assert release.wait(timeout=5)
        return client

    factory = Mock(side_effect=construct)
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    def load():
        start.wait(timeout=5)
        return llm.client

    with ThreadPoolExecutor(max_workers=4) as workers:
        futures = [workers.submit(load) for _ in range(4)]
        try:
            assert constructing.wait(timeout=5)
            assert len({lock.attempts.get(timeout=5) for _ in range(4)}) == 4
        finally:
            release.set()
        assert all(future.result(timeout=5) is client for future in futures)

    assert factory.call_count == 1
    assert factory.call_args.kwargs["base_url"] == llm.base_url
    assert factory.call_args.kwargs["timeout"].read == llm.timeout
    llm.close()
    client.close.assert_called_once()


def test_failed_http_client_construction_can_retry(monkeypatch):
    llm = LLMClient()
    client = Mock()
    factory = Mock(side_effect=[RuntimeError("Client construction failed"), client])
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    with pytest.raises(RuntimeError, match="Client construction failed"):
        _ = llm.client
    assert llm._client is None

    with ThreadPoolExecutor(max_workers=1) as workers:
        assert workers.submit(lambda: llm.client).result(timeout=5) is client
    assert llm.client is client
    assert factory.call_count == 2
    llm.close()


def test_close_waits_for_http_client_construction(monkeypatch):
    llm = LLMClient()
    lock = _ObservedLock()
    llm._client_lock = lock
    constructing = Event()
    release = Event()
    original, fresh = Mock(), Mock()

    def construct(**_kwargs):
        constructing.set()
        assert release.wait(timeout=5)
        return original

    factory = Mock(side_effect=construct)
    monkeypatch.setattr("scinexusrag.generation.llm.httpx.Client", factory)

    with ThreadPoolExecutor(max_workers=2) as workers:
        loading = workers.submit(lambda: llm.client)
        try:
            assert constructing.wait(timeout=5)
            lock.attempts.get(timeout=5)
            closing = workers.submit(llm.close)
            lock.attempts.get(timeout=5)
            assert not closing.done()
        finally:
            release.set()
        assert loading.result(timeout=5) is original
        closing.result(timeout=5)

    original.close.assert_called_once()
    assert llm._client is None
    factory.side_effect = None
    factory.return_value = fresh
    assert llm.client is fresh
    llm.close()
    fresh.close.assert_called_once()
