"""Sentence grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature
from threading import Lock
from typing import Any, Literal

from scinexusrag.settings import HF_REVISIONS
from scinexusrag.utils.text import split_sentences

__all__ = ["GroundingReport", "GroundingVerifier", "SentenceGrounding", "split_sentences"]


@dataclass
class SentenceGrounding:
    sentence: str
    entailment: float
    source_index: int | None
    grounded: bool


@dataclass
class GroundingReport:
    faithfulness: float
    sentences: list[SentenceGrounding]
    unsupported: list[str] = field(default_factory=list)


class GroundingVerifier:
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        threshold: float = 0.5,
        device: Literal["cpu", "cuda", "mps"] | None = None,
        revision: str | None = None,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("grounding threshold must be between 0 and 1")
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self.revision = revision or HF_REVISIONS.get(model_name)
        self._model: Any = None
        self._model_lock = Lock()
        self.entail_idx = 1
        self.contra_idx = 0

    @property
    def model(self) -> Any:
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.model_name, device=self.device, revision=self.revision)
                id2label = getattr(model.model.config, "id2label", {})
                entail_idx, contra_idx = self.entail_idx, self.contra_idx
                for idx, label in id2label.items():
                    label = str(label).strip().lower()
                    if label.startswith("entail"):
                        entail_idx = int(idx)
                    elif label.startswith("contra"):
                        contra_idx = int(idx)
                self.entail_idx, self.contra_idx = entail_idx, contra_idx
                self._model = model
            return self._model

    def class_probs(self, pairs: list[tuple[str, str]]) -> Any:
        """Class probabilities."""
        import numpy as np

        if not pairs:
            return np.zeros((0, 3))
        predict = self.model.predict
        activation_name = (
            "activation_fct"
            if "activation_fct" in signature(predict).parameters
            else "activation_fn"
        )
        raw_activation = {activation_name: lambda values: values}
        logits = np.asarray(
            predict(
                pairs,
                convert_to_numpy=True,
                apply_softmax=False,
                **raw_activation,
            ),
            dtype=np.float64,
        )
        if logits.ndim == 1:
            logits = logits[:, None]
        if logits.ndim != 2 or logits.shape[0] != len(pairs) or logits.shape[1] == 0:
            raise ValueError("NLI predictions need one row per pair and at least one class")
        if not np.isfinite(logits).all():
            raise ValueError("NLI predictions must be finite")
        with np.errstate(over="ignore", under="ignore"):
            if logits.shape[1] == 1:
                probability = np.exp(-np.logaddexp(0.0, -logits[:, 0]))
                return np.column_stack([1.0 - probability, probability])
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)

    def entailment_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Entailment scores."""
        probs = self.class_probs(pairs)
        if probs.shape[0] == 0:
            return []
        if not 0 <= self.entail_idx < probs.shape[1]:
            raise ValueError("NLI entailment label is outside the prediction classes")
        return [float(x) for x in probs[:, self.entail_idx]]

    def verify(self, answer: str, sources: list[Any]) -> GroundingReport:
        sentences = split_sentences(answer)
        source_texts: list[tuple[int, str]] = []
        for index, source in enumerate(sources):
            if source is None:
                continue
            text = getattr(source, "full_content", "") or getattr(source, "content", str(source))
            if not isinstance(text, str):
                raise ValueError("source text must be a string")
            if text.strip():
                source_texts.append((index, text))

        if not sentences or not source_texts:
            return GroundingReport(faithfulness=0.0, sentences=[], unsupported=sentences)

        graded: list[SentenceGrounding] = []
        unsupported: list[str] = []
        for sentence in sentences:
            pairs = [(text, sentence) for _, text in source_texts]
            scores = self.entailment_scores(pairs)
            best = max(range(len(scores)), key=lambda i: scores[i])
            grounded = scores[best] >= self.threshold
            graded.append(
                SentenceGrounding(
                    sentence=sentence,
                    entailment=scores[best],
                    source_index=source_texts[best][0] if grounded else None,
                    grounded=grounded,
                )
            )
            if not grounded:
                unsupported.append(sentence)

        faithfulness = sum(1 for g in graded if g.grounded) / len(graded)
        return GroundingReport(faithfulness=faithfulness, sentences=graded, unsupported=unsupported)
