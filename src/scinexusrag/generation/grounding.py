"""Sentence grounding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal

from scinexusrag.generation.synthesizer import Source
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
        """Softmax over raw NLI logits; a single-logit model gets a sigmoid instead."""
        import numpy as np

        if not pairs:
            return np.zeros((0, 3))
        logits = np.asarray(
            self.model.predict(
                pairs,
                convert_to_numpy=True,
                apply_softmax=False,
                activation_fn=lambda values: values,
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
        probs = self.class_probs(pairs)
        if probs.shape[0] == 0:
            return []
        if not 0 <= self.entail_idx < probs.shape[1]:
            raise ValueError("NLI entailment label is outside the prediction classes")
        return [float(x) for x in probs[:, self.entail_idx]]

    def verify(self, answer: str, sources: Sequence[Source | str]) -> GroundingReport:
        """Grade every answer sentence against every non-blank source in one NLI batch."""
        sentences = split_sentences(answer)
        texts = [(index, _source_text(source)) for index, source in enumerate(sources)]
        texts = [(index, text) for index, text in texts if text.strip()]
        if not sentences or not texts:
            return GroundingReport(faithfulness=0.0, sentences=[], unsupported=sentences)

        scores = self.entailment_scores(
            [(text, sentence) for sentence in sentences for _, text in texts]
        )
        graded: list[SentenceGrounding] = []
        for row, sentence in enumerate(sentences):
            row_scores = scores[row * len(texts) : (row + 1) * len(texts)]
            best = max(range(len(texts)), key=row_scores.__getitem__)
            grounded = row_scores[best] >= self.threshold
            graded.append(
                SentenceGrounding(
                    sentence=sentence,
                    entailment=row_scores[best],
                    source_index=texts[best][0] if grounded else None,
                    grounded=grounded,
                )
            )

        unsupported = [grade.sentence for grade in graded if not grade.grounded]
        faithfulness = sum(1 for grade in graded if grade.grounded) / len(graded)
        return GroundingReport(faithfulness=faithfulness, sentences=graded, unsupported=unsupported)


def _source_text(source: Source | str) -> str:
    return source if isinstance(source, str) else source.full_content or source.content
