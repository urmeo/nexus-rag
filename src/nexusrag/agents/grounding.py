"""Sentence-level grounding via natural language inference."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(0-9])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter."""
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 2]


@dataclass
class SentenceGrounding:
    """One answer sentence and its best supporting source."""

    sentence: str
    entailment: float
    source_index: int | None
    grounded: bool


@dataclass
class GroundingReport:
    """Faithfulness of an answer against its sources."""

    faithfulness: float
    sentences: list[SentenceGrounding]
    unsupported: list[str] = field(default_factory=list)


class GroundingVerifier:
    """Checks whether each answer sentence is entailed by a source.

    Uses a cross-encoder NLI model. A sentence is grounded when some
    source entails it above ``threshold``. Unlike a citation-index check,
    this inspects the actual semantic content.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        threshold: float = 0.5,
        device: Literal["cpu", "cuda", "mps"] | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._model: Any = None
        self.entail_idx = 1
        self.contra_idx = 0

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            id2label = getattr(self._model.model.config, "id2label", {})
            for idx, label in id2label.items():
                label = str(label).lower()
                if label.startswith("entail"):
                    self.entail_idx = int(idx)
                elif label.startswith("contra"):
                    self.contra_idx = int(idx)
        return self._model

    def class_probs(self, pairs: list[tuple[str, str]]) -> Any:
        """Softmax NLI probabilities, shape (N, num_classes)."""
        import numpy as np

        if not pairs:
            return np.zeros((0, 3))
        logits = self.model.predict(pairs, convert_to_numpy=True)
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def entailment_scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis)."""
        probs = self.class_probs(pairs)
        if probs.shape[0] == 0:
            return []
        idx = min(self.entail_idx, probs.shape[1] - 1)
        return [float(x) for x in probs[:, idx]]

    def verify(self, answer: str, sources: list[Any]) -> GroundingReport:
        """Score every answer sentence against the source texts."""
        sentences = split_sentences(answer)
        source_texts = [
            getattr(s, "full_content", "") or getattr(s, "content", str(s)) for s in sources
        ]

        if not sentences or not source_texts:
            return GroundingReport(faithfulness=0.0, sentences=[], unsupported=sentences)

        graded: list[SentenceGrounding] = []
        unsupported: list[str] = []
        for sent in sentences:
            pairs = [(src, sent) for src in source_texts]
            scores = self.entailment_scores(pairs)
            best = max(range(len(scores)), key=lambda i: scores[i])
            grounded = scores[best] >= self.threshold
            graded.append(
                SentenceGrounding(
                    sentence=sent,
                    entailment=scores[best],
                    source_index=best if grounded else None,
                    grounded=grounded,
                )
            )
            if not grounded:
                unsupported.append(sent)

        faithfulness = sum(1 for g in graded if g.grounded) / len(graded)
        return GroundingReport(faithfulness=faithfulness, sentences=graded, unsupported=unsupported)
