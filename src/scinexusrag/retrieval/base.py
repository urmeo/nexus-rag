"""Retrieval results and the interface every retriever implements."""

from dataclasses import dataclass
from typing import Protocol

from scinexusrag.models import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str = "dense"


class Retriever(Protocol):
    """Anything that ranks passages for a query."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]: ...
