"""Lightweight query classification and cleanup."""

import re
from dataclasses import dataclass
from enum import Enum

from nexusrag.retrieval.stopwords import STOP_WORDS


class QueryType(Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DEFINITION = "definition"
    LIST = "list"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


@dataclass
class AnalyzedQuery:
    original: str
    normalized: str
    query_type: QueryType
    keywords: list[str]


QUERY_PATTERNS = {
    QueryType.COMPARISON: [r"compare", r"difference between", r"versus|vs\.?", r"better than"],
    QueryType.SUMMARY: [r"^summar", r"^overview", r"main (points?|ideas?|findings?)"],
    QueryType.METHODOLOGY: [r"^how (did|do|does)", r"method(ology)?", r"approach", r"technique"],
    QueryType.DEFINITION: [r"^define", r"^what does .+ mean", r"definition of"],
    QueryType.LIST: [r"^list", r"^enumerate", r"^what are (the|all)"],
    QueryType.CAUSAL: [r"^why", r"\bcause", r"\breason"],
    QueryType.TEMPORAL: [r"^when", r"timeline", r"\byear\b"],
    QueryType.RESULTS: [r"results?", r"findings?", r"accuracy", r"performance"],
    QueryType.FACTUAL: [r"^what (is|are|was|were)", r"^who (is|are)", r"^which"],
}

VAGUE_REWRITES = {
    "tell me about the paper": "What is the main topic, methodology, and key findings of this paper?",
    "tell me about this paper": "What is the main topic, methodology, and key findings of this paper?",
    "what is this about": "What is the main topic and key findings of this document?",
    "summarize": "What are the main findings and conclusions?",
    "summary": "What are the main findings and conclusions?",
    "what does it say": "What are the key points and findings in this document?",
}


class QueryAnalyzer:
    def analyze(self, query: str) -> AnalyzedQuery:
        normalized = re.sub(r"\s+", " ", query.lower().strip()).rstrip("?").strip()
        return AnalyzedQuery(
            original=query,
            normalized=normalized,
            query_type=self._classify(normalized),
            keywords=self._keywords(normalized),
        )

    def _classify(self, query: str) -> QueryType:
        for qtype, patterns in QUERY_PATTERNS.items():
            if any(re.search(p, query) for p in patterns):
                return qtype
        return QueryType.UNKNOWN

    def _keywords(self, query: str) -> list[str]:
        out: list[str] = []
        for w in re.findall(r"\b[a-z][a-z0-9]+\b", query):
            if w not in STOP_WORDS and len(w) > 2 and w not in out:
                out.append(w)
        return out

    def rewrite_vague_query(self, query: str) -> str:
        low = query.lower().strip()
        for vague, specific in VAGUE_REWRITES.items():
            if low == vague or low.startswith(vague):
                return specific
        return query
