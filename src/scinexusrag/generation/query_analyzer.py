"""Query analysis."""

import re
from enum import Enum


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


QUERY_PATTERNS = {
    QueryType.COMPARISON: [
        r"compare",
        r"difference between",
        r"\b(?:versus|vs)\b\.?",
        r"better than",
    ],
    QueryType.SUMMARY: [r"^summar", r"^overview", r"main (points?|ideas?|findings?)"],
    QueryType.METHODOLOGY: [r"^how (did|do|does)", r"method(ology)?", r"approach", r"technique"],
    QueryType.DEFINITION: [r"^define", r"^what does .+ mean", r"definition of"],
    QueryType.LIST: [r"^list", r"^enumerate", r"^what are (the|all)"],
    QueryType.CAUSAL: [r"^why", r"\bcause", r"\breasons?\b"],
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
    def classify(self, query: str) -> QueryType:
        normalized = re.sub(r"\s+", " ", query.lower().strip()).rstrip("?").strip()
        for qtype, patterns in QUERY_PATTERNS.items():
            if any(re.search(p, normalized) for p in patterns):
                return qtype
        return QueryType.UNKNOWN

    def rewrite_vague_query(self, query: str) -> str:
        low = re.sub(r"\s+", " ", query.lower().strip()).rstrip("?.").strip()
        return VAGUE_REWRITES.get(low, query)
