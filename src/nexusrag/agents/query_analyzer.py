"""Query analysis and expansion for improved retrieval."""

import logging
import re
from dataclasses import dataclass
from enum import Enum

from nexusrag.retrieval.stopwords import STOP_WORDS

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types."""

    FACTUAL = "factual"  # "What is X?"
    COMPARISON = "comparison"  # "Compare X and Y"
    SUMMARY = "summary"  # "Summarize..."
    METHODOLOGY = "methodology"  # "How did they...?"
    RESULTS = "results"  # "What were the findings?"
    DEFINITION = "definition"  # "Define X"
    LIST = "list"  # "List the..."
    CAUSAL = "causal"  # "Why did...?"
    TEMPORAL = "temporal"  # "When did...?"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Query complexity level."""

    SIMPLE = "simple"  # Single fact needed
    MODERATE = "moderate"  # Multiple facts from one section
    COMPLEX = "complex"  # Multiple facts from multiple sections


@dataclass
class AnalyzedQuery:
    """Result of query analysis."""

    original: str
    normalized: str
    query_type: QueryType
    complexity: QueryComplexity
    keywords: list[str]
    expanded_terms: list[str]
    search_queries: list[str]  # Multiple queries for retrieval


# Common synonyms for query expansion
SYNONYMS = {
    "method": ["methodology", "approach", "technique", "procedure"],
    "result": ["finding", "outcome", "conclusion", "discovery"],
    "show": ["demonstrate", "indicate", "reveal", "prove"],
    "use": ["utilize", "employ", "apply"],
    "study": ["research", "investigation", "analysis", "experiment"],
    "data": ["dataset", "information", "evidence"],
    "improve": ["enhance", "increase", "boost", "optimize"],
    "problem": ["issue", "challenge", "limitation"],
    "model": ["algorithm", "system", "architecture", "network"],
    "performance": ["accuracy", "efficiency", "effectiveness"],
    "train": ["training", "learning", "fine-tune"],
    "evaluate": ["evaluation", "assess", "measure", "test"],
}

# Query type patterns
QUERY_PATTERNS = {
    QueryType.FACTUAL: [
        r"^what (is|are|was|were)",
        r"^who (is|are|was|were)",
        r"^which",
    ],
    QueryType.COMPARISON: [
        r"compare",
        r"difference between",
        r"versus|vs\.?",
        r"better than",
    ],
    QueryType.SUMMARY: [
        r"^summarize",
        r"^summary",
        r"^overview",
        r"^main (points?|ideas?|findings?)",
    ],
    QueryType.METHODOLOGY: [
        r"^how (did|do|does|was|were)",
        r"method(ology)?",
        r"approach",
        r"technique",
        r"procedure",
    ],
    QueryType.RESULTS: [
        r"results?",
        r"findings?",
        r"outcome",
        r"conclusion",
        r"achieve",
        r"accuracy",
        r"performance",
    ],
    QueryType.DEFINITION: [
        r"^define",
        r"^what does .+ mean",
        r"definition of",
    ],
    QueryType.LIST: [
        r"^list",
        r"^enumerate",
        r"^what are (the|all)",
    ],
    QueryType.CAUSAL: [
        r"^why",
        r"cause",
        r"reason",
        r"because",
    ],
    QueryType.TEMPORAL: [
        r"^when",
        r"timeline",
        r"date",
        r"year",
    ],
}


class QueryAnalyzer:
    """Analyzes and expands queries for better retrieval."""

    def __init__(self, llm: object = None) -> None:
        """
        Initialize analyzer.

        Args:
            llm: Optional LLM client for advanced query expansion
        """
        self.llm = llm
        logger.info("Initialized QueryAnalyzer")

    def analyze(self, query: str) -> AnalyzedQuery:
        """
        Analyze a query to extract type, keywords, and expansions.

        Args:
            query: User's question

        Returns:
            AnalyzedQuery with analysis results
        """
        logger.info(f"Analyzing query: {query[:50]}...")

        # Normalize query
        normalized = self._normalize(query)

        # Classify query type
        query_type = self._classify_type(normalized)

        # Determine complexity
        complexity = self._assess_complexity(normalized)

        # Extract keywords
        keywords = self._extract_keywords(normalized)

        # Expand with synonyms
        expanded = self._expand_terms(keywords)

        # Generate search queries
        search_queries = self._generate_search_queries(normalized, keywords, expanded, query_type)

        result = AnalyzedQuery(
            original=query,
            normalized=normalized,
            query_type=query_type,
            complexity=complexity,
            keywords=keywords,
            expanded_terms=expanded,
            search_queries=search_queries,
        )

        logger.info(
            f"Query analysis: type={query_type.value}, "
            f"complexity={complexity.value}, keywords={keywords[:5]}"
        )

        return result

    def _normalize(self, query: str) -> str:
        """Normalize query text."""
        # Convert to lowercase
        text = query.lower().strip()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove trailing punctuation if it's a question mark (keep others)
        text = re.sub(r"\?+$", "", text).strip()
        return text

    def _classify_type(self, query: str) -> QueryType:
        """Classify the query type based on patterns."""
        for query_type, patterns in QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return query_type
        return QueryType.UNKNOWN

    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity."""
        words = query.split()
        word_count = len(words)

        # Check for complexity indicators
        has_multiple_questions = query.count("?") > 1 or " and " in query
        has_comparison = any(w in query for w in ["compare", "versus", "vs", "difference"])

        if word_count <= 5 and not has_multiple_questions:
            return QueryComplexity.SIMPLE
        elif has_multiple_questions or has_comparison or word_count > 15:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.MODERATE

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Domain-specific words to also filter out
        extra_stopwords = {"tell", "please", "paper", "document", "article"}

        # Extract words
        words = re.findall(r"\b[a-z][a-z0-9]+\b", query.lower())

        # Filter and deduplicate
        keywords = []
        seen: set[str] = set()
        for word in words:
            if (
                word not in STOP_WORDS
                and word not in extra_stopwords
                and word not in seen
                and len(word) > 2
            ):
                keywords.append(word)
                seen.add(word)

        return keywords

    def _expand_terms(self, keywords: list[str]) -> list[str]:
        """Expand keywords with synonyms."""
        expanded = []
        for keyword in keywords:
            if keyword in SYNONYMS:
                expanded.extend(SYNONYMS[keyword])
        return list(set(expanded))  # Deduplicate

    def _generate_search_queries(
        self,
        normalized: str,
        keywords: list[str],
        expanded: list[str],
        query_type: QueryType,
    ) -> list[str]:
        """Generate multiple search queries for retrieval."""
        queries = [normalized]

        # Add keyword-focused query
        if keywords:
            keyword_query = " ".join(keywords[:5])
            if keyword_query != normalized:
                queries.append(keyword_query)

        # Add expanded query
        if expanded:
            all_terms = keywords[:3] + expanded[:3]
            expanded_query = " ".join(all_terms)
            if expanded_query not in queries:
                queries.append(expanded_query)

        # Add type-specific query
        type_query = self._type_specific_query(normalized, query_type)
        if type_query and type_query not in queries:
            queries.append(type_query)

        return queries[:4]  # Limit to 4 queries

    def _type_specific_query(self, query: str, query_type: QueryType) -> str | None:
        """Generate type-specific search query."""
        if query_type == QueryType.METHODOLOGY:
            return f"method approach technique {query}"
        elif query_type == QueryType.RESULTS:
            return f"results findings accuracy performance {query}"
        elif query_type == QueryType.SUMMARY:
            return f"main contribution key finding {query}"
        return None

    def rewrite_vague_query(self, query: str) -> str:
        """
        Rewrite vague queries to be more specific.

        Args:
            query: Original vague query

        Returns:
            More specific query
        """
        query_lower = query.lower().strip()

        # Common vague queries and their rewrites
        rewrites = {
            "tell me about the paper": "What is the main topic, methodology, and key findings of this paper?",
            "tell me about this paper": "What is the main topic, methodology, and key findings of this paper?",
            "what is this about": "What is the main topic and key findings of this document?",
            "summarize": "What are the main findings and conclusions?",
            "summary": "What are the main findings and conclusions?",
            "what does it say": "What are the key points and findings in this document?",
        }

        for vague, specific in rewrites.items():
            if query_lower == vague or query_lower.startswith(vague):
                logger.info(f"Rewrote vague query: '{query}' -> '{specific}'")
                return specific

        return query
