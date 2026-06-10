"""Query planning and decomposition."""

import re
from dataclasses import dataclass, field
from enum import Enum

from nexusrag.agents.llm import LLMClient


class QueryComplexity(Enum):
    """Query complexity levels."""

    SIMPLE = "simple"  # Single fact lookup
    MODERATE = "moderate"  # Requires synthesis
    COMPLEX = "complex"  # Multi-hop reasoning


@dataclass
class QueryStep:
    """A single step in a query plan."""

    query: str
    purpose: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class QueryPlan:
    """Execution plan for a query."""

    original_query: str
    complexity: QueryComplexity
    steps: list[QueryStep]
    reasoning: str = ""

    @property
    def is_multi_step(self) -> bool:
        return len(self.steps) > 1


PLANNING_PROMPT = """Analyze this research question and create a retrieval plan.

Question: {query}

Determine:
1. Complexity: SIMPLE (single fact), MODERATE (synthesis needed), or COMPLEX (multi-hop reasoning)
2. If MODERATE/COMPLEX, decompose into sub-queries that can be answered independently

Respond in this exact format:
COMPLEXITY: [SIMPLE|MODERATE|COMPLEX]
REASONING: [Brief explanation]
STEPS:
1. [First sub-query] | PURPOSE: [Why this is needed]
2. [Second sub-query] | PURPOSE: [Why this is needed] | DEPENDS: 1
...

Use DEPENDS to indicate steps that need results from earlier steps (comma-separated step numbers).
For SIMPLE queries, provide just one step.
"""


class QueryPlanner:
    """Analyzes and decomposes complex queries."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, query: str) -> QueryPlan:
        """
        Create an execution plan for the query.

        Args:
            query: User's research question

        Returns:
            QueryPlan with complexity assessment and steps
        """
        # Quick heuristic check for simple queries
        if self._is_likely_simple(query):
            return QueryPlan(
                original_query=query,
                complexity=QueryComplexity.SIMPLE,
                steps=[QueryStep(query=query, purpose="Direct answer")],
                reasoning="Simple factual query",
            )

        # Use LLM for complex query analysis
        prompt = PLANNING_PROMPT.format(query=query)
        response = self.llm.generate(prompt, temperature=0.1, max_tokens=200)

        return self._parse_plan(query, response)

    def _is_likely_simple(self, query: str) -> bool:
        """Heuristic check for simple queries."""
        words = query.split()

        # Complex indicators
        complex_patterns = [
            r"\b(compare|contrast|difference|similarity)\b",
            r"\b(how does .+ relate to)\b",
            r"\b(what are the .+ and .+)\b",
            r"\b(analyze|evaluate|assess)\b",
            r"\b(multiple|several|various)\b",
        ]

        for pattern in complex_patterns:
            if re.search(pattern, query.lower()):
                return False

        return len(words) <= 15

    def _parse_plan(self, original_query: str, response: str) -> QueryPlan:
        """Parse LLM response into QueryPlan."""
        lines = response.strip().split("\n")

        complexity = QueryComplexity.MODERATE
        reasoning = ""
        steps: list[QueryStep] = []

        for line in lines:
            line = line.strip()

            if line.startswith("COMPLEXITY:"):
                complexity_str = line.split(":", 1)[1].strip().upper()
                complexity = self._parse_complexity(complexity_str)

            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

            elif re.match(r"^\d+\.", line):
                step = self._parse_step(line)
                if step:
                    steps.append(step)

        # Fallback if parsing failed
        if not steps:
            steps = [QueryStep(query=original_query, purpose="Direct answer")]

        return QueryPlan(
            original_query=original_query,
            complexity=complexity,
            steps=steps,
            reasoning=reasoning,
        )

    def _parse_complexity(self, text: str) -> QueryComplexity:
        """Parse complexity from text."""
        if "SIMPLE" in text:
            return QueryComplexity.SIMPLE
        if "COMPLEX" in text:
            return QueryComplexity.COMPLEX
        return QueryComplexity.MODERATE

    def _parse_step(self, line: str) -> QueryStep | None:
        """Parse a step line, including optional DEPENDS section."""
        # Remove leading number
        content = re.sub(r"^\d+\.\s*", "", line)

        depends_on: list[int] = []

        if "|" in content:
            parts = content.split("|")
            query = parts[0].strip()
            purpose = "Sub-query"

            for part in parts[1:]:
                part = part.strip()
                if part.upper().startswith("DEPENDS:"):
                    deps_str = part.split(":", 1)[1].strip()
                    for dep in deps_str.split(","):
                        dep = dep.strip()
                        if dep.isdigit():
                            depends_on.append(int(dep))
                elif part.upper().startswith("PURPOSE:"):
                    purpose = part.split(":", 1)[1].strip()
                else:
                    purpose = part.replace("PURPOSE:", "").strip()
        else:
            query = content.strip()
            purpose = "Sub-query"

        if query:
            return QueryStep(query=query, purpose=purpose, depends_on=depends_on)
        return None

    def reformulate(self, query: str, context: str = "") -> str:
        """
        Reformulate a query for better retrieval.

        Args:
            query: Original query
            context: Optional context from previous attempts

        Returns:
            Reformulated query
        """
        prompt = f"""Reformulate this query to improve document retrieval.
Keep the same meaning but use different words or phrasing.

Original query: {query}
{f"Context: {context}" if context else ""}

Reformulated query:"""

        response = self.llm.generate(prompt, temperature=0.3, max_tokens=100)
        return response.strip().strip('"')
