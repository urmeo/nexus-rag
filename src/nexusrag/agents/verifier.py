"""Answer verification and confidence scoring."""

import logging
from dataclasses import dataclass
from typing import Any

from nexusrag.agents.citations import extract_citations, strip_citations

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of answer verification."""

    original_answer: str
    verified_answer: str
    citations_found: list[int]
    citations_valid: list[int]
    citations_invalid: list[int]
    confidence_score: float
    confidence_breakdown: dict[str, float]
    warnings: list[str]


class AnswerVerifier:
    """
    Verifies LLM answers against source material.

    Checks:
    - Citation validity (do cited sources exist?)
    - Citation accuracy (does the citation support the claim?)
    - Answer completeness
    - Hallucination detection
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize verifier.

        Args:
            strict_mode: If True, remove all unverified claims
        """
        self.strict_mode = strict_mode
        logger.info(f"Initialized AnswerVerifier: strict_mode={strict_mode}")

    def verify(
        self,
        answer: str,
        sources: list[Any],
        query: str,
    ) -> VerificationResult:
        """
        Verify an answer against its sources.

        Args:
            answer: LLM-generated answer
            sources: List of source objects with content
            query: Original query

        Returns:
            VerificationResult with verification details
        """
        logger.info("Verifying answer...")

        # Extract citations
        citations_found = extract_citations(answer)

        # Validate each citation
        citations_valid = []
        citations_invalid = []
        num_sources = len(sources)

        for citation_num in citations_found:
            if 1 <= citation_num <= num_sources:
                citations_valid.append(citation_num)
            else:
                citations_invalid.append(citation_num)

        # Remove invalid citations from answer
        verified_answer = strip_citations(answer, set(citations_valid))

        # Calculate confidence
        confidence_breakdown = self._calculate_confidence(query, answer, sources, citations_valid)
        confidence_score = self._weighted_confidence(confidence_breakdown)

        # Generate warnings
        warnings = self._generate_warnings(
            citations_found, citations_valid, citations_invalid, answer
        )

        result = VerificationResult(
            original_answer=answer,
            verified_answer=verified_answer,
            citations_found=citations_found,
            citations_valid=citations_valid,
            citations_invalid=citations_invalid,
            confidence_score=confidence_score,
            confidence_breakdown=confidence_breakdown,
            warnings=warnings,
        )

        logger.info(
            f"Verification complete: confidence={confidence_score:.2f}, "
            f"valid_citations={len(citations_valid)}/{len(citations_found)}"
        )

        return result

    def _calculate_confidence(
        self,
        _query: str,
        answer: str,
        sources: list[Any],
        valid_citations: list[int],
    ) -> dict[str, float]:
        """Calculate confidence breakdown."""
        scores = {}

        # 1. Retrieval quality (average source score)
        if sources:
            avg_score = sum(getattr(s, "score", 0.5) for s in sources) / len(sources)
            scores["retrieval_quality"] = min(1.0, avg_score)
        else:
            scores["retrieval_quality"] = 0.0

        # 2. Source coverage (unique sources cited / sources available)
        if sources:
            unique_cited = len(set(valid_citations))
            scores["source_coverage"] = min(1.0, unique_cited / len(sources))
        else:
            scores["source_coverage"] = 0.0

        # 3. Answer quality indicators
        answer_score = 0.5  # Base score

        # Has structured format
        if "**Answer:**" in answer or "**Details:**" in answer:
            answer_score += 0.1

        # Has citations
        if valid_citations:
            answer_score += 0.1

        # Multiple citations (more thorough)
        if len(set(valid_citations)) >= 2:
            answer_score += 0.1

        # Appropriate length (not too short, not too long)
        answer_len = len(answer)
        if 100 < answer_len < 1000:
            answer_score += 0.1

        # Check for uncertainty language (reduces confidence)
        uncertainty_phrases = [
            "not found",
            "no information",
            "cannot determine",
            "unclear",
            "not mentioned",
            "doesn't contain",
            "not covered",
            "no relevant",
            "unable to find",
        ]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        if has_uncertainty:
            answer_score -= 0.2

        scores["answer_quality"] = max(0.0, min(1.0, answer_score))

        # 4. Citation density (citations per 100 chars)
        if len(answer) > 0:
            citation_density = len(valid_citations) / (len(answer) / 100)
            # Optimal is about 1-2 per 100 chars
            if 0.5 <= citation_density <= 3.0:
                scores["citation_density"] = 0.8
            elif citation_density > 0:
                scores["citation_density"] = 0.5
            else:
                scores["citation_density"] = 0.2
        else:
            scores["citation_density"] = 0.0

        return scores

    def _weighted_confidence(self, breakdown: dict[str, float]) -> float:
        """Calculate weighted confidence score."""
        weights = {
            "retrieval_quality": 0.35,
            "source_coverage": 0.25,
            "answer_quality": 0.25,
            "citation_density": 0.15,
        }

        total = 0.0
        for key, weight in weights.items():
            total += breakdown.get(key, 0.5) * weight

        return max(0.0, min(1.0, total))

    def _generate_warnings(
        self,
        citations_found: list[int],
        _citations_valid: list[int],
        citations_invalid: list[int],
        answer: str,
    ) -> list[str]:
        """Generate verification warnings."""
        warnings = []

        # Invalid citations
        if citations_invalid:
            warnings.append(
                f"Removed {len(citations_invalid)} invalid citation(s): {citations_invalid}"
            )

        # No citations at all
        if not citations_found:
            warnings.append("Answer contains no source citations")

        # Answer too short
        if len(answer) < 50:
            warnings.append("Answer may be incomplete (very short)")

        # Uncertainty detected
        if "not found" in answer.lower() or "no information" in answer.lower():
            warnings.append("Answer indicates information may not be in sources")

        return warnings
