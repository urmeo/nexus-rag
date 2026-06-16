"""Citation validation for generated answers."""

from dataclasses import dataclass
from typing import Any

from nexusrag.agents.citations import extract_citations, strip_citations


@dataclass
class VerificationResult:
    original_answer: str
    verified_answer: str
    citations_found: list[int]
    citations_valid: list[int]
    citations_invalid: list[int]
    warnings: list[str]


class AnswerVerifier:
    """Checks that every [n] citation points to a real source."""

    def verify(self, answer: str, sources: list[Any]) -> VerificationResult:
        found = extract_citations(answer)
        n = len(sources)
        valid = [c for c in found if 1 <= c <= n]
        invalid = [c for c in found if c < 1 or c > n]
        verified = strip_citations(answer, set(valid))

        warnings: list[str] = []
        if invalid:
            warnings.append(f"removed {len(invalid)} out-of-range citation(s): {sorted(set(invalid))}")
        if not found:
            warnings.append("answer cites no sources")

        return VerificationResult(
            original_answer=answer,
            verified_answer=verified,
            citations_found=found,
            citations_valid=valid,
            citations_invalid=invalid,
            warnings=warnings,
        )
