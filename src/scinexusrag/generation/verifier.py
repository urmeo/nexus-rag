"""Answer verification."""

from dataclasses import dataclass

from scinexusrag.generation.citations import (
    count_unparseable_citations,
    extract_citations,
    strip_citations,
)
from scinexusrag.generation.synthesizer import Source


@dataclass
class VerificationResult:
    original_answer: str
    verified_answer: str
    citations_found: list[int]
    citations_valid: list[int]
    citations_invalid: list[int]
    warnings: list[str]


class AnswerVerifier:
    def verify(self, answer: str, sources: list[Source]) -> VerificationResult:
        found = extract_citations(answer)
        source_indices = [source.index for source in sources]
        if any(type(index) is not int or index <= 0 for index in source_indices) or len(
            set(source_indices)
        ) != len(source_indices):
            raise ValueError("source indices must be unique positive integers")
        indices = set(source_indices)
        valid = [c for c in found if c in indices]
        invalid = [c for c in found if c not in indices]
        verified = strip_citations(answer, set(valid))

        warnings: list[str] = []
        if invalid:
            warnings.append(
                f"removed {len(invalid)} out-of-range citation(s): {sorted(set(invalid))}"
            )
        if unparseable := count_unparseable_citations(answer):
            warnings.append(f"removed {unparseable} unparseable numeric citation(s)")
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
