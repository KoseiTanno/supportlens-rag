"""Application orchestration for evidence-gated answers."""

from supportlens.generation import (
    AnswerGenerator,
    EvidenceGate,
    ExtractiveAnswerGenerator,
    build_prompt,
    citations_from_hits,
)
from supportlens.models import AnswerResponse
from supportlens.retrieval import HybridRetriever


class AnswerService:
    def __init__(
        self, retriever: HybridRetriever, generator: AnswerGenerator, gate: EvidenceGate
    ) -> None:
        self.retriever, self.generator, self.gate = retriever, generator, gate

    def answer(self, question: str, top_k: int = 3, mode: str = "hybrid_rerank") -> AnswerResponse:
        hits = self.retriever.search(question, limit=top_k, mode=mode)
        decision = self.gate.assess(hits)
        if not decision.accepted:
            return AnswerResponse(
                status="insufficient_evidence", answer="", citations=[], reason=decision.reason
            )
        citations = citations_from_hits(hits)
        if isinstance(self.generator, ExtractiveAnswerGenerator):
            answer = self.generator.generate_from_citations(citations).strip()
        else:
            answer = self.generator.generate(build_prompt(question, citations)).strip()
        if not answer:
            return AnswerResponse(
                status="insufficient_evidence", answer="", citations=[], reason="empty_generation"
            )
        return AnswerResponse(
            status="answered", answer=answer, citations=citations, confidence="medium"
        )
