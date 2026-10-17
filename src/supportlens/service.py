"""Application orchestration for evidence-gated answers."""

from supportlens.generation import AnswerGenerator, EvidenceGate, build_prompt, citations_from_hits
from supportlens.models import AnswerResponse
from supportlens.retrieval import HybridRetriever


class AnswerService:
    def __init__(
        self, retriever: HybridRetriever, generator: AnswerGenerator, gate: EvidenceGate
    ) -> None:
        self.retriever, self.generator, self.gate = retriever, generator, gate

    def answer(self, question: str, top_k: int = 3) -> AnswerResponse:
        hits = self.retriever.search(question, limit=top_k)
        decision = self.gate.assess(hits)
        if not decision.accepted:
            return AnswerResponse(
                status="insufficient_evidence", answer="", citations=[], reason=decision.reason
            )
        citations = citations_from_hits(hits)
        answer = self.generator.generate(build_prompt(question, citations)).strip()
        if not answer:
            return AnswerResponse(
                status="insufficient_evidence", answer="", citations=[], reason="empty_generation"
            )
        return AnswerResponse(
            status="answered", answer=answer, citations=citations, confidence="medium"
        )
