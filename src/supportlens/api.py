"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException

from supportlens.models import AnswerRequest, AnswerResponse, SearchRequest
from supportlens.service import AnswerService


def create_app(service: AnswerService | None = None) -> FastAPI:
    app = FastAPI(title="SupportLens RAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/answers", response_model=AnswerResponse)
    def answers(request: AnswerRequest) -> AnswerResponse:
        if service is None:
            raise HTTPException(status_code=503, detail={"code": "dependency_unavailable"})
        return service.answer(request.question, request.top_k)

    @app.post("/v1/search")
    def search(request: SearchRequest) -> list[dict[str, object]]:
        if service is None:
            raise HTTPException(status_code=503, detail={"code": "dependency_unavailable"})
        return [
            hit.model_dump(mode="json")
            for hit in service.retriever.search(request.question, request.limit)
        ]

    return app


app = create_app()
