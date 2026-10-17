"""FastAPI application factory."""

from fastapi import FastAPI, HTTPException

from supportlens.models import AnswerRequest, AnswerResponse, SearchRequest
from supportlens.runtime import build_service
from supportlens.service import AnswerService
from supportlens.settings import Settings


def create_app(service: AnswerService | None = None, startup_error: str | None = None) -> FastAPI:
    app = FastAPI(title="SupportLens RAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok" if service is not None else "not_ready"}

    @app.post("/v1/answers", response_model=AnswerResponse)
    def answers(request: AnswerRequest) -> AnswerResponse:
        if service is None:
            raise HTTPException(status_code=503, detail={"code": "dependency_unavailable"})
        return service.answer(request.question, request.top_k, request.mode)

    @app.post("/v1/search")
    def search(request: SearchRequest) -> list[dict[str, object]]:
        if service is None:
            raise HTTPException(status_code=503, detail={"code": "dependency_unavailable"})
        return [
            hit.model_dump(mode="json")
            for hit in service.retriever.search(request.question, request.limit, request.mode)
        ]

    if startup_error:
        app.state.startup_error = startup_error
    return app


def create_runtime_app() -> FastAPI:
    try:
        return create_app(build_service(Settings()))
    except Exception as error:  # surfaced as a deterministic 503 until operators index a snapshot
        return create_app(startup_error=str(error))


app = create_runtime_app()
