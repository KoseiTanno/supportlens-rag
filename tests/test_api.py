from fastapi.testclient import TestClient

from supportlens.api import create_app


def test_health_returns_ok() -> None:
    assert TestClient(create_app()).get("/health").json() == {"status": "ok"}


def test_answers_without_ready_index_returns_503() -> None:
    assert (
        TestClient(create_app()).post("/v1/answers", json={"question": "Secrets は？"}).status_code
        == 503
    )
