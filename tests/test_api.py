from fastapi.testclient import TestClient

from supportlens.api import create_app


def test_health_returns_ok() -> None:
    assert TestClient(create_app()).get("/health").json() == {"status": "not_ready"}


def test_home_returns_question_app() -> None:
    response = TestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "SupportLens" in response.text
    assert "質問する" in response.text


def test_answers_without_ready_index_returns_503() -> None:
    assert (
        TestClient(create_app()).post("/v1/answers", json={"question": "Secrets は？"}).status_code
        == 503
    )
