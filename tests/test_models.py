from uuid import uuid4

import pytest
from pydantic import ValidationError

from supportlens.models import AnswerResponse


def test_answered_response_requires_citation() -> None:
    with pytest.raises(ValidationError):
        AnswerResponse(
            status="answered",
            answer="Secrets を設定します",
            citations=[],
            trace_id=uuid4(),
        )


def test_insufficient_evidence_has_empty_answer() -> None:
    response = AnswerResponse(
        status="insufficient_evidence",
        answer="",
        citations=[],
        reason="low_evidence",
        trace_id=uuid4(),
    )

    assert response.answer == ""
