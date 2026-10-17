import pytest
from pydantic import ValidationError

from supportlens.settings import Settings


def test_settings_uses_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.qdrant_collection == "supportlens_chunks"
    assert settings.max_question_chars == 1_000


def test_settings_rejects_non_positive_question_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(max_question_chars=0, _env_file=None)
