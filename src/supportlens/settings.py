"""Runtime configuration loaded from environment variables."""

from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings with safe local defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUPPORTLENS_")

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "supportlens_chunks"
    corpus_dir: str = "data/snapshot"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    enable_reranker: bool = False
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-4.1-mini"
    max_question_chars: Annotated[int, Field(gt=0, le=10_000)] = 1_000
    # RRF hybrid scores are rank-based and typically around 0.03 for relevant hits.
    evidence_score_threshold: Annotated[float, Field(ge=0, le=1)] = 0.02
