"""Stable domain and API contracts for SupportLens."""

from datetime import UTC, datetime
from typing import Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl, model_validator


class Chunk(BaseModel):
    chunk_id: str
    text: str = Field(min_length=1)
    source_url: HttpUrl
    heading_path: list[str]
    document_sha: str = Field(min_length=1)
    ordinal: int = Field(ge=0)


class Citation(BaseModel):
    chunk_id: str
    title: str
    url: HttpUrl
    excerpt: str = Field(min_length=1)
    document_sha: str


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = Field(default=None, ge=1)
    bm25_rank: int | None = Field(default=None, ge=1)
    rerank_score: float | None = None


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    top_k: int = Field(default=3, ge=1, le=10)
    mode: Literal["hybrid_rerank", "hybrid", "dense"] = "hybrid_rerank"


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    limit: int = Field(default=10, ge=1, le=20)
    mode: Literal["hybrid_rerank", "hybrid", "dense"] = "hybrid_rerank"


class AnswerResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"] | None = None
    reason: str | None = None
    trace_id: UUID = Field(default_factory=uuid4)

    @model_validator(mode="after")
    def validate_status_shape(self) -> Self:
        if self.status == "answered" and (not self.answer or not self.citations):
            raise ValueError("answered response requires answer and citation")
        if self.status == "insufficient_evidence" and self.answer:
            raise ValueError("insufficient evidence response must have empty answer")
        return self


class DocumentRecord(BaseModel):
    source_url: HttpUrl
    local_path: str
    title: str
    document_sha: str
    retrieved_at: datetime
    license: Literal["CC-BY-4.0"] = "CC-BY-4.0"


class SnapshotManifest(BaseModel):
    docs_commit_sha: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    documents: list[DocumentRecord]
