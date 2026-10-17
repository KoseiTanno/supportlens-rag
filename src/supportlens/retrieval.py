"""Lexical and rank-fusion retrieval primitives."""

import re
from collections.abc import Callable, Sequence
from typing import Protocol

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from supportlens.models import Chunk, SearchHit


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class Bm25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.index = BM25Okapi([_tokens(chunk.text) for chunk in self.chunks])

    @classmethod
    def from_chunks(cls, chunks: Sequence[Chunk]) -> "Bm25Index":
        return cls(chunks)

    def search(self, question: str, limit: int = 10) -> list[SearchHit]:
        scores = self.index.get_scores(_tokens(question))
        ordered = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:limit]
        return [
            SearchHit(chunk=self.chunks[index], score=float(score), bm25_rank=rank + 1)
            for rank, (index, score) in enumerate(ordered)
        ]


def rrf_fuse(rankings: list[list[SearchHit]], k: int = 60) -> list[SearchHit]:
    """Fuse rankings by rank, avoiding incomparable raw score scales."""
    fused: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            current = fused.get(key)
            if current is None:
                fused[key] = hit
            elif hit.bm25_rank is not None:
                current.bm25_rank = hit.bm25_rank
            elif hit.dense_rank is not None:
                current.dense_rank = hit.dense_rank
    return [
        fused[key].model_copy(update={"score": scores[key]})
        for key in sorted(scores, key=lambda item: scores[item], reverse=True)
    ]


class HybridRetriever:
    """Combines a caller-provided dense ranking with BM25 using RRF."""

    def __init__(
        self, bm25: Bm25Index, dense_search: Callable[[str, int], list[SearchHit]]
    ) -> None:
        self.bm25 = bm25
        self.dense_search = dense_search

    def search(self, question: str, limit: int = 10) -> list[SearchHit]:
        return rrf_fuse([self.dense_search(question, limit), self.bm25.search(question, limit)])[
            :limit
        ]
