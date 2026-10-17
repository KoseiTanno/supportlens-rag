"""Qdrant collection setup, persistence, and dense retrieval."""

from qdrant_client import QdrantClient, models

from supportlens.models import Chunk, SearchHit


class QdrantChunkStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client, self.collection = client, collection

    def ensure_collection(self, dimension: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config={
                    "dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)
                },
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=chunk.chunk_id, vector={"dense": vector}, payload=chunk.model_dump(mode="json")
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points)

    def search(self, vector: list[float], limit: int = 10) -> list[SearchHit]:
        """Return typed chunks with their Qdrant similarity score."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using="dense",
            limit=limit,
            with_payload=True,
        )
        return [
            SearchHit(
                chunk=Chunk.model_validate(point.payload),
                score=float(point.score),
                dense_rank=rank,
            )
            for rank, point in enumerate(result.points, start=1)
        ]
