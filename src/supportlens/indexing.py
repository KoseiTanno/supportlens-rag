"""Qdrant collection setup and chunk persistence."""

from qdrant_client import QdrantClient, models

from supportlens.models import Chunk


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
