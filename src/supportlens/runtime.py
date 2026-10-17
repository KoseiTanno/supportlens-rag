"""Runtime wiring for the API and explicit indexing command.

Imports are kept here so unit tests can exercise domain code without downloading
an embedding model or requiring a Qdrant server.
"""

from collections.abc import Iterable
from pathlib import Path

from supportlens.chunking import chunk_markdown
from supportlens.generation import AnswerGenerator, EvidenceGate, ExtractiveAnswerGenerator
from supportlens.indexing import QdrantChunkStore
from supportlens.models import Chunk, SearchHit, SnapshotManifest
from supportlens.retrieval import Bm25Index, HybridRetriever
from supportlens.service import AnswerService
from supportlens.settings import Settings


class FastEmbedEmbedder:
    """CPU-local multilingual vector embedding via FastEmbed/ONNX Runtime."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [list(vector) for vector in self.model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FastEmbedReranker:
    """Optional multilingual cross-encoder reranker, run locally on CPU."""

    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name)

    def rerank(self, question: str, hits: list[SearchHit]) -> list[SearchHit]:
        scores = list(self.model.rerank(question, [hit.chunk.text for hit in hits]))
        return [
            hit.model_copy(update={"rerank_score": float(score), "score": float(score)})
            for hit, score in sorted(
                zip(hits, scores, strict=True), key=lambda item: item[1], reverse=True
            )
        ]


class OpenAIAnswerGenerator:
    """Optional provider-backed generator; only constructed with an API key."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text


def _load_chunks(corpus_dir: Path) -> list[Chunk]:
    manifest = SnapshotManifest.model_validate_json((corpus_dir / "manifest.json").read_text())
    chunks: list[Chunk] = []
    for record in manifest.documents:
        markdown = (corpus_dir / record.local_path).read_text(encoding="utf-8")
        chunks.extend(
            chunk_markdown(
                markdown,
                str(record.source_url),
                record.document_sha,
            )
        )
    return chunks


def index_snapshot(settings: Settings) -> int:
    """Embed and persist a checked-in snapshot. This is an explicit operator action."""
    chunks = _load_chunks(Path(settings.corpus_dir))
    if not chunks:
        raise ValueError("snapshot produced no chunks")
    embedder = FastEmbedEmbedder(settings.embedding_model)
    vectors = embedder.embed([chunk.text for chunk in chunks])
    from qdrant_client import QdrantClient

    store = QdrantChunkStore(QdrantClient(url=settings.qdrant_url), settings.qdrant_collection)
    store.ensure_collection(len(vectors[0]))
    store.upsert(chunks, vectors)
    return len(chunks)


def build_service(settings: Settings) -> AnswerService:
    """Build a live service only when both corpus and Qdrant collection are ready."""
    chunks = _load_chunks(Path(settings.corpus_dir))
    if not chunks:
        raise ValueError("snapshot produced no chunks")
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    if not client.collection_exists(settings.qdrant_collection):
        raise RuntimeError("collection is not indexed; run scripts/index_corpus.py")
    embedder = FastEmbedEmbedder(settings.embedding_model)
    store = QdrantChunkStore(client, settings.qdrant_collection)
    reranker = FastEmbedReranker(settings.reranker_model) if settings.enable_reranker else None
    retriever = HybridRetriever(
        Bm25Index.from_chunks(chunks),
        lambda question, limit: store.search(embedder.embed_query(question), limit),
        reranker=reranker,
    )
    generator: AnswerGenerator
    if settings.openai_api_key is None:
        generator = ExtractiveAnswerGenerator()
    else:
        generator = OpenAIAnswerGenerator(
            settings.openai_api_key.get_secret_value(), settings.openai_model
        )
    return AnswerService(retriever, generator, EvidenceGate(settings.evidence_score_threshold))
