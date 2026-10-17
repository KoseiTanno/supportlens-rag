from supportlens.models import Chunk, SearchHit
from supportlens.retrieval import Bm25Index, rrf_fuse


def chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=identifier,
        text=text,
        source_url="https://docs.github.com/ja/actions/test",
        heading_path=["test"],
        document_sha="sha",
        ordinal=0,
    )


def hit(identifier: str) -> SearchHit:
    return SearchHit(chunk=chunk(identifier, identifier), score=1.0)


def test_rrf_promotes_item_found_in_both_rankings() -> None:
    result = rrf_fuse([[hit("a"), hit("b")], [hit("b"), hit("c")]])
    assert result[0].chunk.chunk_id == "b"


def test_bm25_returns_exact_secret_term() -> None:
    index = Bm25Index.from_chunks(
        [chunk("a", "暗号化された secrets を設定する"), chunk("b", "ワークフローを実行する")]
    )
    assert index.search("secrets", limit=1)[0].chunk.chunk_id == "a"
