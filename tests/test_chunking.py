from supportlens.chunking import chunk_markdown


def test_chunking_preserves_heading_path_and_url() -> None:
    chunks = chunk_markdown(
        "# Actions\n本文\n## Secrets\n設定方法",
        "https://docs.github.com/ja/actions/test",
        "abc",
    )

    assert chunks[1].heading_path == ["Actions", "Secrets"]
    assert str(chunks[1].source_url) == "https://docs.github.com/ja/actions/test"


def test_chunking_splits_long_text_with_overlap() -> None:
    chunks = chunk_markdown(
        "# A\n" + "あ" * 1900,
        "https://docs.github.com/ja/actions/test",
        "abc",
        max_chars=900,
        overlap_chars=100,
    )

    assert len(chunks) == 3
    assert chunks[0].text[-100:] == chunks[1].text[:100]
