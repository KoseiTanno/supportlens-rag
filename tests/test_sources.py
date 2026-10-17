from pathlib import Path

import httpx
import pytest

from supportlens.sources import SourceSpec, snapshot_sources, validate_source_url


class FakeClient:
    def get(self, url: str, *, follow_redirects: bool = False) -> httpx.Response:
        assert follow_redirects is False
        return httpx.Response(200, text="<main><h1>Actions</h1><p>本文</p></main>")


def test_source_rejects_external_redirect() -> None:
    with pytest.raises(ValueError, match="docs.github.com"):
        validate_source_url("https://example.com/actions")


def test_snapshot_writes_manifest_and_normalized_markdown(tmp_path: Path) -> None:
    manifest = snapshot_sources(
        [SourceSpec(url="https://docs.github.com/ja/actions/test", title="test")],
        tmp_path,
        FakeClient(),
        docs_commit_sha="abc",
    )

    assert manifest.documents[0].license == "CC-BY-4.0"
    assert (tmp_path / "documents" / "actions-test.md").exists()
