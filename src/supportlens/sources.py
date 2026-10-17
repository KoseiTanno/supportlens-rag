"""Safe snapshotting of a small allowlist of GitHub Docs pages."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl

from supportlens.models import DocumentRecord, SnapshotManifest

# The Japanese workflow-syntax reference is currently about 2.3 MiB.
# Keep a bounded response while allowing that official reference to be indexed.
MAX_BYTES = 3 * 1024 * 1024


class SourceSpec(BaseModel):
    url: str
    title: str


class HttpGetter(Protocol):
    def get(self, url: str, *, follow_redirects: bool = False) -> httpx.Response: ...


def validate_source_url(url: str) -> HttpUrl:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "docs.github.com":
        raise ValueError("source must use https://docs.github.com")
    if not parsed.path.startswith("/ja/actions/"):
        raise ValueError("source path must begin with /ja/actions/")
    return HttpUrl(url)


def _to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main")
    if main is None:
        raise ValueError("source page has no main content")
    lines: list[str] = []
    for element in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name and element.name.startswith("h"):
            lines.append("#" * int(element.name[1]) + " " + text)
        elif element.name == "li":
            lines.append("- " + text)
        elif element.name == "pre":
            lines.extend(["```", text, "```"])
        else:
            lines.append(text)
    return "\n\n".join(lines) + "\n"


def _slug(url: str) -> str:
    path = urlparse(url).path.removeprefix("/ja/").strip("/").lower()
    return re.sub(r"[^a-z0-9]+", "-", path).strip("-")


def snapshot_sources(
    sources: list[SourceSpec], output_dir: Path, client: HttpGetter, docs_commit_sha: str
) -> SnapshotManifest:
    documents_dir = output_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    records: list[DocumentRecord] = []
    for source in sources:
        url = validate_source_url(source.url)
        response = client.get(str(url), follow_redirects=False)
        content = response.content
        if response.status_code != 200 or len(content) > MAX_BYTES:
            raise ValueError(f"could not snapshot {url}")
        path = documents_dir / f"{_slug(str(url))}.md"
        path.write_text(_to_markdown(response.text), encoding="utf-8")
        records.append(
            DocumentRecord(
                source_url=url,
                local_path=str(path.relative_to(output_dir)),
                title=source.title,
                document_sha=docs_commit_sha,
                retrieved_at=datetime.now(UTC),
            )
        )
    manifest = SnapshotManifest(docs_commit_sha=docs_commit_sha, documents=records)
    temp = output_dir / "manifest.json.tmp"
    temp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    temp.replace(output_dir / "manifest.json")
    return manifest


def load_sources(path: Path) -> list[SourceSpec]:
    return [SourceSpec.model_validate(item) for item in json.loads(path.read_text())]
