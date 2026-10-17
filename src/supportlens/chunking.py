"""Deterministic, heading-aware Markdown chunking."""

import hashlib
import re

from pydantic import HttpUrl

from supportlens.models import Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _clean(markdown: str) -> str:
    text = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.DOTALL)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        pieces.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap_chars
    return pieces


def chunk_markdown(
    markdown: str,
    source_url: str,
    document_sha: str,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[Chunk]:
    """Split Markdown without losing the section that gives text meaning."""
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be non-negative and less than max_chars")
    headings: list[str] = []
    sections: list[tuple[list[str], list[str]]] = []
    body: list[str] = []
    for line in _clean(markdown).splitlines():
        match = _HEADING.match(line)
        if match:
            if body:
                sections.append((headings.copy(), body))
                body = []
            level, title = len(match.group(1)), match.group(2)
            headings = headings[: level - 1] + [title]
        else:
            body.append(line)
    if body:
        sections.append((headings.copy(), body))

    chunks: list[Chunk] = []
    for path, lines in sections:
        text = "\n".join(lines).strip()
        if not text:
            continue
        for part in _split_text(text, max_chars, overlap_chars):
            payload = f"{document_sha}\0{' / '.join(path)}\0{part}".encode()
            chunks.append(
                Chunk(
                    chunk_id=hashlib.sha256(payload).hexdigest(),
                    text=part,
                    source_url=HttpUrl(source_url),
                    heading_path=path,
                    document_sha=document_sha,
                    ordinal=len(chunks),
                )
            )
    return chunks
