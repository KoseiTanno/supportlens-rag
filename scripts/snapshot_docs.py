"""Snapshot the explicitly allowlisted Japanese GitHub Docs pages."""

import argparse
from pathlib import Path

import httpx

from supportlens.sources import load_sources, snapshot_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-sha", required=True)
    parser.add_argument("--sources", type=Path, default=Path("data/sources.json"))
    parser.add_argument("--output", type=Path, default=Path("snapshot"))
    args = parser.parse_args()
    with httpx.Client(timeout=20) as client:
        snapshot_sources(load_sources(args.sources), args.output, client, args.docs_sha)


if __name__ == "__main__":
    main()
