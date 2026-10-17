"""Embed the verified local snapshot into Qdrant."""

from supportlens.runtime import index_snapshot
from supportlens.settings import Settings

if __name__ == "__main__":
    count = index_snapshot(Settings())
    print(f"Indexed {count} chunks.")
