"""Small, dependency-free retrieval metrics for reproducible evaluation."""

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        raise ValueError("relevant must not be empty")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, identifier in enumerate(retrieved[:k], start=1):
        if identifier in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, identifier in enumerate(retrieved[:k], start=1)
        if identifier in relevant
    )
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0
