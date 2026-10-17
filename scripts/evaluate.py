"""Evaluate a live SupportLens index against a local JSONL relevance set."""

import argparse
import json
from pathlib import Path

from supportlens.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from supportlens.runtime import build_service
from supportlens.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", choices=["dense", "hybrid", "hybrid_rerank"], default="hybrid")
    args = parser.parse_args()
    service = build_service(Settings())
    rows = [
        json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line
    ]
    if not rows:
        raise SystemExit("evaluation dataset is empty")
    scores: list[tuple[float, float, float]] = []
    for row in rows:
        hit_ids = [
            hit.chunk.chunk_id
            for hit in service.retriever.search(row["question"], limit=10, mode=args.mode)
        ]
        relevant = set(row["relevant_chunk_ids"])
        scores.append(
            (
                recall_at_k(hit_ids, relevant, 5),
                reciprocal_rank(hit_ids, relevant, 10),
                ndcg_at_k(hit_ids, relevant, 10),
            )
        )
    names = ["recall_at_5", "mrr_at_10", "ndcg_at_10"]
    metrics = {
        name: sum(score[i] for score in scores) / len(scores) for i, name in enumerate(names)
    }
    print(
        json.dumps(
            {"mode": args.mode, **metrics, "queries": len(scores)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
