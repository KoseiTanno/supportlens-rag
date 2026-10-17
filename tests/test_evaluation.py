import pytest

from supportlens.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics_reward_early_relevant_result() -> None:
    results = ["b", "a", "c"]
    assert recall_at_k(results, {"a", "x"}, 2) == 0.5
    assert reciprocal_rank(results, {"a"}, 10) == 0.5
    assert ndcg_at_k(results, {"a"}, 10) == 1 / 1.5849625007211563


def test_metrics_reject_empty_relevance() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)
