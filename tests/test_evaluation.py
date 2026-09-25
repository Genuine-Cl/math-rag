"""评估指标的单元测试（手算验证）。"""
from __future__ import annotations

import pytest

from math_rag.evaluation import mrr, ndcg, recall_at_k


def _doc(source: str) -> dict:
    return {"metadata": {"source": source}, "content": "x"}


def test_recall_at_k():
    retrieved = [_doc("a"), _doc("b"), _doc("c")]
    assert recall_at_k(retrieved, {"b"}, 3) == 1.0
    assert recall_at_k(retrieved, {"b", "d"}, 3) == 0.5
    assert recall_at_k(retrieved, {"d"}, 3) == 0.0
    assert recall_at_k(retrieved, set(), 3) == 0.0


def test_mrr():
    retrieved = [_doc("a"), _doc("b")]
    assert mrr(retrieved, {"b"}) == 0.5
    assert mrr(retrieved, {"a"}) == 1.0
    assert mrr(retrieved, {"z"}) == 0.0


def test_ndcg_perfect_ranking():
    retrieved = [_doc("a"), _doc("b")]
    assert ndcg(retrieved, {"a", "b"}, 2) == 1.0


def test_ndcg_penalizes_wrong_order():
    retrieved = [_doc("a"), _doc("b")]
    # 相关项排在第 2 位时，nDCG 应小于完美排序
    assert ndcg(retrieved, {"b"}, 2) < 1.0


def test_ndcg_counts_each_paper_only_once():
    # 同一篇论文的多个 chunk 都进 top-k 时，只对第一次命中计分，nDCG 不能超过 1
    retrieved = [_doc("a"), _doc("a"), _doc("b")]
    result = ndcg(retrieved, {"a", "b"}, 3)
    # dcg = 1/log2(2) + 0 + 1/log2(4) = 1.5；idcg = 1/log2(2) + 1/log2(3) ≈ 1.6309
    assert result == pytest.approx(1.5 / 1.6309, abs=1e-3)
    assert result <= 1.0
