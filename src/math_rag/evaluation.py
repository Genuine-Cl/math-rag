"""检索评估：Recall@k / MRR / nDCG。

标注集格式（data/eval/ground_truth.json）：
[
  {"question": "What is the p-Laplace operator?", "relevant": ["sample_paper"]}
]
relevant 里是"包含该问题答案"的论文名，与 Paper.name 一致。

面试可讲点：
- Recall@k：相关论文有没有被捞回来（衡量召回）；
- MRR：第一个相关结果排第几（衡量"答案是否靠前"）；
- nDCG：考虑排序位置的相关性折扣增益（衡量整体排序质量）。
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)


def load_ground_truth(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在：{path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    logger.info("加载标注集：%d 个问题（来自 %s）", len(data), path.name)
    return data


def recall_at_k(retrieved: list[dict], relevant: set[str], k: int) -> float:
    """前 k 个结果命中的相关论文数 / 相关论文总数。"""
    if not relevant:
        return 0.0
    top = {doc["metadata"].get("source") for doc in retrieved[:k]}
    return len(top & relevant) / len(relevant)


def mrr(retrieved: list[dict], relevant: set[str]) -> float:
    """第一个相关结果的倒数排名；第一名命中时为 1.0。"""
    for i, doc in enumerate(retrieved, start=1):
        if doc["metadata"].get("source") in relevant:
            return 1.0 / i
    return 0.0


def ndcg(retrieved: list[dict], relevant: set[str], k: int) -> float:
    """考虑排序位置的相关性折扣累积增益（越大越好，最大 1.0）。

    注意：标注是"论文级"的，而检索结果是"chunk 级"的——同一篇论文的多个 chunk
    可能同时进入 top-k。DCG 只对每篇相关论文的**第一次命中**计分，
    否则会出现 dcg > idcg、nDCG > 1 的非法值。
    """
    dcg = 0.0
    seen: set[str] = set()
    for i, doc in enumerate(retrieved[:k], start=1):
        source = doc["metadata"].get("source")
        if source in relevant and source not in seen:
            seen.add(source)
            dcg += 1.0 / math.log2(i + 1)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_retriever(retriever, ground_truth: list[dict], k: int = 3) -> dict:
    """对标注集整体评估，返回逐题明细 + 平均指标。"""
    details = []
    for item in ground_truth:
        question = item["question"]
        relevant = set(item["relevant"])
        docs = retriever.invoke(question)
        retrieved = [{"metadata": doc.metadata, "content": doc.page_content} for doc in docs]
        details.append({
            "question": question,
            "relevant": sorted(relevant),
            "recall": round(recall_at_k(retrieved, relevant, k), 4),
            "mrr": round(mrr(retrieved, relevant), 4),
            "ndcg": round(ndcg(retrieved, relevant, k), 4),
            # 记录实际检索到的来源，便于逐题分析失败原因
            "retrieved_sources": [doc["metadata"].get("source") for doc in retrieved],
        })

    def _mean(key: str) -> float:
        return round(sum(d[key] for d in details) / len(details), 4)

    return {
        "k": k,
        "n_questions": len(details),
        "mean_recall": _mean("recall"),
        "mean_mrr": _mean("mrr"),
        "mean_ndcg": _mean("ndcg"),
        "details": details,
    }
