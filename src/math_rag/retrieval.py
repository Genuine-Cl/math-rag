"""向量库与检索器构建：MMR 稠密检索 + 可选 BM25 混合检索。

面试可讲点：
- 纯稠密向量对数学符号/精确术语不敏感（bge 对 LaTeX 表示能力有限），
  用 BM25 稀疏检索做互补，靠 RRF（Reciprocal Rank Fusion）融合，
  兼顾"语义相近"和"关键词/符号精确命中"；
- RRF 融合是自己实现的（约 30 行），不依赖 langchain 各版本间反复搬家的
  EnsembleRetriever，版本兼容性最好；
- fetch_k 会被钳制为不超过语料块数，避免 MMR 在语料很小时报错。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from .config import RetrievalConfig

logger = logging.getLogger(__name__)

# RRF 的排名平滑常数（业界常用 60）
RRF_K = 60


def build_vector_db(chunks: list[Document], persist_dir: Path, collection_name: str, embeddings) -> Chroma:
    """建立 Chroma 向量库；先清空旧目录，保证每次实验可复现。"""
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    logger.info("构建向量库 %s：%d 块 -> %s", collection_name, len(chunks), persist_dir)
    # 嵌入计算没有逐批进度输出，先告诉用户"正在干，不是卡死"，
    # 首次运行（CPU）可能要几分钟到几十分钟。
    logger.info("开始嵌入 %d 个 chunk（CPU 上首次运行需数分钟，请耐心等待，不要中断）", len(chunks))
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )
    logger.info("向量库构建完成：%s", persist_dir)
    return db


class DenseRetriever:
    """MMR / 相似度稠密检索器。直接调用 Chroma 的原生方法，版本兼容性最好。"""

    def __init__(self, db: Chroma, cfg: RetrievalConfig, n_docs: int):
        self.db = db
        self.cfg = cfg
        self.fetch_k = min(cfg.fetch_k, n_docs)

    def invoke(self, query: str) -> list[Document]:
        if self.cfg.search_type == "mmr":
            return self.db.max_marginal_relevance_search(
                query,
                k=self.cfg.k,
                fetch_k=self.fetch_k,
                lambda_mult=self.cfg.lambda_mult,
            )
        return self.db.similarity_search(query, k=self.cfg.k)


def build_dense_retriever(db: Chroma, cfg: RetrievalConfig, n_docs: int) -> DenseRetriever:
    """构建纯稠密检索器，fetch_k 用语料块数钳制。"""
    return DenseRetriever(db, cfg, n_docs)


class HybridRetriever:
    """稠密向量 + BM25 混合检索：RRF 加权融合（自实现）。

    融合公式：score(d) = Σ weight_i / (K + rank_i(d))，取加权分最高的 k 个。
    同一个 chunk 在两边都出现时分数累加，相当于"两个检索器都认可"的片段被抬升。
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, dense_weight: float, k: int):
        self.dense = dense
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.k = k

    @staticmethod
    def _key(doc: Document) -> tuple:
        meta = doc.metadata
        return (meta.get("source"), meta.get("chunk_id"), meta.get("split_method"))

    def _fuse(self, lists: list[list[Document]], weights: list[float]) -> list[Document]:
        scores: dict[tuple, float] = {}
        doc_by_key: dict[tuple, Document] = {}
        for docs, weight in zip(lists, weights):
            for rank, doc in enumerate(docs, start=1):
                key = self._key(doc)
                scores[key] = scores.get(key, 0.0) + weight / (RRF_K + rank)
                doc_by_key.setdefault(key, doc)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_by_key[key] for key, _ in ranked[: self.k]]

    def invoke(self, query: str) -> list[Document]:
        dense_docs = self.dense.invoke(query)
        bm25_docs = self.bm25.invoke(query)
        return self._fuse([dense_docs, bm25_docs], [self.dense_weight, 1 - self.dense_weight])


def build_hybrid_retriever(db: Chroma, chunks: list[Document], cfg: RetrievalConfig) -> HybridRetriever:
    """构建稠密 + BM25 混合检索器。"""
    dense = build_dense_retriever(db, cfg, len(chunks))
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = cfg.k
    return HybridRetriever(dense=dense, bm25=bm25, dense_weight=cfg.dense_weight, k=cfg.k)
