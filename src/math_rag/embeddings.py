"""嵌入模型工厂。"""
from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


def build_embeddings(cfg: EmbeddingConfig) -> HuggingFaceEmbeddings:
    """构建本地 HuggingFace 嵌入模型（首次运行会下载权重）。"""
    logger.info("加载嵌入模型：%s（device=%s）", cfg.model_name, cfg.device)
    return HuggingFaceEmbeddings(
        model_name=cfg.model_name,
        model_kwargs={"device": cfg.device},
        encode_kwargs={
            "normalize_embeddings": cfg.normalize_embeddings,
            "batch_size": cfg.batch_size,
        },
    )
