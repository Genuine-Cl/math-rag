"""DeepSeek 问答：严格基于检索证据作答，并强制标注证据编号。

相比原版改进：
- 模型名、超时、重试都来自配置，不再硬编码；
- 调用失败（网络/限流）只记录并返回错误信息，不让整个流程崩掉。
"""
from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)


def _build_llm(cfg: LLMConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=cfg.model,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )


def ask_deepseek_with_sources(question: str, docs: list[Document], cfg: LLMConfig) -> str:
    """让 LLM 仅依据检索证据作答，回答末尾标注使用的证据编号。"""
    if not os.getenv("DEEPSEEK_API_KEY"):
        return "未设置 DEEPSEEK_API_KEY，无法调用 DeepSeek。"

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", f"paper_id={doc.metadata.get('paper_id')}")
        chunk_id = doc.metadata.get("chunk_id", "?")
        parts.append(f"[证据 {i} | {source} | chunk {chunk_id}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(parts)

    llm = _build_llm(cfg)
    prompt = f"""你是数学论文问答助手。请严格只依据下方提供的论文片段回答问题。

规则：
1. 不要使用片段之外的知识，不要编造定理、条件或结论。
2. 如果证据不足，直接说："当前文献库没有足够证据回答该问题。"
3. 用中文回答；数学符号和论文术语可保留英文或 LaTeX。
4. 回答末尾必须标注你使用的证据编号，例如：[证据 1]。

问题：
{question}

论文证据：
{context}
"""

    try:
        return llm.invoke(prompt).content
    except Exception as exc:  # noqa: BLE001 - 网络/限流等，不让主流程崩溃
        logger.error("DeepSeek 调用失败：%s", exc)
        return f"DeepSeek 调用失败：{exc}"
