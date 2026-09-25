"""文本切分：基线通用切分 vs 数学感知切分。

数学感知的核心思路（面试可讲）：
公式（equation/align/gather/$$...$$/\[...\]）先整体替换成占位符，
用通用切分器切完后，再把公式原样还原。这样保证：
- 公式环境是"原子单元"，绝不会被 chunk 边界拦腰切断；
- 一个 chunk 要么含完整公式 + 上下文，要么是纯文本，便于标记 block_type。

这比"只在分隔符里塞 \\end{equation}"更可靠：后者仍可能把长公式切断，
而且 RecursiveCharacterTextSplitter 默认把 separator 当字面量处理。
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# 公式环境（equation/align/gather/multline/displaymath，带不带 * 都算）
_FORMULA_ENV_RE = re.compile(
    r"\\begin\{(equation|equation\*|align|align\*|gather|gather\*|multline|multline\*|displaymath)\}"
    r".*?"
    r"\\end\{\1\}",
    flags=re.DOTALL,
)
# 独立成行的 $$...$$ 或 \[...\] 显示公式
_DISPLAY_MATH_RE = re.compile(r"(?:\$\$.*?\$\$|\\\[.*?\\\])", flags=re.DOTALL)

_PLACEHOLDER = "__FORMULA_{}__"


def _protect_formulas(text: str) -> tuple[str, list[str]]:
    """把公式环境整体替换为占位符，返回 (替换后文本, 公式列表)。"""
    formulas: list[str] = []

    def _repl(match: re.Match) -> str:
        formulas.append(match.group(0))
        return _PLACEHOLDER.format(len(formulas) - 1)

    protected = _FORMULA_ENV_RE.sub(_repl, text)
    protected = _DISPLAY_MATH_RE.sub(_repl, protected)
    return protected, formulas


def _restore_formulas(text: str, formulas: list[str]) -> str:
    for i, formula in enumerate(formulas):
        text = text.replace(_PLACEHOLDER.format(i), formula)
    return text


def _is_formula_chunk(chunk: str) -> bool:
    """判断 chunk 是否包含公式（用于标记 block_type）。"""
    if _FORMULA_ENV_RE.search(chunk) or _DISPLAY_MATH_RE.search(chunk):
        return True
    if re.search(r"\$[^$]+\$", chunk):  # 行内公式 $...$
        return True
    if re.search(r"\\(?:frac|sum|int|partial|nabla|Delta|leq|geq|left|right)\b", chunk):
        return True
    return False


def baseline_split(papers: Iterable, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """基线组：通用递归切分，不感知 LaTeX 结构。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks: list[Document] = []
    for paper_id, paper in enumerate(papers):
        for chunk_id, chunk in enumerate(splitter.split_text(paper.text)):
            chunks.append(Document(
                page_content=chunk,
                metadata={
                    "source": paper.name,
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "split_method": "baseline",
                },
            ))
    return chunks


def math_aware_split(papers: Iterable, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """数学感知组：公式环境原子化，切分后还原，并标记 block_type。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for paper_id, paper in enumerate(papers):
        protected, formulas = _protect_formulas(paper.text)
        for chunk_id, raw_chunk in enumerate(splitter.split_text(protected)):
            restored = _restore_formulas(raw_chunk, formulas)
            chunks.append(Document(
                page_content=restored,
                metadata={
                    "source": paper.name,
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "split_method": "math_aware",
                    "block_type": "formula_context" if _is_formula_chunk(restored) else "text",
                },
            ))
    return chunks
