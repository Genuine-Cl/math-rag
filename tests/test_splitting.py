"""切分逻辑的单元测试，重点验证公式不被切断。"""
from __future__ import annotations

from math_rag.latex_loader import Paper
from math_rag.splitting import math_aware_split


def test_formula_is_not_split_across_chunks():
    text = (
        "Intro paragraph. " * 60
        + "\\begin{equation} E = mc^2 \\end{equation}"
        + " Conclusion paragraph. " * 60
    )
    paper = Paper(name="p1", text=text)
    chunks = math_aware_split([paper], chunk_size=200, chunk_overlap=0)

    full_formula = "\\begin{equation} E = mc^2 \\end{equation}"
    assert any(full_formula in chunk.page_content for chunk in chunks)


def test_metadata_carries_source_and_block_type():
    paper = Paper(name="p1", text="\\begin{equation} a+b \\end{equation} plain text")
    chunks = math_aware_split([paper], chunk_size=500, chunk_overlap=0)

    assert chunks[0].metadata["source"] == "p1"
    assert chunks[0].metadata["split_method"] == "math_aware"
    assert chunks[0].metadata["block_type"] == "formula_context"


def test_pure_text_chunk_marked_as_text():
    paper = Paper(name="p1", text="Just a plain paragraph without any math.")
    chunks = math_aware_split([paper], chunk_size=500, chunk_overlap=0)

    assert chunks[0].metadata["block_type"] == "text"
