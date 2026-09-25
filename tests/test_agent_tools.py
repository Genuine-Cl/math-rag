"""Agent 工具层的单元测试：用假检索器验证，不起真实向量库、不调真实 LLM。"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import patch

from math_rag.agent import _parse_tool_call, _run_react_loop
from math_rag.config import AgentConfig, Config
from math_rag.latex_loader import Paper
from math_rag.tools import make_get_chunk_tool, make_list_papers_tool, make_search_tool

# 与 math_rag.latex_loader.Document 无冲突：langchain 的 Document 只需这两个字段
from langchain_core.documents import Document


def _fake_chunk(text, source, chunk_id):
    return Document(page_content=text, metadata={"source": source, "chunk_id": chunk_id})


def test_search_tool_formats_evidence_with_source():
    docs = [_fake_chunk("内容A", "paperA.tex", 0), _fake_chunk("内容B", "paperB.tex", 2)]
    retriever = SimpleNamespace(invoke=lambda q: docs)
    tool = make_search_tool(retriever)

    result = tool.execute({"query": "第一特征值", "k": 2})

    assert "paperA.tex" in result and "paperB.tex" in result
    assert "内容A" in result and "内容B" in result


def test_search_tool_k_is_respected_as_cap():
    docs = [_fake_chunk(f"内容{i}", "p.tex", i) for i in range(5)]
    retriever = SimpleNamespace(invoke=lambda q: docs)
    tool = make_search_tool(retriever)

    result = tool.execute({"query": "q", "k": 2})

    assert "共返回 2 条证据" in result
    assert "内容3" not in result


def test_search_tool_empty_result_is_informative():
    retriever = SimpleNamespace(invoke=lambda q: [])
    tool = make_search_tool(retriever)

    result = tool.execute({"query": "q"})

    assert "没有检索到任何相关片段" in result


def test_get_chunk_tool_returns_exact_chunk():
    chunks = [_fake_chunk("目标内容", "paperA.tex", 3)]
    tool = make_get_chunk_tool(chunks)

    result = tool.execute({"source": "paperA.tex", "chunk_id": "3"})

    assert "目标内容" in result


def test_get_chunk_tool_missing_chunk_hints_available_papers():
    chunks = [_fake_chunk("x", "paperA.tex", 3)]
    tool = make_get_chunk_tool(chunks)

    result = tool.execute({"source": "paperA.tex", "chunk_id": "99"})

    assert "未找到" in result and "paperA.tex" in result


def test_list_papers_tool_enumerates_papers():
    papers = [Paper(name="a.tex", text=""), Paper(name="b.tex", text="")]
    tool = make_list_papers_tool(papers)

    result = tool.execute({})

    assert "a.tex" in result and "b.tex" in result


def test_parse_tool_call_supports_dict_and_tuple():
    assert _parse_tool_call({"name": "t", "args": {"k": 1}, "id": "id1"}) == ("t", {"k": 1}, "id1")
    assert _parse_tool_call(("t", {"k": 1}, "id1")) == ("t", {"k": 1}, "id1")


def test_react_loop_exits_without_api_key(capsys):
    """无 API Key 时 react 模式应立即退出而不是崩掉（不建 LLM、不检索）。"""
    cfg = Config.__new__(Config)
    cfg.agent = AgentConfig(max_iterations=3)

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DEEPSEEK_API_KEY", None)
        _run_react_loop([], None, [], cfg)

    assert "未设置 DEEPSEEK_API_KEY" in capsys.readouterr().out
