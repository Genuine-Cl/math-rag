"""Agent 工具层：把检索能力封装成 LLM 可调用的工具。

面试可讲点：
- 工具 = 函数 + 说明书。name/description/parameters 就是给模型看的"使用说明书"，
  说明书写得清不清楚，直接决定模型会不会乱调；
- 每个工具返回"纯文本结果"，工具报错也要转成文本喂回模型（而不是把异常抛出去
  崩掉整个循环），这样模型能自己纠错、换个参数重试；
- 工具与 Chroma/检索器解耦：调用方传入带 invoke() 的对象即可，
  因此可以用假检索器直接做单元测试，不必起真实向量库。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.documents import Document

# 单次工具结果塞进上下文的最大字符数，防止把整篇论文灌进提示词
MAX_RESULT_CHARS = 4000


@dataclass(frozen=True)
class Tool:
    """一个可被 LLM 调用的工具：参数 schema + 执行函数。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str] = field(compare=False, repr=False)

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI function-calling 的 JSON Schema，供 llm.bind_tools 使用。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, args: dict[str, Any]) -> str:
        """执行工具；任何异常都转成文本错误信息，让模型有机会自我纠错。"""
        try:
            return self.func(**args)
        except Exception as exc:  # noqa: BLE001 - 工具错误必须喂回模型，不能崩循环
            return f"工具 {self.name} 调用失败：{exc}。请检查参数后重试。"


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…（结果过长已截断）"


def make_search_tool(retriever) -> Tool:
    """search_papers：语义/关键词混合检索（MMR + BM25，复用现有检索器）。

    注意：实际返回条数受检索器配置 k 的上限约束（MMR 先捞 fetch_k 再精选），
    所以 k 参数只作为"期望条数"，可能拿不到那么多——如实向模型说明即可。
    """

    def search_papers(query: str, k: int = 3) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "没有检索到任何相关片段，建议换更具体的关键词或改写问题。"
        docs = docs[: max(1, k)]
        lines = [f"共返回 {len(docs)} 条证据："]
        for i, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", "?")
            cid = doc.metadata.get("chunk_id", "?")
            lines.append(f"[证据 {i} | 来源 {src} | chunk {cid}] {doc.page_content}")
        return _truncate("\n\n".join(lines))

    return Tool(
        name="search_papers",
        description=(
            "在数学论文库中检索与给定问题最相关的论文片段（稠密向量+关键词混合检索）。"
            "适合找定义、定理、证明和公式。查询可用中文或英文，建议包含关键数学术语，"
            "例如 LaTeX 符号（如 p-Laplacian、第一特征值）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询，中英文皆可"},
                "k": {"type": "integer", "description": "期望返回的片段条数，默认 3"},
            },
            "required": ["query"],
        },
        func=search_papers,
    )


def make_get_chunk_tool(chunks: list[Document]) -> Tool:
    """get_chunk：按来源论文 + chunk 编号取原文，供模型深挖某条证据的上下文。"""
    index = {(c.metadata.get("source"), str(c.metadata.get("chunk_id"))): c for c in chunks}

    def get_chunk(source: str, chunk_id: str) -> str:
        doc = index.get((source, chunk_id))
        if doc is None:
            available = sorted({s for s, _ in index})
            return f"未找到 {source} 的 chunk {chunk_id}。可用的论文有：{available}"
        return f"[{source} | chunk {chunk_id}]\n{doc.page_content}"

    return Tool(
        name="get_chunk",
        description=(
            "根据来源论文文件名和 chunk 编号，取出某条已检索证据的原文。"
            "用于核实细节或阅读某条证据的完整上下文。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "论文文件名（与检索结果中的来源一致）"},
                "chunk_id": {"type": "string", "description": "chunk 编号，如 3"},
            },
            "required": ["source", "chunk_id"],
        },
        func=get_chunk,
    )


def make_list_papers_tool(papers) -> Tool:
    """list_papers：列出语料库中全部论文，让模型先摸清库里有什么再决定检索方向。"""

    def list_papers() -> str:
        return "\n".join(f"- {paper.name}" for paper in papers)

    return Tool(
        name="list_papers",
        description="列出论文库中包含的全部论文文件名，用于了解语料覆盖范围、判断从哪篇论文入手检索。",
        parameters={"type": "object", "properties": {}},
        func=list_papers,
    )


def build_agent_tools(papers, retriever, chunks: list[Document]) -> list[Tool]:
    """组装 Agent 的全部工具。"""
    return [
        make_search_tool(retriever),
        make_get_chunk_tool(chunks),
        make_list_papers_tool(papers),
    ]
