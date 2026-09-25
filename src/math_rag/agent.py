"""交互式问答 Agent：simple（固定流水线）与 react（工具调用循环）两种模式。

面试可讲点：
- simple 模式是"检索一次 -> 生成一次"的固定流水线（经典 RAG），
  LLM 没有决策权，只被调用一次；
- react 模式把决策权交给模型：LLM 每轮自己决定"要不要调工具、调哪个、
  用什么参数"，代码只负责执行工具并把结果喂回模型，直到模型给出最终答案。
  这就是 Agent 与 RAG 的本质区别；
- 循环有护栏：最大迭代次数上限（agent.max_iterations），工具异常转成文本
  喂回模型而不是崩掉，最终无工具调用即视为回答结束；
- 保留了工具调用历史（LangChain 消息列表），每一轮模型都看到完整上下文，
  因此支持"先列论文 -> 再检索 -> 深挖某条证据"的多步推理。
"""
from __future__ import annotations

import logging
import os

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from .config import Config
from .llm import _build_llm, ask_deepseek_with_sources
from .retrieval import build_dense_retriever, build_hybrid_retriever, build_vector_db
from .splitting import math_aware_split
from .tools import build_agent_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是数学论文问答助手。你可以调用工具检索论文库来回答问题。

工作方式：
1. 先想清楚：用户问题需要哪些信息？论文库里可能有哪些论文？
   （可先调用 list_papers 了解语料覆盖范围。）
2. 用 search_papers 检索相关片段；必要时改写关键词多检索几次，
   或调用 get_chunk 深挖某条证据的完整上下文。
   注意：同一个问题一般检索 2~3 轮就该综合作答，不要无限换关键词检索；
   证据已覆盖问题要点时立即作答。
3. 基于检索到的证据作答；检索足够后直接给出最终答案，不要再调用工具。

回答规则：
- 严格只依据检索到的论文片段，不编造定理、条件或结论；
- 证据不足时明确说"当前文献库没有足够证据回答该问题"；
- 用中文回答；数学符号和论文术语可保留英文或 LaTeX；
- 回答末尾标注使用的证据，例如：[来源 xxx.tex | chunk 3]。"""


def _build_db_and_retriever(papers, cfg: Config, embeddings):
    """建立（或复用）Agent 向量库并返回检索器，simple 与 react 两种模式共用。"""
    chunks = math_aware_split(papers, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    agent_db_dir = cfg.paths.persist_dir / "agent"

    if agent_db_dir.exists():
        # 复用已建好的索引，省去每次启动的几分钟重建；论文更新后删除该目录即可强制重建
        logger.info("复用已有 Agent 向量库：%s", agent_db_dir)
        db = Chroma(
            persist_directory=str(agent_db_dir),
            collection_name="math_paper_agent",
            embedding_function=embeddings,
        )
    else:
        db = build_vector_db(
            chunks=chunks,
            persist_dir=agent_db_dir,
            collection_name="math_paper_agent",
            embeddings=embeddings,
        )

    if cfg.retrieval.hybrid:
        retriever = build_hybrid_retriever(db, chunks, cfg.retrieval)
    else:
        retriever = build_dense_retriever(db, cfg.retrieval, len(chunks))
    return chunks, retriever


def _run_simple_loop(retriever, cfg: Config) -> None:
    """simple 模式：检索一次 -> 生成一次，无决策循环（经典 RAG）。"""
    while True:
        try:
            question = input("你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in {"exit", "quit", "退出"}:
            print("Agent 已退出。")
            break
        if not question:
            continue

        docs = retriever.invoke(question)
        if not docs:
            print("\n未检索到相关文献片段。\n")
            continue

        print("\n检索到的证据：")
        for i, doc in enumerate(docs, start=1):
            print(
                f"[证据 {i}] source={doc.metadata.get('source')} "
                f"chunk={doc.metadata.get('chunk_id')} "
                f"type={doc.metadata.get('block_type', 'text')}"
            )

        print("\nDeepSeek 回答：")
        print(ask_deepseek_with_sources(question, docs, cfg.llm))
        print("\n" + "=" * 70 + "\n")


def _force_final_answer(question: str, messages, cfg: Config) -> None:
    """模型超预算仍不收敛时：汇总已收集的全部工具结果，用不绑定工具的裸调用强制作答。"""
    evidence = "\n\n---\n\n".join(
        str(m.content) for m in messages if isinstance(m, ToolMessage)
    ) or "（未收集到任何证据）"
    prompt = (
        "你在检索过程中收集到以下论文证据：\n\n"
        f"{evidence}\n\n"
        f"用户问题：{question}\n\n"
        "请仅依据以上证据用中文回答；证据不足就明确说明。"
        "回答末尾标注使用的证据来源（论文名和 chunk 编号）。"
    )
    # 不 bind_tools：模型拿不到工具，从根上杜绝继续调用工具
    final_llm = _build_llm(cfg.llm)
    try:
        answer = final_llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as exc:  # noqa: BLE001 - 收尾失败也要给用户一个交代
        logger.error("强制收尾调用失败：%s", exc)
        answer = f"（检索轮数已用尽，且收尾调用失败：{exc}）"
    print(f"已达到最大检索轮数 {cfg.agent.max_iterations}，强制综合已有证据作答：")
    print(answer)
    print("\n" + "=" * 70 + "\n")


def _run_react_loop(papers, retriever, chunks, cfg: Config) -> None:
    """react 模式：LLM 自主决定调用哪些工具，直到给出最终答案。"""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("未设置 DEEPSEEK_API_KEY，无法启动 react 模式。请先设置环境变量。")
        return

    tools = build_agent_tools(papers, retriever, chunks)
    llm = _build_llm(cfg.llm).bind_tools([t.to_openai_schema() for t in tools])
    tools_by_name = {t.name: t for t in tools}

    while True:
        try:
            question = input("你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in {"exit", "quit", "退出"}:
            print("Agent 已退出。")
            break
        if not question:
            continue

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
        print()

        # ---- Agent 循环：模型决策 -> 执行工具 -> 结果喂回，直到无工具调用 ----
        for step in range(1, cfg.agent.max_iterations + 1):
            response = llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                print("DeepSeek 回答：")
                print(response.content)
                print("\n" + "=" * 70 + "\n")
                break

            messages.append(response)  # 保留模型本轮的工具调用决策
            for call in tool_calls:
                name, args, call_id = _parse_tool_call(call)
                tool = tools_by_name.get(name)
                print(f"[第 {step} 步] 调用工具 {name}{args}")
                result = (
                    tool.execute(args) if tool else f"未知工具 {name}，可用工具：{sorted(tools_by_name)}"
                )
                print(f"         → {result[:120]}{'…' if len(result) > 120 else ''}")
                # ToolMessage 的 tool_call_id 必须与模型生成的调用 id 对应
                messages.append(ToolMessage(content=result, tool_call_id=call_id))
                logger.info("agent step %d 调用 %s(%s)，返回 %d 字符", step, name, args, len(result))

            # 预算即将用尽：明确提醒模型下一轮必须收尾，避免无限检索发散
            if step >= cfg.agent.max_iterations - 1:
                messages.append(
                    SystemMessage(
                        content="工具调用轮数即将用尽：请基于以上已检索到的证据直接给出最终答案"
                        "（用中文，末尾标注证据来源），不要再调用任何工具。"
                    )
                )
        else:
            # for-else：跑满轮数模型仍在调工具 → 强制综合已有证据作答，保证一定有答案
            _force_final_answer(question, messages, cfg)


def _parse_tool_call(call) -> tuple[str, dict, str]:
    """兼容不同 langchain 版本的 tool_call 结构：dict 或 (name, args, id) 元组。"""
    if isinstance(call, dict):
        return call.get("name", "?"), call.get("args") or {}, call.get("id") or "?"
    if isinstance(call, (tuple, list)) and len(call) >= 2:
        return call[0], call[1] or {}, (call[2] if len(call) > 2 else "?")
    return "?", {}, "?"


def run_agent(papers, cfg: Config, embeddings, mode: str = "react") -> None:
    """建立（或复用）Agent 向量库并启动交互式问答。

    mode="simple"：原固定流水线（检索一次 + 生成一次）；
    mode="react"：LLM 自主工具调用循环（默认）。
    """
    # react 模式离开密钥无法工作：在建向量库之前就拦下，避免白等几分钟。
    # simple 模式不在这里拦：它只在真正生成答案时才需要密钥（llm.py 里会提示）。
    if mode == "react" and not os.environ.get("DEEPSEEK_API_KEY"):
        print("未设置 DEEPSEEK_API_KEY，无法启动 react 模式。")
        print("请先设置后重试：set DEEPSEEK_API_KEY=sk-你的密钥")
        return

    chunks, retriever = _build_db_and_retriever(papers, cfg, embeddings)

    print("\n数学论文 Agent 已启动（输入 exit 退出）。\n")
    if mode == "react":
        _run_react_loop(papers, retriever, chunks, cfg)
    else:
        _run_simple_loop(retriever, cfg)
