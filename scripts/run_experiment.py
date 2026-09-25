"""实验入口：对比 baseline 与 math_aware 的检索效果，可评估、可问答。

用法（在项目根目录 math_rag/ 下）：
    python scripts/run_experiment.py                  # 跑对比实验，输出 JSON
    python scripts/run_experiment.py --eval           # 额外在标注集上算 Recall@k / MRR / nDCG
    python scripts/run_experiment.py --with-deepseek  # 检索后调用 DeepSeek 生成答案
    python scripts/run_experiment.py --agent          # 交互式问答（react：LLM 自主调用工具）
    python scripts/run_experiment.py --agent --agent-mode simple  # 固定流水线问答
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 允许不安装包也能直接运行；pip install -e . 之后这句可省略
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 只保留无重依赖的模块；langchain/torch 相关模块改为"延迟导入"，
# 这样 --dry-run 在未安装任何依赖的机器上也能先检查论文。
from math_rag.config import load_config  # noqa: E402
from math_rag.latex_loader import load_math_papers  # noqa: E402
from math_rag.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("run_experiment")


def _format_docs(docs) -> list[dict]:
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]


def run_compare(papers, cfg, embeddings, questions, use_deepseek: bool, do_eval: bool) -> None:
    # 延迟导入：这些模块依赖 langchain/torch，只有真正跑实验时才需要
    from math_rag.evaluation import evaluate_retriever, load_ground_truth
    from math_rag.llm import ask_deepseek_with_sources
    from math_rag.retrieval import build_dense_retriever, build_hybrid_retriever, build_vector_db
    from math_rag.splitting import baseline_split, math_aware_split

    base_chunks = baseline_split(papers, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    exp_chunks = math_aware_split(papers, cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
    logger.info("基线组 %d 块，实验组 %d 块", len(base_chunks), len(exp_chunks))

    base_db = build_vector_db(base_chunks, cfg.paths.persist_dir / "base", "baseline", embeddings)
    exp_db = build_vector_db(exp_chunks, cfg.paths.persist_dir / "exp", "math_aware", embeddings)

    if cfg.retrieval.hybrid:
        base_retriever = build_hybrid_retriever(base_db, base_chunks, cfg.retrieval)
        exp_retriever = build_hybrid_retriever(exp_db, exp_chunks, cfg.retrieval)
    else:
        base_retriever = build_dense_retriever(base_db, cfg.retrieval, len(base_chunks))
        exp_retriever = build_dense_retriever(exp_db, cfg.retrieval, len(exp_chunks))

    result_log = []
    for question in questions:
        base_docs = base_retriever.invoke(question)
        exp_docs = exp_retriever.invoke(question)

        item = {
            "question": question,
            "baseline_retrieve": _format_docs(base_docs),
            "math_aware_retrieve": _format_docs(exp_docs),
        }
        if use_deepseek:
            item["baseline_deepseek_answer"] = ask_deepseek_with_sources(question, base_docs, cfg.llm)
            item["math_aware_deepseek_answer"] = ask_deepseek_with_sources(question, exp_docs, cfg.llm)
        result_log.append(item)
        logger.info("完成问题：%s", question)

    if do_eval:
        if cfg.paths.ground_truth is None:
            logger.warning("未配置 ground_truth，跳过评估。")
        else:
            gt = load_ground_truth(cfg.paths.ground_truth)
            result_log.append({
                "evaluation": {
                    "baseline": evaluate_retriever(base_retriever, gt, cfg.retrieval.k),
                    "math_aware": evaluate_retriever(exp_retriever, gt, cfg.retrieval.k),
                }
            })

    cfg.paths.log_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.log_file.write_text(json.dumps(result_log, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("结果已保存：%s", cfg.paths.log_file.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="数学论文 RAG 对比实验")
    parser.add_argument("--with-deepseek", action="store_true", help="检索后调用 DeepSeek 生成答案")
    parser.add_argument("--agent", action="store_true", help="启动交互式问答 Agent（默认 react 模式）")
    parser.add_argument(
        "--agent-mode",
        choices=["react", "simple"],
        default="react",
        help="Agent 模式：react=LLM 自主调用工具（默认）；simple=固定流水线（检索一次+生成一次）",
    )
    parser.add_argument("--eval", action="store_true", help="在标注集上计算 Recall@k / MRR / nDCG")
    parser.add_argument("--config", default=None, help="配置文件路径（默认项目根目录 config.yaml）")
    parser.add_argument("--dry-run", action="store_true", help="只加载论文并打印统计，不下载嵌入模型")
    parser.add_argument("--limit", type=int, default=None, help="只使用前 N 篇论文（第一次试跑建议 --limit 5）")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    papers = load_math_papers(cfg.paths.papers_dir)

    if args.limit is not None:
        papers = papers[: args.limit]
        logger.info("按 --limit 只使用前 %d 篇论文试跑", len(papers))

    if args.dry_run:
        total_chars = sum(len(p.text) for p in papers)
        print("\n===== 论文加载检查（不下载模型、不建向量库） =====")
        for p in papers:
            est = max(1, len(p.text) // 800)
            print(f"  - {p.name}: {len(p.text)} 字符，约 {est} 个 chunk")
        print(f"\n共 {len(papers)} 篇论文，{total_chars} 字符，预计约 {total_chars // 800} 个 chunk。")
        print("检查没问题的话，下一步小规模试跑：")
        print("  python scripts\\run_experiment.py --limit 5 --eval")
        return

    # 延迟导入：这里才会加载 torch / langchain（首次运行还会下载嵌入模型）
    from math_rag.embeddings import build_embeddings

    embeddings = build_embeddings(cfg.embedding)

    # 语义完整的测试问题（比裸关键词更能体现检索质量）
    questions = [
        "What is the p-Laplace operator and how is it defined?",
        "How is the first eigenvalue of the p-Laplacian characterized?",
        "What is the first eigenvalue under Neumann boundary conditions?",
        "What is the geodesic equation on a Riemannian manifold?",
    ]

    if args.agent:
        from math_rag.agent import run_agent  # 延迟导入

        run_agent(papers, cfg, embeddings, mode=args.agent_mode)
    else:
        run_compare(papers, cfg, embeddings, questions, args.with_deepseek, args.eval)


if __name__ == "__main__":
    main()
