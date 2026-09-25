# math-rag：数学论文检索增强问答（RAG）

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/Genuine-Cl/math-rag/actions/workflows/ci.yml/badge.svg)

针对 LaTeX 数学论文的检索增强问答系统，包含：LaTeX 论文加载与清洗、两种切分策略对比、
MMR + BM25 混合检索、离线评估（Recall@k / MRR / nDCG），以及基于证据的 DeepSeek 问答 Agent。

> 这是一份面向面试的项目版本：每个设计决策都写明了"为什么"，方便你在面试时讲清楚。
>
> `data/papers/` 下的 `.tex` 文件为第三方 arXiv 论文源文件，仅用于检索实验与评估。

## 目录结构

```
math_rag/
├── config.yaml                 # 所有可调参数（配置与逻辑分离）
├── pyproject.toml              # 可安装的 Python 包
├── requirements.txt
├── src/math_rag/
│   ├── config.py               # 配置加载（dataclass + 环境变量覆盖）
│   ├── latex_loader.py         # 递归展开 \input、清洗 LaTeX、source 溯源
│   ├── splitting.py            # baseline 切分 vs 数学感知（公式原子化）切分
│   ├── embeddings.py           # 嵌入模型工厂（bge-m3，多语言）
│   ├── retrieval.py            # Chroma 向量库 + MMR/BM25 混合检索
│   ├── llm.py                  # DeepSeek 问答（基于证据 + 错误处理）
│   ├── tools.py                # Agent 工具层（search/get_chunk/list_papers）
│   ├── evaluation.py           # Recall@k / MRR / nDCG
│   └── agent.py                # 交互式 Agent：simple（固定流水线）/ react（工具调用循环）
├── scripts/run_experiment.py   # 实验入口 CLI
├── tests/                      # 单元测试（pytest）
└── data/
    ├── papers/                 # 示例论文（.tex）
    └── eval/ground_truth.json  # 检索评估标注集
```

## 快速开始

> 新手请先读《使用指南.md》（一步步的中文教程，含国内镜像加速）。

```bash
cd math_rag

# 0. 先检查论文能否被加载（不下载模型，很快）
python scripts/run_experiment.py --dry-run

# 1. 安装依赖（首次运行会下载嵌入模型，默认约 420MB，需联网）
pip install -e ".[dev]"

# 2. 小规模试跑（只用前 5 篇，确认链路能通）
python scripts/run_experiment.py --limit 5 --eval

# 3. 全量对比实验（输出到 logs/experiment.json）
python scripts/run_experiment.py --eval

# 4. 交互式问答 Agent（react：LLM 自主决定调用哪些工具，默认）
python scripts/run_experiment.py --agent

# 4b. 固定流水线问答（检索一次 + 生成一次，方便和 react 对比）
python scripts/run_experiment.py --agent --agent-mode simple

# 5. 检索后调用 DeepSeek（需设置环境变量）
#    Windows:  set DEEPSEEK_API_KEY=sk-xxx
#    Linux/Mac: export DEEPSEEK_API_KEY=sk-xxx
python scripts/run_experiment.py --with-deepseek

# 6. 跑单元测试
pytest
```

## 设计决策（面试讲解要点）

### 1. 为什么用 MMR 而不是纯相似度检索？
纯相似度（top-k 余弦）容易把同一个语义片段反复捞回来，浪费 k 个名额。
MMR（Maximal Marginal Relevance）在"与 query 相关"和"与已选结果不重复"之间取平衡，
由 `lambda_mult` 控制：越大越重相关性，越小越重多样性。这里取 0.7，偏相关性同时保留一定多样性。

### 2. `lambda_mult` / `chunk_size` / `fetch_k` 怎么定的？
- 这些是超参，没有"唯一正确答案"，正确做法是**做消融实验**（在 config.yaml 里改几档对比）。
- `fetch_k` 是 MMR 的候选池大小，会先用大候选池捞出 `fetch_k` 个，再从中选 `k` 个多样结果。
- 代码里 `fetch_k` 会被钳制为 `min(fetch_k, 语料块数)`，避免语料很小时 MMR 报错。

### 3. 数学感知切分到底做了什么？
普通切分会把 `\begin{equation}...\end{equation}` 拦腰切断，破坏公式完整性。
本实现先把公式环境整体替换成占位符 → 切分 → 再还原，**保证公式是原子单元**。
相比"只在分隔符里塞 `\end{equation}`"的做法，它更可靠（后者仍可能切断长公式）。
每个 chunk 标记 `block_type`（formula_context / text），便于后续分析。

### 4. 为什么加 BM25 混合检索？
bge 这类稠密向量对 LaTeX 数学符号（`\Delta_p`、`\nabla`、`\int`）的表示能力有限，
纯向量检索容易漏掉"符号/关键词精确匹配"的片段。BM25 是稀疏检索，天然擅长精确词项匹配。
两者用 RRF 加权融合，兼顾"语义相近"和"符号精确命中"，是工业界常用的混合检索方案。

### 5. 为什么默认用小模型而不是 bge-m3？
Agent 支持中英文问题，`bge-small-en-v1.5` 是纯英文模型，中文 query 会掉质量，所以不选它。
默认 `paraphrase-multilingual-MiniLM-L12-v2`：多语言（论文有英/法/西/意文也能处理）、
模型小（约 420MB）、CPU 上快，适合 50 篇级别的本地实验。
`BAAI/bge-m3` 是质量升级选项（约 2GB，CPU 慢 4~5 倍），跑通流程、有多余算力时再换。

### 6. 检索结果如何溯源？
`load_math_papers` 返回带论文名的 `Paper`，切分时把论文名写进 metadata 的 `source` 字段。
这样每一条检索结果都能显示"来自哪篇论文、哪个 chunk"，用户可回原文核验——这是 RAG 的基本要求。

### 7. 怎么证明实验组更好？
单看两组检索结果不算"对比实验"。这里补充了离线评估：标注集 `data/eval/ground_truth.json`
（每个问题标注"答案在哪些论文"），用 `--eval` 计算 Recall@k / MRR / nDCG。
进一步可加 LLM-as-judge 对生成答案打分，或做 chunk_size / lambda_mult 的消融。

### 8. Agent 的 react 模式和 simple 模式有什么区别？
simple 模式是固定流水线：检索一次 → 把 top-k 塞进提示词 → 生成一次，LLM 没有决策权，
这正是"RAG"而不是"Agent"。react 模式把决策权交给模型：每轮 LLM 自己决定
"要不要调工具、调哪个、用什么参数"，代码只负责执行工具并把纯文本结果喂回模型，
直到模型不再调用工具、直接给出最终答案。这个循环（模型决策 → 执行 → 反馈）
就是 Agent 与 RAG 的本质区别。

### 9. Agent 的工具层是怎么设计的？
工具 = 函数 + 说明书：每个工具有 name/description/parameters（JSON Schema），
通过 `bind_tools` 传给 DeepSeek 的 function calling。三个工具各司其职——
`search_papers`（混合检索）、`get_chunk`（按论文+chunk 取原文，深挖上下文）、
`list_papers`（先摸清语料里有什么再决定检索方向）。设计要点：
- 工具与 Chroma/检索器解耦，只要求传入带 `invoke()` 的对象，因此可用假检索器做单元测试；
- 工具异常一律转成文本错误信息喂回模型（而不是抛出崩溃），模型可自我纠错、换参数重试；
- 每条 ToolMessage 的 `tool_call_id` 必须与模型生成的工具调用 id 一一对应，否则模型会串上下文；
- 循环有护栏：`agent.max_iterations` 限制最大轮数，防止死循环和控制 token 成本；
  预算快用完时注入"必须收尾"的系统提醒，超预算仍不收敛则把已收集证据汇总、
  用不绑定工具的裸模型强制生成答案（保证每个问题一定有回答，而不是空手强制停止）。

react 模式换来的是 single-shot 拿不到的能力：多跳检索（先搜到定理名，再拿定理名搜证明）、
检索无果时自动改写 query 重试、证据不足时先 list_papers 再决定查哪篇。

## 已知边界

- LaTeX 清洗用正则，不覆盖嵌套花括号的标题、极复杂宏定义；生产级可用 `latex2text` 或 AST 解析。
- 语料规模到百万级时，应换 HNSW 索引 / 分片 / 加 reranker 重排。
- `\input` 展开只处理相对路径与 `.tex` 后缀补全，不解析 `\includeonly` 过滤。

## 常见追问自测

1. MMR 和纯相似度、以及 BM25 的差异是什么？
2. `lambda_mult` 变大变小分别会发生什么？
3. chunk 800 对公式证明是否合适？会不会切断推导？
4. 语料变多后如何扩展？稠密检索的召回怎么补？
5. 如何度量"引用忠实度"（生成答案是否忠于检索片段）？
6. Agent 的 react 模式和 simple 模式（普通 RAG）本质区别是什么？
7. 工具报错时为什么要把异常转成文本喂回模型，而不是直接抛出？
8. `max_iterations` 没有上限会有什么风险？
