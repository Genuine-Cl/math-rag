"""数学论文 RAG 检索工具包。

核心能力：
- LaTeX 论文加载与清洗（递归展开 \\input、去导言区/参考文献）
- 基线切分 vs 数学感知切分（公式环境原子化，不被切断）
- MMR 稠密检索 + BM25 混合检索
- 检索评估（Recall@k / MRR / nDCG）
- 基于证据的 DeepSeek 问答 Agent
"""

__version__ = "0.2.0"
