"""配置加载：把 config.yaml 转成类型安全的数据类，并支持环境变量覆盖。

面试可讲点：
- 用 dataclass 而不是到处读 dict，IDE 有补全、拼错字段会直接报错；
- 路径统一以项目根目录解析，避免"相对当前工作目录"的隐患。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
except ImportError:  # 依赖未安装时，--dry-run 仍可检查论文
    yaml = None

logger = logging.getLogger(__name__)

# src/math_rag/config.py -> parents[0]=math_rag, [1]=src, [2]=项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    papers_dir: Path
    persist_dir: Path
    log_file: Path
    ground_truth: Path | None = None


@dataclass
class EmbeddingConfig:
    # 与 config.yaml 保持一致；实际值以 config.yaml 为准
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True
    batch_size: int = 32


@dataclass
class ChunkingConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100


@dataclass
class RetrievalConfig:
    search_type: str = "mmr"
    k: int = 3
    fetch_k: int = 12
    lambda_mult: float = 0.7
    hybrid: bool = True
    dense_weight: float = 0.6


@dataclass
class LLMConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    timeout: int = 60
    max_retries: int = 2


@dataclass
class AgentConfig:
    # react 模式的最大工具调用轮数：防止模型陷入死循环、控制成本
    max_iterations: int = 6


@dataclass
class Config:
    paths: PathsConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    agent: AgentConfig = field(default_factory=AgentConfig)


def _resolve(p: str) -> Path:
    """相对路径一律按项目根目录解析。"""
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _default_config() -> Config:
    """依赖尚未安装时的兜底配置（只够 --dry-run 检查论文用）。"""
    return Config(
        paths=PathsConfig(
            papers_dir=PROJECT_ROOT / "data" / "papers",
            persist_dir=PROJECT_ROOT / "chroma_db",
            log_file=PROJECT_ROOT / "logs" / "experiment.json",
            ground_truth=None,
        ),
        embedding=EmbeddingConfig(),
        chunking=ChunkingConfig(),
        retrieval=RetrievalConfig(),
        llm=LLMConfig(),
    )


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"

    if yaml is None:
        # 未安装 pyyaml：说明依赖还没装，用默认配置让 --dry-run 能跑
        if path is not None:
            raise RuntimeError("未安装 pyyaml，无法读取指定配置文件。请先 pip install pyyaml。")
        logger.warning("未安装 pyyaml，使用内置默认配置（仅适用于 --dry-run 检查）")
        return _default_config()

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path.resolve()}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    paths_raw = raw["paths"]
    paths = PathsConfig(
        papers_dir=_resolve(paths_raw["papers_dir"]),
        persist_dir=_resolve(paths_raw["persist_dir"]),
        log_file=_resolve(paths_raw["log_file"]),
        ground_truth=_resolve(paths_raw["ground_truth"]) if paths_raw.get("ground_truth") else None,
    )

    embedding = EmbeddingConfig(**raw.get("embedding", {}))
    chunking = ChunkingConfig(**raw.get("chunking", {}))
    retrieval = RetrievalConfig(**raw.get("retrieval", {}))
    llm = LLMConfig(**raw.get("llm", {}))
    llm.model = os.getenv("DEEPSEEK_MODEL", llm.model)
    agent = AgentConfig(**raw.get("agent", {}))

    return Config(paths=paths, embedding=embedding, chunking=chunking, retrieval=retrieval, llm=llm, agent=agent)
