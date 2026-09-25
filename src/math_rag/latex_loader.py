"""LaTeX 论文加载：递归展开 \\input/\\include，并清洗出正文。

相比原版改进：
1. 返回带论文名的 Paper 对象，检索结果能溯源到具体论文；
2. \\input 支持带可选参数 \\input[opts]{file}、\\subfile，缺失文件只告警不静默吞掉；
3. 清理逻辑覆盖更多引用命令（\\parencite/\\textcite/\\cref/\\autoref 等）。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """一篇论文：名字（用于溯源）+ 清洗后的正文。"""
    name: str
    text: str


# 匹配 \input{file} / \include{file} / \subfile{file}，可选参数 [opts]
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)(?:\[[^\]]*\])?\{([^}]+)\}")


def read_tex_with_inputs(tex_path: Path, visited: set[Path] | None = None) -> str:
    """读取主 tex 文件并递归展开 \\input{} / \\include{} / \\subfile{}。

    - 自动补 .tex 后缀；
    - 用 visited 集合防止循环引用；
    - 子文件缺失时记录告警并跳过，而不是静默返回空串。
    """
    if visited is None:
        visited = set()

    tex_path = tex_path.resolve()
    if tex_path in visited or not tex_path.exists():
        return ""
    visited.add(tex_path)

    text = tex_path.read_text(encoding="utf-8", errors="ignore")

    def _replace(match: re.Match) -> str:
        relative = match.group(1).strip()
        child = tex_path.parent / relative
        if child.suffix.lower() != ".tex":
            child = child.with_suffix(".tex")
        if not child.exists():
            logger.warning("引用的子文件不存在，已跳过：%s（来自 %s）", child, tex_path.name)
            return ""
        return read_tex_with_inputs(child, visited)

    return _INPUT_RE.sub(_replace, text)


def clean_latex_for_retrieval(latex_text: str) -> str:
    """删除导言区/注释/参考文献，保留正文与数学命令（如 \\Delta、\\int、\\frac）。"""
    # 1. 去掉注释（% 到行尾，保留 \% 这类转义）
    latex_text = re.sub(r"(?<!\\)%.*", "", latex_text)

    # 2. 只保留 document 环境之间的正文
    match = re.search(
        r"\\begin\{document\}(.*?)\\end\{document\}",
        latex_text,
        flags=re.DOTALL,
    )
    if match:
        latex_text = match.group(1)

    # 3. 删除参考文献环境，避免标题造成假命中
    latex_text = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        "",
        latex_text,
        flags=re.DOTALL,
    )
    latex_text = re.sub(r"\\(?:bibliography|addbibresource)\{[^{}]*\}", "", latex_text)
    latex_text = re.sub(r"\\printbibliography", "", latex_text)

    # 4. 删除不携带正文语义的引用 / 交叉引用 / 标签
    latex_text = re.sub(
        r"\\(?:cite|citet|citep|parencite|textcite|nocite|label|ref|eqref|pageref|autoref|cref|Cref)\*?\{[^{}]*\}",
        "",
        latex_text,
    )

    # 5. 标题命令保留为纯文本标题
    latex_text = re.sub(
        r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^{}]*)\}",
        r"\n\n\1\n",
        latex_text,
    )

    # 6. 题目/作者信息：标题当正文，其余命令删除
    latex_text = re.sub(r"\\title\{([^{}]*)\}", r"\n\n\1\n", latex_text)
    latex_text = re.sub(r"\\(?:author|date)\{[^{}]*\}", "", latex_text)
    latex_text = re.sub(r"\\maketitle", "", latex_text)

    # 7. 处理空白符号与常见排版命令，压缩空行
    latex_text = latex_text.replace("~", " ")
    latex_text = re.sub(
        r"\\(?:noindent|par|smallskip|medskip|bigskip|vspace|hspace)(?:\{[^{}]*\})?",
        " ",
        latex_text,
    )
    latex_text = re.sub(r"\n{3,}", "\n\n", latex_text)

    return latex_text.strip()


def load_math_papers(papers_dir: str | Path) -> list[Paper]:
    """扫描目录下所有含 \\documentclass 的 .tex 主文件，返回 Paper 列表。

    论文名取主文件所在子目录的相对路径（如 sample_paper），用于检索溯源。
    """
    root = Path(papers_dir)
    if not root.exists():
        raise FileNotFoundError(f"找不到论文目录：{root.resolve()}")

    main_tex_files = []
    skipped_files = []
    for tex_path in sorted(root.rglob("*.tex")):
        source = tex_path.read_text(encoding="utf-8", errors="ignore")
        if r"\documentclass" in source:
            main_tex_files.append(tex_path)
        else:
            skipped_files.append(tex_path)

    if not main_tex_files:
        raise ValueError(f"在 {root.resolve()} 下没有找到含 \\documentclass 的 .tex 主文件。")

    for tex_path in skipped_files:
        logger.info("跳过（不含 \\documentclass，多为插图导出件或非主文件）：%s", tex_path.name)

    papers: list[Paper] = []
    for tex_path in main_tex_files:
        raw = read_tex_with_inputs(tex_path)
        clean = clean_latex_for_retrieval(raw)

        if tex_path.parent == root:
            name = tex_path.stem
        else:
            name = tex_path.parent.relative_to(root).as_posix()

        if clean:
            papers.append(Paper(name=name, text=clean))
            logger.info("已加载论文：%s，字符数 %d", name, len(clean))
        else:
            logger.warning("跳过：%s（未得到有效正文）", tex_path)

    if not papers:
        raise ValueError("没有读取到有效的 LaTeX 正文。")

    return papers
