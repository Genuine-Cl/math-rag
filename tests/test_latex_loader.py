"""LaTeX 加载与清洗的单元测试。"""
from __future__ import annotations

from math_rag.latex_loader import clean_latex_for_retrieval, load_math_papers, read_tex_with_inputs


def test_clean_removes_preamble_and_keeps_body():
    tex = r"""\documentclass{article}
\usepackage{amsmath}
\begin{document}
\section{Introduction}
The p-Laplace operator is $\Delta_p u$.
\end{document}
"""
    out = clean_latex_for_retrieval(tex)
    assert "documentclass" not in out
    assert "usepackage" not in out
    assert "Introduction" in out
    assert "p-Laplace" in out


def test_clean_removes_comments_and_citations():
    tex = r"""\begin{document}
A well-known result \cite{foo2020}. % this is a comment
\bibliography{refs}
\end{document}
"""
    out = clean_latex_for_retrieval(tex)
    assert "cite" not in out
    assert "comment" not in out
    assert "refs" not in out


def test_clean_removes_bibliography_environment():
    tex = r"""\begin{document}
Body text.
\begin{thebibliography}{1}
\bibitem{x} A reference title that should not match.
\end{thebibliography}
\end{document}
"""
    out = clean_latex_for_retrieval(tex)
    assert "Body text." in out
    assert "reference title" not in out


def test_read_tex_with_inputs_expands_subfile(tmp_path):
    main = tmp_path / "main.tex"
    sub = tmp_path / "section1.tex"
    sub.write_text(r"\section{Results} Some text.", encoding="utf-8")
    main.write_text(
        r"\documentclass{article}\begin{document}\input{section1}\end{document}",
        encoding="utf-8",
    )
    out = read_tex_with_inputs(main)
    assert "Some text." in out


def test_load_math_papers_returns_named_papers(tmp_path):
    paper_dir = tmp_path / "sample_paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(
        r"\documentclass{article}\begin{document}Hello world.\end{document}",
        encoding="utf-8",
    )
    papers = load_math_papers(tmp_path)
    assert len(papers) == 1
    assert papers[0].name == "sample_paper"
    assert "Hello world." in papers[0].text
