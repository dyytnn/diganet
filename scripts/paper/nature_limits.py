#!/usr/bin/env python3
"""Enforce the Nature Methods *Analysis* limits on paper_nature/main.tex mechanically.

Analysis limits: abstract <=150 words and unreferenced; main text 3 000 words (5 000 at editorial
discretion) excluding abstract, Methods, references and figure legends; <=6 display items; ~50 references.
Exits non-zero if a hard limit is breached, so scripts/reproduce_paper.sh --nature fails loudly.

    python scripts/paper/nature_limits.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper_nature/main.tex"
BBL = ROOT / "paper_nature/main.bbl"
SOFT, HARD, ABSTRACT, ITEMS, REFS = 3000, 5000, 150, 6, 50


def strip(tex: str) -> str:
    tex = re.sub(r"\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}", "", tex, flags=re.S)
    return re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", tex)


def main() -> None:
    t = TEX.read_text()
    body = t[t.index("\\maketitle") + len("\\maketitle") : t.index("\\section{Methods}")]
    words = len(strip(body).split())
    abstract = t[t.index("\\abstract{") + len("\\abstract{") :]
    abstract = abstract[: abstract.index("}\n\n\\keywords")]
    n_abstract = len(strip(abstract).split())
    # only main-text floats count against the limit; Extended Data is exempt
    main_only = t[: t.index("\\section*{Extended Data}")] if "\\section*{Extended Data}" in t else t
    items = main_only.count("\\begin{figure") + main_only.count("\\begin{table")
    ed = t.count("\\begin{figure") + t.count("\\begin{table") + t.count("\\begin{sidewaystable") - items
    refs = BBL.read_text().count("\\bibitem") if BBL.exists() else -1

    def line(name: str, value: int, limit: int, hard: bool = True) -> bool:
        ok = value <= limit
        flag = "OK " if ok else ("FAIL" if hard else "over")
        print(f"  {name:<22s} {value:5d}   limit {limit:<5d} {flag}")
        return ok or not hard

    print("Nature Methods Analysis limits:")
    good = [
        line("main text (counted)", words, HARD),
        line("abstract words", n_abstract, ABSTRACT),
        line("display items", items, ITEMS),
        line("references", refs, REFS, hard=False),
    ]
    print(f"  {'extended data items':<22s} {ed:5d}   (exempt)")
    if words > SOFT:
        print(f"  note: main text exceeds the {SOFT}-word guideline; {HARD} needs editorial discretion")
    if "\\cite" in abstract:
        print("  FAIL  abstract contains a citation; Analysis abstracts are unreferenced")
        good.append(False)
    codes = re.findall(r"\b(S\d\d|V2_\d|L\d\d|B\d+)\b", strip(t))
    if codes:
        print(f"  FAIL  internal experiment codes in the text: {sorted(set(codes))}")
        good.append(False)
    sys.exit(0 if all(good) else 1)


if __name__ == "__main__":
    main()
