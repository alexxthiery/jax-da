"""Documentation: links resolve, every doc page is reachable from the README."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)#\s]+)")


def test_relative_links_resolve():
    for md in [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")]:
        for target in LINK.findall(md.read_text()):
            if not target.startswith("http"):
                assert (md.parent / target).exists(), f"{md.name}: broken link {target}"


def test_every_doc_page_is_linked_from_the_readme():
    readme = (ROOT / "README.md").read_text()
    missing = [p.name for p in (ROOT / "docs").glob("*.md") if f"docs/{p.name}" not in readme]
    assert not missing, f"README does not link: {missing}"


def test_every_readme_python_block_runs():
    blocks = re.findall(r"```python\n(.*?)```", (ROOT / "README.md").read_text(), flags=re.S)
    assert len(blocks) >= 2
    for i, block in enumerate(blocks):
        exec(compile(block, f"README python block {i}", "exec"), {})


def test_no_escaped_underscores_in_markdown_math():
    """GitHub applies markdown escapes inside inline math, so `\\_` reaches MathJax as `_` in text mode.

    `$\\text{in\\_dim}$` then fails with "'_' allowed only in math mode"; write such names in code
    spans instead.
    """
    offenders = []
    for md in [*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")]:
        text = re.sub(r"```.*?```", "", md.read_text(), flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)
        offenders += [f"{md.name}: {line.strip()[:60]}" for line in text.splitlines() if "\\_" in line]
    assert not offenders, offenders
