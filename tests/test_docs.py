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


def test_readme_quick_start_runs():
    text = (ROOT / "README.md").read_text()
    block = text.split("## Quick start", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(block, "README quick start", "exec"), {})
