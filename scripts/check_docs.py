"""Check local relative Markdown links for missing targets.

Usage: python scripts/check_docs.py
This intentionally does not check remote URLs or Markdown anchors.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def check_docs(root: Path = ROOT) -> list[str]:
    """Return broken local Markdown links from README and docs/ Markdown files."""
    paths = [root / "README.md", *(root / "docs").rglob("*.md")]
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            target = raw.strip().split()[0].strip('<>')
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not target:
                continue
            if not (path.parent / target).exists():
                errors.append(f"{path.relative_to(root)}: missing {target}")
    return errors


if __name__ == "__main__":
    problems = check_docs()
    if problems:
        print("Broken documentation links:")
        for item in problems:
            print(f" - {item}")
        raise SystemExit(1)
    print("Documentation links OK")
