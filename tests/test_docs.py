"""Regression checks for the documentation link validator."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_documentation_links() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = spec_from_file_location("upliftlab_docs_check", root / "scripts" / "check_docs.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_docs(root) == []
