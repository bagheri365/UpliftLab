"""M9 publication integrity: no claims of verified causality or test use."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_publication_documents_exist():
    for relative in (
        'docs/final_research_report.md',
        'docs/final_reproducibility.md',
        'docs/release_checklist.md',
    ):
        assert (ROOT / relative).is_file(), relative


def test_readme_links_to_report_and_exploratory_status():
    text = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'docs/final_research_report.md' in text
    assert 'exploratory' in text.lower()
    assert 'held-out test' in text.lower()


def test_report_lists_six_models_and_unverified_evidence():
    text = (ROOT / 'docs/final_research_report.md').read_text(encoding='utf-8')
    for model in ('S-Learner', 'T-Learner', 'X-Learner', 'R-Learner', 'DR-Learner', 'Causal Forest'):
        assert model in text
    for term in ('randomized', 'propensity', 'timing', 'ACIC', 'unopened'):
        assert term.lower() in text.lower()


def test_release_never_claims_test_gate_open():
    checklist = (ROOT / 'docs/release_checklist.md').read_text(encoding='utf-8')
    assert '`false`' in checklist
    assert 'manual' in checklist.lower()
