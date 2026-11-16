"""Smoke tests with Streamlit's AppTest.  Usage:  XBCI_ARTIFACT_ROOT=<folder> python -m pytest tests -q   (or python tests/test_smoke.py)"""
import os, sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["🏠 Start here", "🧪 Playground", "⚖️ Compare models", "🔬 Under the hood", "📊 Performance", "❓ Science Q&A", "📚 Notes & Blog", "ℹ️ About"]


def run_page(page, actions=None, timeout=300):
    at = AppTest.from_file(APP, default_timeout=timeout).run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key="nav").set_value(page).run()
    assert not at.exception, (page, [e.value for e in at.exception])
    if actions:
        actions(at)
        assert not at.exception, (page, [e.value for e in at.exception])
    return at


def test_all_pages_render():
    if not os.environ.get("XBCI_ARTIFACT_ROOT"):
        import pytest
        pytest.skip("set XBCI_ARTIFACT_ROOT to run the UI smoke test")
    for p in PAGES:
        run_page(p)


if __name__ == "__main__":
    test_all_pages_render()
    print("all pages render OK")
