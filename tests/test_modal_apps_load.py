"""Syntax and structure checks for all modal_*.py apps."""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MODAL_FILES = sorted(REPO_ROOT.glob("modal_*.py"))


@pytest.fixture(params=MODAL_FILES, ids=lambda p: p.name)
def modal_file(request):
    return request.param


def test_syntax(modal_file):
    """Each modal app file should be valid Python."""
    source = modal_file.read_text()
    ast.parse(source, filename=modal_file.name)


def _is_app_call(node):
    """Check for App(...) or modal.App(...)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "App":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "App":
        return True
    return False


def test_has_app(modal_file):
    """Each modal app file should define an App(...)."""
    source = modal_file.read_text()
    tree = ast.parse(source)
    assert any(_is_app_call(node) for node in ast.walk(tree)), (
        f"{modal_file.name} missing App() call"
    )


def test_has_app_decorator(modal_file):
    """Each modal app file should have at least one @app. decorator."""
    source = modal_file.read_text()
    assert "@app." in source, f"{modal_file.name} missing @app. decorator"


def test_expected_app_count():
    """Sanity check: we should have at least 19 modal app files."""
    assert len(MODAL_FILES) >= 19, f"Expected >= 19 modal apps, found {len(MODAL_FILES)}"
