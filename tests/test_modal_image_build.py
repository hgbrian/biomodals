"""Verify Modal can parse each app file (image definition, App object, decorators).

This is distinct from ast.parse (syntax check) — it validates that Modal's
runtime can resolve each app's structure.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MODAL_FILES = sorted(REPO_ROOT.glob("modal_*.py"))

@pytest.fixture(params=MODAL_FILES, ids=lambda p: p.name)
def modal_file(request):
    return request.param


def test_modal_can_parse_app(modal_file):
    """Verify `uv run modal run <file> --help` succeeds for each app."""
    result = subprocess.run(
        ["uv", "run", "--with", "modal", "modal", "run", str(modal_file), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Modal failed to parse {modal_file.name}:\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
