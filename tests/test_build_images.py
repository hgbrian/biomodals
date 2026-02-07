"""Test that Modal app images build successfully.

WARNING: These tests are SLOW (5-30 min per app) and use Modal cloud resources.
They are excluded from the default test run. Runs one image at a time.

Run all (sequential, stop on first failure):
    uv run run_tests.py tests/test_build_images.py -x

Run one:
    uv run run_tests.py tests/test_build_images.py -k alphafold
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MODAL_FILES = sorted(REPO_ROOT.glob("modal_*.py"))

# Apps that can't be imported locally (missing deps outside image.imports())
SKIP_APPS = {"modal_afdesign.py"}


def pytest_collection_modifyitems(config, items):
    build_tests = [i for i in items if "test_build_images" in str(i.fspath)]
    if build_tests:
        print(
            "\n"
            "WARNING: Image build tests are SLOW (5-30 min per app) "
            "and use Modal cloud resources.\n"
            "Run one app: uv run run_tests.py tests/test_build_images.py -k alphafold\n",
            file=sys.stderr,
        )


@pytest.fixture(
    params=[f for f in MODAL_FILES if f.name not in SKIP_APPS],
    ids=lambda p: p.name,
)
def modal_file(request):
    return request.param


def test_image_builds(modal_file, tmp_path):
    """Build an app's Modal image by importing it and running a trivial function."""
    module_name = modal_file.stem
    script = tmp_path / "build_test.py"
    script.write_text(textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, "{REPO_ROOT}")
        from {module_name} import image
        import modal

        app = modal.App("test-image-build-{module_name}")

        @app.function(image=image, timeout=120)
        def ping():
            import platform
            return platform.python_version()

        @app.local_entrypoint()
        def main():
            result = ping.remote()
            print(f"OK: {{result}}")
    """))

    result = subprocess.run(
        ["uv", "run", "--with", "modal", "modal", "run", str(script)],
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min per image
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Image build failed for {modal_file.name}:\n{result.stderr[-1000:]}"
    )
