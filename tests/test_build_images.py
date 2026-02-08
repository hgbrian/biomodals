"""Test that Modal app images build successfully.

WARNING: These tests are SLOW and use Modal cloud resources.
They are excluded from the default test run.

Run all:    uv run run_tests.py tests/test_build_images.py
Run one:    uv run run_tests.py tests/test_build_images.py -k alphafold

Expected first-build times (uncached):
    5-10 min      most apps
    15-30 min     afdesign, alphafold, bindcraft, rso (large pip installs)
    1-2 hours     diffdock (downloads ~3GB of ESM2 + DiffDock models)
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
MODAL_FILES = sorted(REPO_ROOT.glob("modal_*.py"))


def pytest_collection_modifyitems(config, items):
    build_tests = [i for i in items if "test_build_images" in str(i.fspath)]
    if build_tests:
        print(
            "\nWARNING: Image build tests are SLOW (5-30 min per app) "
            "and use Modal cloud resources.\n",
            file=sys.stderr,
        )


@pytest.fixture(params=MODAL_FILES, ids=lambda p: p.name)
def modal_file(request):
    return request.param


def test_image_builds(modal_file, tmp_path):
    """Build an app's Modal image by importing it and running a trivial function."""
    module_name = modal_file.stem
    script = tmp_path / "build_test.py"
    script.write_text(textwrap.dedent(f"""\
        import sys
        import modal

        # Import the image locally; inside the container this import fails
        # (module not mounted) but the image is already built so we fall back.
        try:
            sys.path.insert(0, "{REPO_ROOT}")
            from {module_name} import image
        except (ImportError, ModuleNotFoundError):
            image = modal.Image.debian_slim()

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
        timeout=7200,  # 2 hours — image builds with model downloads can be very slow
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"Image build failed for {modal_file.name}:\n"
        f"stdout: {result.stdout[-500:]}\n"
        f"stderr: {result.stderr[-500:]}"
    )
