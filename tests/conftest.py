"""Shared pytest hooks for the biomodals test suite.

Hooks must live in conftest.py to be reliably picked up by pytest.
"""

import sys


def pytest_collection_modifyitems(config, items):
    build_items = [i for i in items if "test_build_images" in str(i.fspath)]
    quick_items = [i for i in items if "test_quick_runs" in str(i.fspath)]

    if build_items:
        print(
            "\n"
            "============================================================\n"
            f"WARNING: About to run {len(build_items)} image-build test(s).\n"
            "These build Modal images on the cloud — SLOW (5-30 min per app,\n"
            "up to 1-2 hr for diffdock) but do NOT use GPU.\n"
            "Image-build compute is metered on Modal.\n"
            "Ctrl-C now to abort.\n"
            "============================================================\n",
            file=sys.stderr,
        )

    if quick_items:
        print(
            "\n"
            "============================================================\n"
            f"WARNING: About to run {len(quick_items)} functional test(s).\n"
            "These USE GPU cloud resources and COST MONEY.\n"
            "Full suite: ~30-60 min parallel (1-2 hr sequential), ~$5-15.\n"
            "Slowest apps: bindcraft (30-60 min, A100, ~$2-4),\n"
            "              germinal (10-30 min, ~$1-2).\n"
            "Ctrl-C now to abort.\n"
            "============================================================\n",
            file=sys.stderr,
        )
