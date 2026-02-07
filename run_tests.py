# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
#     "pytest",
#     "numpy",
#     "pyyaml",
# ]
# ///
import subprocess
import sys

args = sys.argv[1:]

# If explicit test paths given, run those; otherwise run tests/ minus slow build tests
has_paths = any(a.endswith(".py") or a.startswith("tests/") for a in args)
if has_paths:
    sys.exit(subprocess.call(["pytest", "-v", *args]))
else:
    sys.exit(subprocess.call([
        "pytest", "tests/", "-v",
        "--ignore=tests/test_build_images.py",
        *args,
    ]))
