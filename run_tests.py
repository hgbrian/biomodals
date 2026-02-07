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

sys.exit(subprocess.call([
    "pytest", "tests/", "-v",
    "--ignore=tests/test_build_images.py",
    *sys.argv[1:],
]))
