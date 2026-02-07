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

sys.exit(subprocess.call(["pytest", "tests/", "-v", *sys.argv[1:]]))
