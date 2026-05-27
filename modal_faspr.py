# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""FASPR - Fast and Accurate Side-chain Packing and Rotamer prediction.

https://github.com/tommyhuangthu/FASPR

FASPR repacks side chains of an input PDB (which must have complete main-chain atoms)
and can optionally introduce mutations via a sequence file.

```
# Repack side chains only
modal run modal_faspr.py --input-pdb protein.pdb

# Repack with mutations (single-letter sequence file)
modal run modal_faspr.py --input-pdb protein.pdb --seq mutations.txt
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("MODAL_GPU", None)  # CPU only
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", 30))
COMMIT = "0d55732fd6307f373018c6bddd842291c355c5f7"

image = (
    Image.debian_slim()
    .apt_install("git", "wget", "build-essential", "g++")
    .run_commands(
        "git clone https://github.com/tommyhuangthu/FASPR.git /opt/FASPR",
        f"cd /opt/FASPR && git checkout {COMMIT}",
        "cd /opt/FASPR && g++ -O3 --fast-math -o FASPR src/*.cpp",
        "chmod +x /opt/FASPR/FASPR",
    )
)

app = App("faspr", image=image)


def _fix_pdb_atom_numbering_inplace(pdb_path: Path):
    """FASPR's output atom numbering can confuse pymol; renumber sequentially."""
    atom_num = 1
    fixed_pdb = []

    for line in pdb_path.open():
        if line.startswith("ATOM"):
            fixed_pdb.append(f"ATOM{atom_num:7d}{line[11:]}")
            atom_num += 1
        elif line.startswith("TER"):
            fixed_pdb.append(f"TER{atom_num:8d}{line[11:]}")
            atom_num += 1
        else:
            fixed_pdb.append(line)

    pdb_path.write_text("".join(fixed_pdb))


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def faspr(input_pdb_str: str, seq_str: str | None = None) -> list:
    """Repack side chains of `input_pdb_str` with optional mutation `seq_str`."""
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        pdb_path = Path(td_in) / "input.pdb"
        pdb_path.write_text(input_pdb_str)

        output_path = Path(td_out) / "faspr_out.pdb"
        cmd = ["/opt/FASPR/FASPR", "-i", str(pdb_path), "-o", str(output_path)]

        if seq_str:
            seq_path = Path(td_in) / "sequence.txt"
            seq_path.write_text(seq_str)
            cmd.extend(["-s", str(seq_path)])

        print(f"Running: {' '.join(cmd)}")
        # change to FASPR directory to ensure rotamer library is found
        run(cmd, check=True, cwd="/opt/FASPR")

        _fix_pdb_atom_numbering_inplace(output_path)

        return [(Path("faspr_out.pdb"), output_path.read_bytes())]


@app.local_entrypoint()
def main(
    input_pdb: str,
    seq: str | None = None,
    out_dir: str = "./out/faspr",
    run_name: str | None = None,
):
    from datetime import datetime

    input_pdb_str = Path(input_pdb).read_text()
    seq_str = Path(seq).read_text() if seq else None

    outputs = faspr.remote(input_pdb_str=input_pdb_str, seq_str=seq_str)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        output_path = out_dir_full / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"Results saved to: {out_dir_full}")
