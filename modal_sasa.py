# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Solvent-accessible surface area via dr_sasa.

https://github.com/nioroso-x3/dr_sasa_n

Annotates a PDB (or CIF, auto-converted via openbabel) with per-atom SASA in the
B-factor column. Open the output in pymol:
```
pymol out/sasa/<run>/<input>.asa.pdb
# split_chains
# spectrum b, blue_white_red
```

```
modal run modal_sasa.py --input-pdb protein.pdb
modal run modal_sasa.py --input-pdb structure.cif
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("MODAL_GPU", None)
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", 5))

image = (
    Image.debian_slim()
    .apt_install("wget", "openbabel")
    .run_commands(
        "wget https://github.com/nioroso-x3/dr_sasa_n/releases/download/v0.4b/dr_sasa.tar.gz",
        "tar xzvf dr_sasa.tar.gz",
    )
)

app = App("sasa", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def sasa_from_pdb(pdb_str: str, pdb_name: str | None = None) -> list[tuple[Path, bytes]]:
    import tempfile
    from subprocess import run

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pdb_name = pdb_name or "sasa_input.pdb"
        pdb_path = td_path / pdb_name
        pdb_path.write_text(pdb_str)

        run(f'cd {td} && /dr_sasa.bin -m 0 -i "{pdb_path}"', shell=True, check=True)

        return [
            (f.relative_to(td_path), f.read_bytes())
            for f in td_path.glob("**/*")
            if f.is_file()
        ]


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def sasa_from_cif(cif_str: str, cif_name: str | None = None) -> list[tuple[Path, bytes]]:
    """dr_sasa only takes PDB, so convert CIF→PDB with openbabel first."""
    import tempfile
    from subprocess import run

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cif_name = cif_name or "sasa_input.cif"
        cif_path = td_path / cif_name
        cif_path.write_text(cif_str)

        pdb_path = td_path / f"{cif_path.stem}.pdb"
        run(["obabel", str(cif_path), "-O", str(pdb_path)], check=True)
        return sasa_from_pdb.local(pdb_path.read_text(), pdb_path.name)


@app.local_entrypoint()
def main(
    input_pdb: str,
    out_dir: str = "./out/sasa",
    run_name: str | None = None,
):
    from datetime import datetime

    input_str = Path(input_pdb).read_text()
    name = Path(input_pdb).name

    if input_pdb.endswith(".cif"):
        outputs = sasa_from_cif.remote(input_str, name)
    elif input_pdb.endswith(".pdb"):
        outputs = sasa_from_pdb.remote(input_str, name)
    else:
        raise ValueError(f"Unsupported input: {input_pdb} (need .pdb or .cif)")

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    for out_file, out_content in outputs:
        target = out_dir_full / out_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out_content)

    print(f"Results saved to: {out_dir_full}")
