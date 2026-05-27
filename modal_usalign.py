# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""US-align — universal structural alignment (successor to TM-align).

https://zhanggroup.org/US-align/

Aligns 3D structures (proteins/RNA/DNA, monomers or complexes) and reports
TM-scores and RMSD. The `-mm` and `-ter` flags control alignment mode:

    -mm  0: monomer alignment (default)  1: oligomer alignment  2: chains-to-oligomer
    -ter 0: all chains all models  1: all chains first model  2: first chain only

```
# All-chain-by-all-chain alignment (default --ter 0)
modal run modal_usalign.py --pdb a.pdb --vs-pdbs b.pdb,c.pdb

# Single-chain alignment within complexes (two-step: align complex, then score one chain)
modal run modal_usalign.py --pdb a.pdb --vs-pdbs b.pdb --chain A
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("MODAL_GPU", None)
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", 60))

image = (
    Image.micromamba(python_version="3.11")
    .apt_install("git", "wget", "g++")
    .uv_pip_install("ProDy==2.4.1")
    .run_commands(
        "cd /usr/local/bin && "
        "wget https://zhanggroup.org/US-align/bin/module/USalign.cpp "
        "&& g++ -static -O3 -ffast-math -o USalign USalign.cpp "
        "&& chmod +x USalign "
        "&& rm USalign.cpp"
    )
)

app = App("usalign", image=image)

USALIGN_EXE = "/usr/local/bin/USalign"


def _unique_vs_names(pdb_name: str, vs_pdb_strs, vs_pdb_names) -> list[str]:
    """Filenames must be unique so USalign output rows are distinguishable."""
    if vs_pdb_names is None:
        return [f"vs_{i}.pdb" for i in range(len(vs_pdb_strs))]
    if isinstance(vs_pdb_names, str):
        return [vs_pdb_names if vs_pdb_names != pdb_name else f"0_{vs_pdb_names}"]
    not_unique = len(set(vs_pdb_names)) < len(vs_pdb_names)
    if any(pdb_name == v for v in vs_pdb_names) or not_unique:
        return [f"{i}_{v}" for i, v in enumerate(vs_pdb_names)]
    return list(vs_pdb_names)


def _align_pdb_by_matrix(pdb_file: str, matrix_txt: str) -> bytes:
    """Apply USalign's transformation matrix to a PDB/CIF and return aligned PDB bytes.
    Prody can't write CIF, so even CIF inputs are emitted as PDB."""
    import numpy as np
    import prody as pd
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        td_path = Path(td)
        if Path(pdb_file).suffix == ".cif":
            mol = pd.parseMMCIF(pdb_file)
        else:
            mol = pd.parsePDB(pdb_file)
        mat = open(matrix_txt).readlines()[2:5]
        arr = np.array([l.split()[1:] for l in mat]).astype(float)
        trans = pd.Transformation(arr[:, 1:], arr[:, 0])
        pd.writePDB(str(td_path / "aligned.pdb"), pd.applyTransformation(trans, mol))
        return (td_path / "aligned.pdb").read_bytes()


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def usalign(
    pdb_str: str,
    vs_pdb_strs: str | list[str],
    pdb_name: str | None = None,
    vs_pdb_names: str | list[str] | None = None,
    params_str: str | None = None,
) -> list[tuple[Path, bytes]]:
    """Align one structure against one or more others; default `-ter 0` (all-vs-all chains)."""
    import shlex
    from subprocess import run
    from tempfile import TemporaryDirectory

    if isinstance(vs_pdb_strs, str):
        vs_pdb_strs = [vs_pdb_strs]

    if params_str is None:
        params_str = "-ter 0"
    else:
        assert "outfmt" not in params_str, "outfmt is hardcoded"

    pdb_name = pdb_name or "input.pdb"
    vs_pdb_names = _unique_vs_names(pdb_name, vs_pdb_strs, vs_pdb_names)

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        in_path, out_path = Path(td_in), Path(td_out)
        (in_path / pdb_name).write_text(pdb_str)

        for vs_str, vs_name in zip(vs_pdb_strs, vs_pdb_names):
            (in_path / vs_name).write_text(vs_str)

            # align vs_pdb to pdb so the matrix transforms vs_pdb onto pdb
            cmd = [
                USALIGN_EXE,
                str(in_path / vs_name),
                str(in_path / pdb_name),
                "-outfmt", "2",
            ] + shlex.split(params_str)

            out_obj = run(cmd, check=True, capture_output=True)
            tsv_name = f"{Path(vs_name).stem}_{Path(pdb_name).stem}.tsv"
            (out_path / tsv_name).write_text(
                out_obj.stdout.decode().replace(str(in_path) + "/", "")
            )

        return [
            (f.relative_to(out_path), f.read_bytes())
            for f in out_path.glob("**/*")
            if f.is_file()
        ]


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def usalign_one_chain(
    pdb_str: str,
    vs_pdb_strs: str | list[str],
    pdb_name: str | None = None,
    vs_pdb_names: str | list[str] | None = None,
    params_str: str | None = None,
    chain: str = "A",
) -> list[tuple[Path, bytes]]:
    """Align complexes then report TM-score for one chain (two-step).

    Step 1: align the complexes with `-mm 1 -ter 1` (oligomer alignment).
    Step 2: re-run on the aligned output with `-TMscore 5 -se -chain1/-chain2`
    to score a specific chain without re-aligning.
    """
    import shlex
    from subprocess import run
    from tempfile import TemporaryDirectory

    if isinstance(vs_pdb_strs, str):
        vs_pdb_strs = [vs_pdb_strs]

    if params_str is None:
        print("Using default params (-mm 1 -ter 1).")
        params_str = "-mm 1 -ter 1"
    else:
        assert "outfmt" not in params_str, "outfmt is hardcoded"

    pdb_name = pdb_name or "input.pdb"
    vs_pdb_names = _unique_vs_names(pdb_name, vs_pdb_strs, vs_pdb_names)

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        in_path, out_path = Path(td_in), Path(td_out)
        (in_path / pdb_name).write_text(pdb_str)

        for vs_str, vs_name in zip(vs_pdb_strs, vs_pdb_names):
            (in_path / vs_name).write_text(vs_str)

            # Step 1: align complexes, output transformation matrix
            cmd = [
                USALIGN_EXE,
                str(in_path / vs_name),
                str(in_path / pdb_name),
                "-m", str(out_path / "matrix.txt"),
                "-o", str(out_path / f"sup_{Path(vs_name).stem}"),
                "-outfmt", "2",
            ] + shlex.split(params_str)
            out_obj = run(cmd, check=True, capture_output=True)
            (out_path / f"{Path(vs_name).stem}_{Path(pdb_name).stem}.tsv").write_text(
                out_obj.stdout.decode().replace(str(in_path) + "/", "")
            )

            # Also emit a transformed PDB for easy viewing in pymol
            aligned_pdb = _align_pdb_by_matrix(
                str(in_path / vs_name), str(out_path / "matrix.txt")
            )
            (out_path / f"align_{Path(vs_name).stem}_to_{Path(pdb_name).stem}.pdb").write_bytes(
                aligned_pdb
            )

            # Step 2: score a single chain (no re-alignment)
            cmd = [
                USALIGN_EXE,
                str(out_path / f"sup_{Path(vs_name).stem}.pdb"),
                str(in_path / pdb_name),
                "-chain1", chain,
                "-chain2", chain,
                "-TMscore", "5",
                "-se",
                "-outfmt", "2",
            ]
            out_obj = run(cmd, check=True, capture_output=True)
            (out_path / f"chain_{chain}_{Path(vs_name).stem}_{Path(pdb_name).stem}.tsv").write_text(
                out_obj.stdout.decode()
                .replace(str(in_path) + "/", "")
                .replace(str(out_path) + "/", "")
            )

        return [
            (f.relative_to(out_path), f.read_bytes())
            for f in out_path.glob("**/*")
            if f.is_file()
        ]


@app.local_entrypoint()
def main(
    pdb: str,
    vs_pdbs: str,
    chain: str | None = None,
    params_str: str | None = None,
    out_dir: str = "./out/usalign",
    run_name: str | None = None,
):
    """Args:
        pdb: reference structure (PDB or CIF)
        vs_pdbs: comma-separated list of structures to align against the reference
        chain: if set, use single-chain mode and report TM-score for this chain
        params_str: extra USalign flags (e.g. '-mm 1 -ter 1')
    """
    from datetime import datetime

    pdb_name, pdb_str = Path(pdb).name, Path(pdb).read_text()
    vs_paths = [Path(p) for p in vs_pdbs.split(",")]
    vs_pdb_strs = [p.read_text() for p in vs_paths]
    vs_pdb_names = [p.name for p in vs_paths]

    fn = usalign_one_chain if chain else usalign
    kwargs = dict(
        pdb_str=pdb_str,
        vs_pdb_strs=vs_pdb_strs,
        pdb_name=pdb_name,
        vs_pdb_names=vs_pdb_names,
        params_str=params_str,
    )
    if chain:
        kwargs["chain"] = chain
    outputs = fn.remote(**kwargs)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    for out_file, out_content in outputs:
        target = out_dir_full / out_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out_content)

    print(f"Results saved to: {out_dir_full}")
