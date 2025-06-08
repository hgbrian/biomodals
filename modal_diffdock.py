# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs DiffDock for molecular docking using a diffusion model on Modal.

# DiffDock

- setting batch_size=1 may help for very large proteins

## Dependencies
There are lots of file dependencies that get downloaded.
Here is the output from running `python -m inference` the first time:

Models not found. Downloading
Attempting download from https://github.com/gcorso/DiffDock/releases/latest/download/diffdock_models.zip
Downloaded and extracted 6 files from https://github.com/gcorso/DiffDock/releases/latest/download/diffdock_models.zip
Attempting download from https://www.dropbox.com/scl/fi/drg90rst8uhd2633tyou0/diffdock_models.zip?rlkey=afzq4kuqor2jb8adah41ro2lz&dl=1
Downloaded and extracted 12 files from https://www.dropbox.com/scl/fi/drg90rst8uhd2633tyou0/diffdock_models.zip?rlkey=afzq4kuqor2jb8adah41ro2lz&dl=1
DiffDock will run on cuda
Generating ESM language model embeddings
Downloading: "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt" to /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt (2.5G)
Downloading: "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt" to /root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt (1Mb)
"""

import os
from pathlib import Path
from warnings import warn

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

app = App()

image = (
    Image.micromamba(python_version="3.9")
    .apt_install(["git", "wget", "nano"])
    .run_commands("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib")
    .pip_install(
        ["torch==1.13.1+cu117"], index_url="https://download.pytorch.org/whl/cu117"
    )
    .micromamba_install("prody==2.2.0", channels=["conda-forge", "bioconda"])
    .pip_install(
        [
            "dllogger@git+https://github.com/NVIDIA/dllogger.git",
            "e3nn==0.5.0",
            "fair-esm[esmfold]==2.0.0",
            "networkx==2.8.4",
            "pandas==1.5.1",
            "pybind11==2.11.1",
            "rdkit==2022.03.3",
            "scikit-learn==1.1.0",
            "scipy==1.12.0",
        ]
    )
    .pip_install(
        [
            "torch-cluster==1.6.0+pt113cu117",
            "torch-geometric==2.2.0",
            "torch-scatter==2.1.0+pt113cu117",
            "torch-sparse==0.6.16+pt113cu117",
            "torch-spline-conv==1.2.1+pt113cu117",
            "torchmetrics==0.11.0",
        ],
        find_links="https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html",
    )
    .run_commands("git clone https://github.com/gcorso/DiffDock.git")
    .run_commands("mkdir /content")
    # running python -m inference triggers downloads we need for the docker container
    .run_commands(
        "cd DiffDock && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib && python -m inference || true"
    )
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
)
def run_diffdock(pdbs_ligands: list, batch_size: int = 5) -> list:
    """Runs DiffDock inference for a list of protein PDB and ligand MOL2 file pairs.

    Args:
        pdbs_ligands (list[tuple[str, str, str, str]]): A list of tuples, where each tuple contains
                                                        (pdb_name, pdb_content, ligand_name, ligand_content).
        batch_size (int): Batch size for DiffDock inference. Defaults to 5.
                          Note: Only 1, 5, or 10 have been tested.

    Returns:
        list[tuple[str, str, str, bytes]]: A list of tuples, where each tuple contains
                                           the input PDB name, input ligand name,
                                           output filename, and its byte content.
    """
    import os
    from subprocess import run
    from tempfile import TemporaryDirectory

    os.chdir("/DiffDock")

    if batch_size not in (1, 5, 10):
        warn("batch_size only tested with 1, 5, or 10. This may not work.")

    outputs = []

    with TemporaryDirectory() as temp_dir:
        for pdb_name, pdb_content, ligand_name, ligand_content in pdbs_ligands:
            # Write input files to temp directory
            pdb_path = Path(temp_dir) / pdb_name
            ligand_path = Path(temp_dir) / ligand_name

            with open(pdb_path, "w") as f:
                f.write(pdb_content)
            with open(ligand_path, "w") as f:
                f.write(ligand_content)

            pdb_stem = Path(pdb_name).stem
            ligand_stem = Path(ligand_name).stem
            out_dir = f"./out_{pdb_stem}_{ligand_stem}"

            run(
                "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib && "
                f"python -m inference"
                f" --protein_path {pdb_path}"
                f" --batch_size {batch_size}"
                f" --ligand {ligand_path}"
                f" --out_dir {out_dir}",
                shell=True,
                check=True,
            )

            outputs.extend(
                [
                    (pdb_name, ligand_name, out_file.name, open(out_file, "rb").read())
                    for out_file in Path(out_dir).glob("**/*.*")
                    if os.path.isfile(out_file)
                ]
            )

    return outputs


@app.local_entrypoint()
def main(
    pdb_file: str,
    mol2_file: str,
    batch_size: int = 5,
    out_dir: str = "./out/diffdock",
    run_name: str = None,
):
    """Local entrypoint to run DiffDock inference for protein-ligand pairs.

    Args:
        pdb_file (str): Comma-separated string of paths to PDB files.
        mol2_file (str): Comma-separated string of paths to MOL2 ligand files.
                         (Order should correspond to `pdb_file`).
        batch_size (int): Batch size for DiffDock inference. Defaults to 5.
        out_dir (str): Directory to save the output files. Defaults to "./out/diffdock".
        run_name (str | None): Optional name for the run, used in the output directory structure.

    Returns:
        None
    """
    from datetime import datetime

    # Read file contents and prepare for remote execution
    pdb_files = [path.strip() for path in pdb_file.split(",")]
    mol2_files = [path.strip() for path in mol2_file.split(",")]

    pdbs_ligands = []
    for pdb_path, mol2_path in zip(pdb_files, mol2_files):
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        with open(mol2_path, "r") as f:
            mol2_content = f.read()

        pdb_name = Path(pdb_path).name
        mol2_name = Path(mol2_path).name

        pdbs_ligands.append((pdb_name, pdb_content, mol2_name, mol2_content))

    outputs = run_diffdock.remote(pdbs_ligands, batch_size)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for pdb_name, ligand_name, out_file, out_content in outputs:
        output_path = (
            out_dir_full
            / f"{Path(pdb_name).stem}_{Path(ligand_name).stem}"
            / out_file
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(output_path, "wb") as out:
                out.write(out_content)
