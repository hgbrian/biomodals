"""
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
from modal import App, Image, Mount

LOCAL_IN = "./in/diffdock"
LOCAL_OUT = "./out/diffdock"
REMOTE_IN = "/in"

GPU_SIZE = os.environ.get("GPU_SIZE", "80GB")
GPU = modal.gpu.A100(size=GPU_SIZE)
TIMEOUT_MINS = int(os.environ.get("TIMEOUT_MINS", 15))

app = App()

image = (
    Image.micromamba(python_version="3.9")
    .apt_install(["git", "wget", "nano"])
    .run_commands("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib")
    .pip_install(["torch==1.13.1+cu117"], index_url="https://download.pytorch.org/whl/cu117")
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
    timeout=60 * TIMEOUT_MINS,
    mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)],
)
def run_diffdock(pdbs_ligands: list, batch_size: int = 5) -> dict:
    import os
    from subprocess import run

    os.chdir("/DiffDock")

    if batch_size not in (1, 5, 10):
        warn("batch_size only tested with 1, 5, or 10. This may not work.")

    outputs = []
    for pdb, ligand in pdbs_ligands:
        _pdb, _ligand = Path(pdb).relative_to(LOCAL_IN), Path(ligand).relative_to(LOCAL_IN)
        remote_pdb, remote_ligand = Path(REMOTE_IN) / _pdb, Path(REMOTE_IN) / _ligand
        out_dir = f"./out_{_pdb.stem}_{_ligand.stem}"

        run(
            "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib && "
            f"python -m inference"
            f" --protein_path {remote_pdb}"
            f" --batch_size {batch_size}"
            f" --ligand {remote_ligand}"
            f" --out_dir {out_dir}",
            shell=True,
        )

        outputs.extend(
            [
                (_pdb, _ligand, out_file.name, open(out_file, "rb").read())
                for out_file in Path(out_dir).glob("**/*.*")
                if os.path.isfile(out_file)
            ]
        )

    return outputs


@app.local_entrypoint()
def main(pdb: str, mol2: str, batch_size: int = 5):
    pdbs_ligands = [
        (_pdb.strip(), _mol2.strip()) for _pdb, _mol2 in zip(pdb.split(","), mol2.split(","))
    ]

    outputs = run_diffdock.remote(pdbs_ligands, batch_size)

    for pdb, ligand, out_file, out_content in outputs:
        out_path = Path(LOCAL_OUT) / Path(f"{pdb}_{ligand}") / Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_path, "wb") as out:
                out.write(out_content)
