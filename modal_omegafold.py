"""Run OmegaFold on an amino acid fasta file to produce a pdb file.

Default to 80GB since I have seen failures at 40GB.
"""

import os
from pathlib import Path
from subprocess import run

import modal
from modal import App, Image, Mount

FORCE_BUILD = False
LOCAL_IN = "./in/omegafold"
LOCAL_OUT = "./out/omegafold"
REMOTE_IN = "/in"
REMOTE_OUT = LOCAL_OUT
GPU_SIZE = os.environ.get("MODAL_GPU_SIZE", "80GB")
GPU = modal.gpu.A100(size=GPU_SIZE)
TIMEOUT_MINS = int(os.environ.get("TIMEOUT_MINS", 15))
app = App()

image = (
    Image.debian_slim()
    .apt_install("git")
    .pip_install("git+https://github.com/HeliXonProtein/OmegaFold.git", force_build=FORCE_BUILD)
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * TIMEOUT_MINS,
    mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)],
)
def omegafold(input_fasta: str, subbatch_size: int) -> list[str, str]:
    input_fasta = Path(input_fasta).relative_to(LOCAL_IN)
    assert input_fasta.suffix in (".faa", ".fasta"), f"not a fasta file: {input_fasta}"

    Path(LOCAL_OUT).mkdir(parents=True, exist_ok=True)

    run(
        f"omegafold --model 2 --subbatch_size {subbatch_size} {Path(REMOTE_IN) / input_fasta.name} {REMOTE_OUT}",
        shell=True,
        check=True,
    )

    return [
        (pdb_file, open(pdb_file, "rb").read()) for pdb_file in Path(REMOTE_OUT).glob("**/*.pdb")
    ]


@app.local_entrypoint()
def main(input_fasta, subbatch_size: int = 224):
    outputs = omegafold.remote(input_fasta, subbatch_size)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
