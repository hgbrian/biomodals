"""Run OmegaFold on a fasta file.

If it runs out of memory, try changing GPU to "80GB".
"""

import glob

from subprocess import run
from pathlib import Path

import modal
from modal import Image, Mount, Stub

FORCE_BUILD = False
LOCAL_IN = "./in/omegafold"
LOCAL_OUT = "./out/omegafold"
REMOTE_IN = "/in"
GPU = modal.gpu.A100(size="40GB")

stub = Stub()

image = (Image
         .debian_slim()
         .apt_install("git")
         .pip_install("git+https://github.com/HeliXonProtein/OmegaFold.git", force_build=FORCE_BUILD)
        )

@stub.function(image=image, gpu=GPU, timeout=60*15,
               mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)])
def omegafold(input_fasta:str, subbatch_size:int) -> list[str, str]:
    input_fasta = Path(input_fasta).relative_to(LOCAL_IN)
    assert input_fasta.suffix in (".faa", ".fasta"), f"not a fasta file: {input_fasta}"

    Path(LOCAL_OUT).mkdir(parents=True, exist_ok=True)

    run(["omegafold",
         "--model", "2",
         "--subbatch_size", str(subbatch_size),
         Path(REMOTE_IN) / input_fasta.name,
         LOCAL_OUT
         ], check=True)

    return [(pdb_file, open(pdb_file, "rb").read())
            for pdb_file in glob.glob(f"{LOCAL_OUT}/**/*.pdb", recursive=True)]

@stub.local_entrypoint()
def main(input_fasta, subbatch_size:int=224):
    outputs = omegafold.remote(input_fasta, subbatch_size)

    for (out_file, out_content) in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, 'wb') as out:
                out.write(out_content)

