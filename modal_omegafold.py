import glob

from subprocess import run
from pathlib import Path

from modal import Image, Mount, Stub

FORCE_BUILD = False
MODAL_IN = "./modal_in/omegafold"
MODAL_OUT = "./modal_out/omegafold"

stub = Stub()

image = (Image
         .debian_slim()
         .apt_install("git")
         .pip_install("git+https://github.com/HeliXonProtein/OmegaFold.git", force_build=FORCE_BUILD)
        )

@stub.function(image=image, gpu="a100", timeout=60*15,
               mounts=[Mount.from_local_dir(MODAL_IN, remote_path="/in")])
def omegafold(input_fasta:str) -> list[str, str]:
    input_fasta = Path(input_fasta)
    assert input_fasta.parent.resolve() == Path(MODAL_IN).resolve(), f"wrong input_fasta dir {input_fasta.parent}"
    assert input_fasta.suffix in (".faa", ".fasta"), f"not fasta file {input_fasta}"

    run(["mkdir", "-p", MODAL_OUT], check=True)
    run(["omegafold", "--model", "2", "--subbatch_size", "224",
         f"/in/{input_fasta.name}", MODAL_OUT], check=True)

    return [(pdb_file, open(pdb_file, "rb").read())
            for pdb_file in glob.glob(f"{MODAL_OUT}/**/*.pdb", recursive=True)]

@stub.local_entrypoint()
def main(input_fasta):
    outputs = omegafold.remote(input_fasta)

    for (out_file, out_content) in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, 'wb') as out:
                out.write(out_content)

