"""
Boltz-1 https://github.com/jwohlwend/boltz

TODO: adding a custom msa dir is not really supported
The user has to get the msa_dir onto modal so it requires a few more steps.

TODO: use yaml instead of fasta, and convert input fasta to yaml

## Example input:
```
>A|PROTEIN|
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQK
PGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYS
STS
>B|DNA|
ACTGACTGGAAGATTTTTTTTTTTCCCCCGTAGTTTTTACCCGACG
>C|smiles
N[C@@H](Cc1ccc(O)cc1)C(=O)O
```
Then run
```
modal run modal_boltz.py --input-faa test_boltz.fasta
```

"""

import os
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", modal.gpu.A100(size="80GB"))
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

CACHE_DIR = "/root/.boltz"
ENTITY_TYPES = {"protein", "dna", "rna", "ccd", "smiles"}
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY"


def download_model():
    """Force download of the Boltz-1 model by running it once"""
    from subprocess import run

    Path(in_dir := "/tmp/tmp_in_boltz").mkdir(parents=True, exist_ok=True)
    open(in_faa := Path(in_dir) / "tmp.fasta", "w").write(">A|PROTEIN|\nMAWTPLLLLLLSH\n")

    run(
        [
            "boltz",
            "predict",
            str(in_faa),
            "--out_dir",
            "/tmp",
            "--cache",
            CACHE_DIR,
            "--use_msa_server",
        ],
        check=True,
    )


image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold"
    )
    .micromamba_install(
        "kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"]
    )
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a100",
    )
    .run_commands("python -m colabfold.download")
    .apt_install("build-essential")
    .pip_install("boltz")
    .run_function(download_model, gpu="a10g")
)

app = App("boltz", image=image)


def fasta_iter(fasta_name):
    """yield stripped seq_ids and seqs"""
    from itertools import groupby

    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def boltz(input_faa_str: str, input_faa_name: str = "input.fasta"):
    """Runs Boltz on a fasta.
    Fasta can contain protein, DNA, RNA, smiles, ccd
    """
    from subprocess import run

    Path(in_dir := "/tmp/in_boltz").mkdir(parents=True, exist_ok=True)
    Path(out_dir := "/tmp/out_boltz").mkdir(parents=True, exist_ok=True)

    in_faa = Path(in_dir) / input_faa_name
    if in_faa.suffix == ".faa":
        in_faa = in_faa.with_suffix(".fasta")
    open(in_faa, "w").write(input_faa_str)

    run(
        [
            "boltz",
            "predict",
            str(in_faa),
            "--out_dir",
            str(out_dir),
            "--cache",
            CACHE_DIR,
            "--use_msa_server",
        ],
        check=True,
    )

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if Path(out_file).is_file()
    ]


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/boltz",
    run_name: str = None,
):
    from datetime import datetime

    input_faa_str = open(input_faa).read()

    outputs = boltz.remote(
        input_faa_str,
        Path(input_faa).name,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
                out.write(out_content)
