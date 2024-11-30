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
>asdf
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQK
```
Then run
```
modal run modal_boltz1.py --input-faa test_boltz1.faa
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
    open(in_faa := "/tmp/tmp.fasta", "w").write(">tmp\nMAWTPLLLLLLSHCTGSLSQPVLT\n")

    fixed_faa_str = _prepare_fasta(in_faa)
    open(fixed_faa := Path(in_dir) / "fixed.fasta", "w").write(fixed_faa_str)

    run(
        [
            "boltz",
            "predict",
            fixed_faa,
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

app = App("boltz1", image=image)


def fasta_iter(fasta_name):
    """yield stripped seq_ids and seqs"""
    from itertools import groupby

    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def _prepare_fasta(input_faa: str) -> str:
    """Basically, add the path for the a3m msa file to the fasta header.
    This is a bit complicated and there is probably a better way!
    For random ids, assume PROTEIN.
    """
    import re

    chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    fixed_fasta = ""

    rx = re.compile(r"^([A-Z])\|(\S+)\|?(.*)$")

    for n, (seq_id, seq) in enumerate(fasta_iter(input_faa)):
        if n >= len(chains):
            raise ValueError(">26 chains not allowed")

        s_info = rx.search(seq_id)

        if s_info and s_info.groups()[1].lower() != "protein":
            fixed_fasta += f">{seq_id}\n{seq}\n"
        else:
            # proteins can have explicit |PROTEIN| or just an id
            # colabfold_batch escapes some characters
            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"not AAs: {seq}"
            fixed_fasta += f">{chains[n]}|PROTEIN|\n{seq.upper()}\n"

    return fixed_fasta


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def boltz(input_faa_str: str, input_faa_name: str = "input.faa"):
    """Runs Boltz on a fasta.
    Fasta can contain protein, DNA, RNA, smiles, ccd
    """
    from subprocess import run

    Path(in_dir := "/tmp/in_boltz").mkdir(parents=True, exist_ok=True)
    Path(out_dir := "/tmp/out_boltz").mkdir(parents=True, exist_ok=True)

    in_faa = Path(in_dir) / input_faa_name
    open(in_faa, "w").write(input_faa_str)

    fixed_faa_str = _prepare_fasta(in_faa)
    open(fixed_faa := Path(in_dir) / "fixed.fasta", "w").write(fixed_faa_str)

    run(
        [
            "boltz",
            "predict",
            fixed_faa,
            "--out_dir",
            out_dir,
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
    out_dir: str = "./out/boltz1",
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
