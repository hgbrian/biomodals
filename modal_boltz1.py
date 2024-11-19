"""
Boltz-1 https://github.com/jwohlwend/boltz

Because I have to add an msa path to the fasta header,
there is some awkwardness to the preprocessing.
I assume protein when it's not explicitly otherwise.

I have tested a bit but if it fails it's probably due to this.

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
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "A100")

CACHE_DIR = "/root/.boltz"
ENTITY_TYPES = {"protein", "dna", "rna", "ccd", "smiles"}
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY"


def _get_msa(in_faa: str, msa_dir: str):
    """Use colabfold_batch (a shared server)
    to get an MSA for the proteins in this fasta.
    Note the .a3m filename is derived from the fasta header,
    with some characters escaped.
    """
    from subprocess import run

    # separate out the proteins
    proteins = [
        f">{s_id}\n{seq}\n"
        for s_id, seq in fasta_iter(in_faa)
        if all(aa in ALLOWED_AAS for aa in seq.upper())
    ]
    open("/tmp/msa.fasta", "w").write("".join(proteins))

    run(f"colabfold_batch /tmp/msa.fasta {msa_dir} --msa-only", shell=True, check=True)

    return msa_dir


def download_model():
    """Force download of the Boltz-1 model by running it once"""
    from subprocess import run

    in_dir = "/tmp/tmp_in_boltz"
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    msa_dir = "/tmp/tmp_msa_boltz"
    in_faa = "/tmp/tmp.fasta"
    open(in_faa, "w").write(">tmp\nMAWTPLLLLLLSHCTGSLSQPVLTQAPTSLSASS\n")

    _get_msa("/tmp/tmp.fasta", "/tmp/tmp_msa_boltz")
    fixed_faa_str = _fix_fasta(in_faa, msa_dir)
    fixed_faa = Path(in_dir) / "fixed.fasta"
    open(fixed_faa, "w").write(fixed_faa_str)

    run(
        f"boltz predict {fixed_faa} --out_dir /tmp --cache {CACHE_DIR}",
        shell=True,
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
    .pip_install("boltz==0.1.0")
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


def _fix_fasta(input_faa: str, msa_dir) -> str:
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
            seq_id_esc = re.sub(r"[|_/ ,\[\]\(\)]", "_", seq_id)

            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"not AAs: {seq}"
            fixed_fasta += f">{chains[n]}|PROTEIN|{msa_dir}/{seq_id_esc}.a3m\n{seq}\n"

    return fixed_fasta


@app.function(timeout=60 * 60, gpu=GPU)
def boltz(input_faa_str: str, input_faa_name: str = "input.faa"):
    """Runs Boltz on a fasta.
    Fasta can contain protein, DNA, RNA, smiles, ccd
    """
    from subprocess import run

    in_dir = "/tmp/in_boltz"
    out_dir = "/tmp/out_boltz"
    msa_dir = "/tmp/msa_boltz"
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    in_faa = Path(in_dir) / input_faa_name
    open(in_faa, "w").write(input_faa_str)

    _get_msa(in_faa, msa_dir)
    fixed_faa_str = _fix_fasta(in_faa, msa_dir)
    fixed_faa = Path(in_dir) / "fixed.fasta"
    open(fixed_faa, "w").write(fixed_faa_str)

    run(
        f"boltz predict {fixed_faa} --out_dir {out_dir} --cache {CACHE_DIR}",
        shell=True,
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
):
    from datetime import datetime

    input_faa_str = open(input_faa).read()

    outputs = boltz.remote(
        input_faa_str,
        Path(input_faa).name,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / today

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
                out.write(out_content)
