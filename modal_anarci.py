"""Run ANARCI
Antibody Numbering and Antigen Receptor ClassIfication
https://github.com/oxpig/ANARCI

usage: ANARCI [-h] [--sequence INPUTSEQUENCE] [--outfile OUTFILE] [--scheme {m,c,k,imgt,kabat,chothia,martin,i,a,aho,wolfguy,w}]
              [--restrict {ig,tr,heavy,light,H,K,L,A,B} [{ig,tr,heavy,light,H,K,L,A,B} ...]] [--csv] [--outfile_hits HITFILE] [--hmmerpath HMMERPATH] [--ncpu NCPU]
              [--assign_germline] [--use_species {human,mouse,rat,rabbit,rhesus,pig,alpaca,cow}] [--bit_score_threshold BIT_SCORE_THRESHOLD]
"""

from subprocess import run
from pathlib import Path

from modal import Image, Mount, App

FORCE_BUILD = False
LOCAL_IN = "./in/anarci"
LOCAL_OUT = "./out/anarci"
REMOTE_IN = "/in"
REMOTE_OUT = LOCAL_OUT

app = App()

image = (
    Image.micromamba()
    .apt_install("git")
    .pip_install("biopython")
    .micromamba_install(["libstdcxx-ng", "hmmer=3.3.2"], channels=["conda-forge", "bioconda"])
    .run_commands(
        "git clone https://github.com/oxpig/ANARCI && cd ANARCI && python setup.py install"
    )
)


@app.function(
    image=image,
    timeout=60 * 15,
    mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)],
)
def anarci(input_fasta: str, kwargs_str: str) -> list[str, str]:
    input_fasta = Path(input_fasta).relative_to(LOCAL_IN)
    assert input_fasta.suffix in (".faa", ".fasta"), f"not a fasta file: {input_fasta}"

    Path(REMOTE_OUT).mkdir(parents=True, exist_ok=True)

    command = (
        f"ANARCI -i {str(Path(REMOTE_IN) / input_fasta)} "
        f"--outfile {str(Path(REMOTE_OUT) / Path(input_fasta).stem)} "
        "--csv --ncpu 2"
    )

    if kwargs_str is not None:
        command += " " + kwargs_str

    run(command, shell=True, check=True)

    return [(out_file, open(out_file, "rb").read()) for out_file in Path(REMOTE_OUT).glob("**/*.*")]


@app.local_entrypoint()
def main(input_fasta, kwargs_str=None):
    outputs = anarci.remote(input_fasta, kwargs_str)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
