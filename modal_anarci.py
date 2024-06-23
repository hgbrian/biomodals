"""Run ANARCI
Antibody Numbering and Antigen Receptor ClassIfication
https://github.com/oxpig/ANARCI

usage: ANARCI [-h] [--sequence INPUTSEQUENCE] [--outfile OUTFILE] [--scheme {m,c,k,imgt,kabat,chothia,martin,i,a,aho,wolfguy,w}]
              [--restrict {ig,tr,heavy,light,H,K,L,A,B} [{ig,tr,heavy,light,H,K,L,A,B} ...]] [--csv] [--outfile_hits HITFILE] [--hmmerpath HMMERPATH] [--ncpu NCPU]
              [--assign_germline] [--use_species {human,mouse,rat,rabbit,rhesus,pig,alpaca,cow}] [--bit_score_threshold BIT_SCORE_THRESHOLD]
"""

from pathlib import Path
from subprocess import run

from modal import App, Image, Mount

FORCE_BUILD = False
LOCAL_IN = "./in/anarci"
LOCAL_OUT = "./out/anarci"
REMOTE_IN = "/in"
REMOTE_OUT = LOCAL_OUT


image = (
    Image.micromamba()
    .apt_install("git")
    .pip_install(["biopython"], force_build=FORCE_BUILD)
    .micromamba_install(["libstdcxx-ng", "hmmer=3.3.2"], channels=["conda-forge", "bioconda"])
    .run_commands(
        "git clone https://github.com/oxpig/ANARCI && cd ANARCI && python setup.py install"
    )
)

app = App("anarci", image=image)


@app.function(
    timeout=60 * 15,
    mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)],
)
def anarci(input_fasta: str, params: str = None) -> list[str, str]:
    Path(REMOTE_OUT).mkdir(parents=True, exist_ok=True)

    if (Path(LOCAL_IN) / input_fasta).exists():
        assert input_fasta.suffix in (".faa", ".fasta"), f"not a fasta file: {v}"
        input = Path(REMOTE_IN) / Path(input_fasta).relative_to(LOCAL_IN)
        output = Path(REMOTE_OUT) / Path(input_fasta).stem
    else:
        input = input_fasta
        output = Path(REMOTE_OUT) / input[:8]

    command = f"ANARCI -i {input} --outfile {output}"

    if params is not None:
        command += " " + params
    else:
        command += " --csv --ncpu 2"

    run(command, shell=True, check=True)

    return [(out_file, open(out_file, "rb").read()) for out_file in Path(REMOTE_OUT).glob("**/*.*")]


@app.local_entrypoint()
def main(input_fasta, params=None):
    outputs = anarci.remote(input_fasta, params)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
