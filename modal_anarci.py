"""Runs ANARCI (Antibody Numbering and Antigen Receptor ClassIfication) on Modal.

ANARCI Usage:
  ANARCI [-h] [--sequence INPUTSEQUENCE] [--outfile OUTFILE] [--scheme {m,c,k,imgt,kabat,chothia,martin,i,a,aho,wolfguy,w}]
                [--restrict {ig,tr,heavy,light,H,K,L,A,B} [{ig,tr,heavy,light,H,K,L,A,B} ...]] [--csv] [--outfile_hits HITFILE] [--hmmerpath HMMERPATH] [--ncpu NCPU]
                [--assign_germline] [--use_species {human,mouse,rat,rabbit,rhesus,pig,alpaca,cow}] [--bit_score_threshold BIT_SCORE_THRESHOLD]

For more details, see https://github.com/oxpig/ANARCI
"""

from pathlib import Path
from subprocess import run

from modal import App, Image

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
    image=image.add_local_dir(LOCAL_IN, remote_path=REMOTE_IN),
    timeout=60 * 15,
)
def anarci(input_fasta: str, params: str = None) -> list[tuple[str, bytes]]:
    """Runs ANARCI on a given FASTA file or sequence string using specified parameters.

    Args:
        input_fasta (str): Path to the input FASTA file (must have .faa or .fasta extension)
                           or a raw FASTA sequence string. If a path, it's treated as relative
                           to `LOCAL_IN` if it exists there, otherwise as an absolute path
                           or direct sequence input within the Modal container.
        params (str | None): Optional string of additional parameters to pass to the ANARCI
                             command. If None, defaults to "--csv --ncpu 2".

    Returns:
        list[tuple[str, bytes]]: A list of tuples, where each tuple contains the output
                                 filename (str) and its byte content. The filenames are
                                 derived from the input.
    """
    Path(REMOTE_OUT).mkdir(parents=True, exist_ok=True)

    if (Path(LOCAL_IN) / input_fasta).exists():
        input_path = Path(input_fasta)
        assert input_path.suffix in (".faa", ".fasta"), f"not a fasta file: {input_fasta}"
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
    """Local entrypoint to run ANARCI via Modal.

    This function takes an input FASTA file (path or string) and optional ANARCI
    parameters, then calls the remote Modal `anarci` function and saves
    the results locally.

    Args:
        input_fasta (str): Path to the input FASTA file or a raw FASTA sequence string.
        params (str | None): Optional string of additional parameters for ANARCI.

    Returns:
        None
    """
    outputs = anarci.remote(input_fasta, params)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
