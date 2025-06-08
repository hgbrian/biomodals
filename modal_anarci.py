# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs ANARCI (Antibody Numbering and Antigen Receptor ClassIfication) on Modal.

ANARCI Usage:
  ANARCI [-h] [--sequence INPUTSEQUENCE] [--outfile OUTFILE] [--scheme {m,c,k,imgt,kabat,chothia,martin,i,a,aho,wolfguy,w}]
                [--restrict {ig,tr,heavy,light,H,K,L,A,B} [{ig,tr,heavy,light,H,K,L,A,B} ...]] [--csv] [--outfile_hits HITFILE] [--hmmerpath HMMERPATH] [--ncpu NCPU]
                [--assign_germline] [--use_species {human,mouse,rat,rabbit,rhesus,pig,alpaca,cow}] [--bit_score_threshold BIT_SCORE_THRESHOLD]

For more details, see https://github.com/oxpig/ANARCI
"""

from pathlib import Path
from subprocess import run
from datetime import datetime
import tempfile

from modal import App, Image

image = (
    Image.micromamba()
    .apt_install("git")
    .pip_install("biopython")
    .micromamba_install(["libstdcxx-ng", "hmmer=3.3.2"], channels=["conda-forge", "bioconda"])
    .run_commands(
        "git clone https://github.com/oxpig/ANARCI && cd ANARCI && python setup.py install"
    )
)

app = App("anarci", image=image)


@app.function(
    timeout=60 * 15,
    # mounts removed
)
def anarci(input_str: str, params: str | None = None) -> list[tuple[str, bytes]]:
    """Runs ANARCI on a given FASTA sequence string using specified parameters.

    Args:
        input_str (str): A raw FASTA sequence string.
        params (str | None): Optional string of additional parameters to pass to the ANARCI
                             command. If None, defaults to "--csv --ncpu 2".

    Returns:
        list[tuple[str, bytes]]: A list of tuples, where each tuple contains the basename
                                 of the output file and its byte content.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_file = tmp_path / "input.fasta"
        output_file_stem = tmp_path / "anarci_results" # ANARCI will add .csv, .txt etc.

        with open(input_file, "w") as f:
            f.write(input_str)

        command = f"ANARCI -i {input_file} --outfile {output_file_stem}"

        if params is not None:
            command += " " + params
        else:
            # Default to CSV output as it's a common structured format
            command += " --csv --ncpu 2"

        run(command, shell=True, check=True)

        # ANARCI can create multiple output files (e.g., .csv, .txt for numbering, aln for alignment)
        # and also subdirectories like IMGT_output for some schemes.
        # We glob for all files in the temporary directory.
        output_files = []
        for out_file_path in tmp_path.glob("**/*"):
            if out_file_path.is_file():
                 # The problem asks for the basename of the output file.
                 # If ANARCI creates subdirectories, we should probably preserve that structure
                 # in the name, or make it flat. The previous version of main was doing Path(out_file_name).name
                 # which flattens. Let's keep it simple and return just the file's direct name.
                 # If ANARCI puts files in "IMGT_output/A.csv", this will return "A.csv".
                 # The main function will then save it as out_dir_full / "A.csv".
                output_files.append((out_file_path.name, out_file_path.read_bytes()))

        return output_files


@app.local_entrypoint()
def main(
    input_fasta: str,
    params: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/anarci",
):
    """Local entrypoint to run ANARCI via Modal.

    This function takes an input FASTA file path, optional ANARCI
    parameters, an optional run name, and an output directory. It then calls
    the remote Modal `anarci` function with the file content and saves the
    results locally, structured by `run_name` and `out_dir`.

    Args:
        input_fasta (str): Path to the input FASTA file.
        params (str | None): Optional string of additional parameters for ANARCI.
        run_name (str | None): Optional name for the run, used for organizing output files.
                               If None, a timestamp-based name is used.
        out_dir (str): Directory to save the output files. Defaults to "./out/anarci".

    Returns:
        None
    """
    input_fasta_path = Path(input_fasta)
    input_content = input_fasta_path.read_text()

    outputs = anarci.remote(input_content, params)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file_name, out_content in outputs:
        # Construct the full output path, ensuring the filename from ANARCI is used.
        # ANARCI output files might be in subdirectories (e.g. "IMGT_output/A.csv"),
        # so we take the filename part from the returned out_file_name.
        output_file_path = out_dir_full / Path(out_file_name).name
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(output_file_path, "wb") as out:
                out.write(out_content)
