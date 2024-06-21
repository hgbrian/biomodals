"""Run minimap2 on short reads. Mostly just a demo.

Runs minimap2 on short reads.
It takes as input a fasta file and a fastq file, and returns the alignment in PAF format.
"""

import os
from pathlib import Path
from subprocess import run

from modal import App, Image, Mount

FORCE_BUILD = False
LOCAL_IN = "./in/minimap2"
LOCAL_OUT = "./out/minimap2"
REMOTE_IN = "/in"
REMOTE_OUT = LOCAL_OUT
TIMEOUT_MINS = int(os.environ.get("TIMEOUT_MINS", 15))

app = App()

image = (
    Image.debian_slim()
    .apt_install("git", "libz-dev")
    .run_commands("git clone https://github.com/lh3/minimap2 && cd minimap2 && make")
)


@app.function(
    image=image,
    gpu=None,
    timeout=60 * TIMEOUT_MINS,
    mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)],
)
def minimap2_short_reads(input_fasta: str, input_reads: str, params: tuple) -> list[str, str]:
    input_fasta = Path(input_fasta).relative_to(LOCAL_IN)
    input_reads = Path(input_reads).relative_to(LOCAL_IN)

    Path(REMOTE_OUT).mkdir(parents=True, exist_ok=True)
    out_path = Path(REMOTE_OUT) / f"{input_fasta.stem}_{input_reads.stem}.paf"

    run(
        f"/minimap2/minimap2 {params} {Path(REMOTE_IN) / input_fasta} {Path(REMOTE_IN) / input_reads} -o {out_path}",
        shell=True,
        check=True,
    )

    return [(out_path, open(out_path, "rb").read())]


@app.local_entrypoint()
def main(input_fasta: str, input_reads: str, params: str = "-ax sr"):
    outputs = minimap2_short_reads.remote(input_fasta, input_reads, params)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
