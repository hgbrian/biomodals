# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Run minimap2 on short reads. Mostly just a demo.

Runs minimap2 on short reads.
It takes as input a fasta file and a fastq file, and returns the alignment in PAF format.
"""

import os
from pathlib import Path
from subprocess import run

from modal import App, Image

TIMEOUT = int(os.environ.get("TIMEOUT", 15))

app = App()

image = (
    Image.debian_slim()
    .apt_install("git", "libz-dev")
    .run_commands("git clone https://github.com/lh3/minimap2 && cd minimap2 && make")
)


@app.function(
    image=image,
    gpu=None,
    timeout=60 * TIMEOUT,
)
def minimap2_short_reads(
    input_fasta_bytes: bytes, input_reads_bytes: bytes, params: str
) -> list[tuple[str, bytes]]:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        Path(input_fasta := f"{td_in}/ref.fasta").write_bytes(input_fasta_bytes)
        Path(input_reads := f"{td_in}/reads.fastq").write_bytes(input_reads_bytes)
        out_path = f"{td_out}/{Path(input_fasta).stem}_{Path(input_reads).stem}.paf"

        run(
            f"/minimap2/minimap2 {params} {input_fasta} {input_reads} -o {out_path}",
            shell=True,
            check=True,
        )

        return [(out_path, open(out_path, "rb").read())]


@app.local_entrypoint()
def main(input_ref_fasta: str, input_reads_fastq: str, params: str = "-ax sr"):
    input_fasta_bytes = Path(input_ref_fasta).read_bytes()
    input_reads_bytes = Path(input_reads_fastq).read_bytes()

    outputs = minimap2_short_reads.remote(input_fasta_bytes, input_reads_bytes, params)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_bytes(out_content)
