from subprocess import run
from pathlib import Path

from modal import Image, Mount, Stub

FORCE_BUILD = False
MODAL_IN = "./modal_in/minimap2"
MODAL_OUT = "./modal_out/minimap2"

stub = Stub()

image = (Image
         .debian_slim()
         .apt_install("git", "libz-dev")
         .run_commands("git clone https://github.com/lh3/minimap2 && cd minimap2 && make")
        )

@stub.function(image=image, gpu=None, timeout=60*15,
               mounts=[Mount.from_local_dir(MODAL_IN, remote_path="/in")])
def minimap2_short_reads(input_fasta:str, input_reads:str) -> list[str, str]:
    input_fasta = Path(input_fasta)
    input_reads = Path(input_reads)
    assert input_fasta.parent.resolve() == Path(MODAL_IN).resolve(), f"wrong input_fasta dir {input_fasta.parent}"
    assert input_reads.parent.resolve() == Path(MODAL_IN).resolve(), f"wrong input_reads dir {input_reads.parent}"
    out_name = f"{MODAL_OUT}/{input_fasta.stem}_{input_reads.stem}.aln"

    run(["mkdir", "-p", MODAL_OUT], check=True)
    run(["/minimap2/minimap2", "-ax", "sr", f"/in/{input_fasta.name}", f"/in/{input_reads.name}", "-o", out_name], check=True)

    return [(out_name, open(out_name, "rb").read())]

@stub.local_entrypoint()
def main(input_fasta, input_reads):
    outputs = minimap2_short_reads.remote(input_fasta, input_reads)

    for (out_file, out_content) in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, 'wb') as out:
                out.write(out_content)
