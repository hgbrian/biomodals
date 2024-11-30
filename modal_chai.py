"""Chai-1r https://github.com/chaidiscovery/chai-lab

Example fasta
```
>protein|name=insulin
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS
>ligand|name=caffeine
CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```
```
modal run modal_chai.py --input-faa test_chai.faa
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))


def download_models():
    """Runs Chai on a fasta file and returns the outputs"""
    import torch
    from chai_lab.chai1 import run_inference

    with open("/tmp/tmp.faa", "w") as out:
        out.write(
            ">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n"
            ">ligand|name=lig\nCC\n"
        )

    _ = run_inference(
        fasta_file=Path("/tmp/tmp.faa"),
        output_dir=Path("/tmp"),
        num_trunk_recycles=1,
        num_diffn_timesteps=10,
        seed=1,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )


image = (
    Image.debian_slim()
    .apt_install("wget")
    .pip_install("uv")
    .run_commands("uv pip install --system --compile-bytecode chai_lab")
    .run_function(download_models, gpu="a100")
)

app = App("chai", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def chai(
    input_faa_str: str,
    input_faa_name: str = "input.faa",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    use_esm_embeddings: bool = True,
    chai_kwargs:dict = {},
) -> list:
    """Runs Chai on a fasta file and returns the outputs"""
    import torch
    from chai_lab.chai1 import run_inference

    Path(in_dir := "/tmp/in_chai").mkdir(parents=True, exist_ok=True)
    Path(out_dir := "/tmp/out_chai").mkdir(parents=True, exist_ok=True)

    fasta_path = Path(in_dir) / input_faa_name
    fasta_path.write_text(input_faa_str)

    _ = run_inference(
        fasta_file=Path(fasta_path),
        output_dir=Path(out_dir),
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        seed=seed,
        device=torch.device("cuda:0"),
        use_esm_embeddings=use_esm_embeddings,
        **chai_kwargs,
    )

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if Path(out_file).is_file()
    ]


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/chai",
    run_name: str = None,
    chai_kwargs: str = None,
):
    from datetime import datetime

    input_faa_str = open(input_faa).read()

    outputs = chai.remote(
        input_faa_str,
        input_faa_name=Path(input_faa).name,
        chai_kwargs=dict(eval(chai_kwargs)) if chai_kwargs else {},
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        (Path(out_dir_full) / Path(out_file)).write_bytes(out_content)
