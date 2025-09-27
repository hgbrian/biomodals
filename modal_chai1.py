# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs Chai-1, a protein-ligand co-folding model, on Modal.

Chai-1r: https://github.com/chaidiscovery/chai-lab

Example fasta
```
>protein|name=insulin
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS
>ligand|name=caffeine
CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```
```
modal run modal_chai1.py --input-faa test_chai1.faa
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 30))


def download_models():
    """Downloads Chai-1 models by running a minimal inference.

    Args:
        None

    Returns:
        None
    """
    import torch
    from tempfile import TemporaryDirectory
    from chai_lab.chai1 import run_inference

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        with open(f"{td_in}/tmp.faa", "w") as out:
            out.write(
                ">protein|name=pro\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSL\n"
                ">ligand|name=lig\nCC\n"
            )

        _ = run_inference(
            fasta_file=Path(f"{td_in}/tmp.faa"),
            output_dir=Path(f"{td_out}"),
            num_trunk_recycles=1,
            num_diffn_timesteps=10,
            seed=1,
            device=torch.device("cuda:0"),
            use_esm_embeddings=True,
        )


image = (
    Image.debian_slim()
    .apt_install("wget")
    .uv_pip_install("chai_lab==0.6.1")
    .run_function(download_models, gpu="a100")
)

app = App("chai1", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def chai1(
    input_faa_str: str,
    input_faa_name: str = "input.faa",
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    seed: int = 42,
    use_esm_embeddings: bool = True,
    chai1_kwargs: dict = {},
) -> list:
    """Runs Chai-1 on a FASTA file string and returns the output files.

    Args:
        input_faa_str (str): Content of the input FASTA file as a string.
        input_faa_name (str): Original name of the input FASTA file (for naming outputs).
                              Defaults to "input.faa".
        num_trunk_recycles (int): Number of trunk recycles for the model. Defaults to 3.
        num_diffn_timesteps (int): Number of diffusion timesteps. Defaults to 200.
        seed (int): Random seed for reproducibility. Defaults to 42.
        use_esm_embeddings (bool): Whether to use ESM embeddings. Defaults to True.
        chai1_kwargs (dict): Additional keyword arguments to pass to `run_inference`.
                             Defaults to an empty dict.

    Returns:
        list[tuple[Path, bytes]]: A list of tuples, where each tuple contains the relative
                                  output file path and its byte content.
    """
    import torch
    from tempfile import TemporaryDirectory
    from chai_lab.chai1 import run_inference

    with TemporaryDirectory() as td_in, TemporaryDirectory() as td_out:
        fasta_path = Path(td_in) / input_faa_name
        fasta_path.write_text(input_faa_str)

        _ = run_inference(
            fasta_file=Path(fasta_path),
            output_dir=Path(td_out),
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            seed=seed,
            device=torch.device("cuda:0"),
            use_esm_embeddings=use_esm_embeddings,
            **chai1_kwargs,
        )

        return [
            (out_file.relative_to(td_out), open(out_file, "rb").read())
            for out_file in Path(td_out).glob("**/*")
            if Path(out_file).is_file()
        ]


@app.local_entrypoint()
def main(
    input_faa: str,
    out_dir: str = "./out/chai1",
    run_name: str | None = None,
    chai1_kwargs: str | None = None,
):
    """Local entrypoint for running Chai-1 predictions using Modal.

    Args:
        input_faa (str): Path to the input FASTA file.
        out_dir (str): Directory to save the output files. Defaults to "./out/chai1".
        run_name (str | None): Optional name for the run, used in the output directory structure.
        chai1_kwargs (str | None): Optional string representation of a dictionary for additional
                                   Chai-1 keyword arguments (e.g., '{"key": "value"}').

    Returns:
        None
    """
    from datetime import datetime

    input_faa_str = open(input_faa).read()

    outputs = chai1.remote(
        input_faa_str,
        input_faa_name=Path(input_faa).name,
        chai1_kwargs=dict(eval(chai1_kwargs)) if chai1_kwargs else {},
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        (Path(out_dir_full) / Path(out_file)).write_bytes(out_content)
