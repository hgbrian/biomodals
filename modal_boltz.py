# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs Boltz-1x for protein structure and complex prediction on Modal.

Boltz-1x: https://github.com/jwohlwend/boltz

## Example input, test_boltz.yaml:
```
sequences:
    - protein:
        id: A
        sequence: TDKLIFGKGTRVTVEP
```

## Example usage

Note yaml is highly recommended over fasta input

```
modal run modal_boltz.py --input-yaml test_boltz.yaml
```

Explicitly state the default params:
```
modal run modal_boltz.py --input-yaml test_boltz.yaml --params-str "--use_msa_server --seed 42"
```

Extra params (recommended for best performance):
```
modal run modal_boltz.py --input-yaml test_boltz.yaml --params-str "--use_msa_server --seed 42 --recycling_steps 10 --step_scale 1.0 --diffusion_samples 10"
```
"""

import os
from pathlib import Path

import modal
from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_MODEL_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

ENTITY_TYPES = {"protein", "dna", "rna", "ccd", "smiles"}
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_PARAMS = "--use_msa_server --seed 42"


def download_model():
    """Forces download of the Boltz-1 model by running it once.

    Args:
        None

    Returns:
        None
    """
    from boltz.main import download_boltz1, download_boltz2
    import urllib
    import urllib.request

    if not Path(f"{CACHE_DIR}/boltz1_conf.ckpt").exists():
        print("downloading boltz 1")
        download_boltz1(Path(CACHE_DIR))

    if not Path(f"{CACHE_DIR}/boltz2_conf.ckpt").exists():
        print("downloading boltz 2")
        download_boltz2(Path(CACHE_DIR))


image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git", "gcc", "g++")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@acc0bf772f22feb7f887ad132b7313ff415c8a9f"
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
    # Install CUDA toolkit via conda
    .pip_install(
        "boltz==2.2.1",
        "pyyaml",
        "pandas",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
    )
    .run_function(
        download_model,
        gpu="a10g",
        volumes={f"/{BOLTZ_VOLUME_NAME}": BOLTZ_MODEL_VOLUME},
    )
)

app = App("boltz", image=image)


def fasta_iter(fasta_name):
    """Yields stripped sequence IDs and sequences from a FASTA file.

    Args:
        fasta_name (str): Path to the FASTA file.

    Yields:
        tuple[str, str]: Tuples of (sequence_id, sequence).
    """
    from itertools import groupby

    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def _fasta_to_yaml(input_faa: str) -> str:
    """Converts Boltz FASTA to Boltz YAML format.

    Note: Only basic protein, rna, dna supported for now. Use YAML directly for more complex inputs.

    Args:
        input_faa (str): Path to the input FASTA file.

    Returns:
        str: A string containing the YAML representation of the FASTA content.

    Raises:
        NotImplementedError: If more than 26 chains are present or unsupported entity types/MSA are encountered.
    """
    import re
    import yaml

    chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    yaml_dict = {"sequences": []}

    rx = re.compile(r"^([A-Z])\|([^\|]+)\|?(.*)$")

    for n, (seq_id, seq) in enumerate(fasta_iter(input_faa)):
        if n >= len(chains):
            raise NotImplementedError(">26 chains not supported")

        s_info = rx.search(seq_id)
        if s_info is not None:
            entity_type = s_info.groups()[1].lower()
            if entity_type not in ["protein", "dna", "rna"]:
                raise NotImplementedError(f"Entity type {entity_type} not supported")
            chain_id = s_info.groups()[0].upper()
            if len(s_info.groups()) > 2 and s_info.groups()[2] not in ["", "empty"]:
                raise NotImplementedError("MSA not supported")
        else:
            entity_type = "protein"
            chain_id = chains[n]

        if entity_type == "protein":
            print(entity_type)
            assert all(aa.upper() in ALLOWED_AAS for aa in seq), f"not AAs: {seq}"

        entity = {entity_type: {"id": chain_id, "sequence": seq}}

        yaml_dict["sequences"].append(entity)

    return yaml.dump(yaml_dict, sort_keys=False)


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={f"/{BOLTZ_VOLUME_NAME}": BOLTZ_MODEL_VOLUME},
)
def boltz(input_str: str, params_str: str | None = None) -> list:
    """Runs Boltz on a YAML or FASTA input string.

    The input can describe proteins, DNA, RNA, SMILES strings, or CCD identifiers.

    Args:
        input_str (str): Input content as a string, can be in FASTA or Boltz YAML format.
        params_str (str | None): Optional string of additional parameters to pass to the
                                 `boltz predict` command. Defaults to `DEFAULT_PARAMS`.

    Returns:
        list[tuple[Path, bytes]]: A list of tuples, where each tuple contains the relative
                                  output file path and its byte content.
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    if params_str is None:
        params_str = DEFAULT_PARAMS

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if input_str[0] == ">":
            open(in_faa := Path(in_dir) / "in.faa", "w").write(input_str)
            fixed_faa_str = _fasta_to_yaml(str(in_faa))
            open(fixed_yaml := Path(in_dir) / "fixed.yaml", "w").write(fixed_faa_str)
        else:
            open(fixed_yaml := Path(in_dir) / "fixed.yaml", "w").write(input_str)

        run(
            f'boltz predict "{fixed_yaml}"'
            f' --out_dir "{out_dir}"'
            f' --cache "{CACHE_DIR}"'
            f" {params_str}",
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
    input_faa: str | None = None,
    input_yaml: str | None = None,
    params_str: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/boltz",
):
    """Local entrypoint to run Boltz predictions using Modal.

    Args:
        input_faa (str | None): Path to an input FASTA file.
        input_yaml (str | None): Path to an input Boltz YAML file.
        params_str (str | None): Optional string of additional parameters for the `boltz predict` command.
        run_name (str | None): Optional name for the run, used for organizing output files.
                               If None, a timestamp-based name is used.
        out_dir (str): Directory to save the output files. Defaults to "./out/boltz".

    Returns:
        None

    Raises:
        AssertionError: If neither `input_faa` nor `input_yaml` is provided.
    """
    from datetime import datetime

    assert input_faa or input_yaml, "input_faa or input_yaml required"

    input_str = open(input_yaml or input_faa).read()

    outputs = boltz.remote(input_str, params_str=params_str)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
            out.write(out_content)
