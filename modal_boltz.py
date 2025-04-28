"""
Boltz-1x https://github.com/jwohlwend/boltz

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

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_MODEL_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

ENTITY_TYPES = {"protein", "dna", "rna", "ccd", "smiles"}
ALLOWED_AAS = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_PARAMS = "--use_msa_server --seed 42"


def download_model():
    """Force download of the Boltz-1 model by running it once"""
    from subprocess import run

    if not Path(f"/{BOLTZ_VOLUME_NAME}/boltz1_conf.ckpt").exists():
        Path(in_dir := "/tmp/tmp_in_boltz").mkdir(parents=True, exist_ok=True)
        open(in_faa := Path(in_dir) / "tmp.fasta", "w").write(
            ">A|PROTEIN|\nMAWTPLLLLLLSH\n"
        )

        run(
            [
                "boltz",
                "predict",
                str(in_faa),
                "--out_dir",
                "/tmp",
                "--cache",
                CACHE_DIR,
                "--use_msa_server",
            ],
            check=True,
        )

    # New: Copy it over since "/root/.boltz" is default in boltz
    if CACHE_DIR != "/root/.boltz":
        Path("/root/.boltz").mkdir(exist_ok=True, parents=True)
        run(f"cp {CACHE_DIR}/* /root/.boltz/", shell=True, check=True)


image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
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
    .pip_install("boltz", "pyyaml")
    .run_function(
        download_model,
        gpu="a10g",
        volumes={f"/{BOLTZ_VOLUME_NAME}": BOLTZ_MODEL_VOLUME},
    )
)

app = App("boltz", image=image)


def fasta_iter(fasta_name):
    """yield stripped seq_ids and seqs"""
    from itertools import groupby

    with open(fasta_name) as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def _fasta_to_yaml(input_faa: str) -> str:
    """Convert Boltz FASTA to Boltz YAML format.

    Only basic protein, rna, dna supported for now.
    Just use yaml if you want real flexibility.
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
    """Runs Boltz on a yaml or fasta.
    File can contain protein, DNA, RNA, smiles, ccd.
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    if params_str is None:
        params_str = DEFAULT_PARAMS

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if input_str[0] == ">":
            open(in_faa := Path(in_dir) / "in.faa", "w").write(input_str)
            fixed_faa_str = _fasta_to_yaml(in_faa)
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
    from datetime import datetime

    # New: Check that at least one input is provided
    assert input_faa or input_yaml, "input_faa or input_yaml required"

    input_str = open(input_yaml or input_faa).read()

    outputs = boltz.remote(input_str, params_str=params_str)  # New: Use boltz_from_file

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
            out.write(out_content)
