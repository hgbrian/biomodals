# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Protenix https://github.com/bytedance/Protenix

Protenix is an open-source PyTorch reproduction of AlphaFold 3.

## Example FASTA input (test_protenix.faa):
```
>protein|A
MAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS
```

## Usage:
```
modal run modal_protenix.py --input-faa test_protenix.faa
```

With custom parameters:
```
modal run modal_protenix.py --input-faa test_protenix.faa --seeds "42,43" --use-msa
```

## Advanced: JSON input (native Protenix format)
```json
[
    {
        "name": "test_protein",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL",
                    "count": 1
                }
            }
        ]
    }
]
```

Usage with JSON:
```
modal run modal_protenix.py --input-json test_protenix.json
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))

ENTITY_TYPES = {"protein", "dna", "rna", "ligand", "ion"}

ENTITY_TYPE_MAP = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
    "ligand": "ligand",
    "ion": "ion",
}

DEFAULT_MODEL = "protenix_base_20250630_v1.0.0"
DEFAULT_SEEDS = "42"


def download_models():
    """Force download/initialization of Protenix models by running once."""
    from subprocess import run
    from tempfile import TemporaryDirectory

    Path(out_dir := "/tmp/out_dm").mkdir(parents=True, exist_ok=True)

    test_json = """[
    {
        "name": "test",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MAWTPLLLLLLSHCTGSLSQPVLTQPTSL",
                    "count": 1
                }
            }
        ]
    }
]"""

    with TemporaryDirectory() as td:
        json_path = Path(td) / "test.json"
        json_path.write_text(test_json)

        print("Downloading base model...")
        run(
            f'protenix pred --input "{json_path}" --out_dir "{out_dir}" '
            f"--seeds 42 --use_msa false --model_name {DEFAULT_MODEL}",
            shell=True,
            check=True,
        )

    print("Model download complete")


image = (
    Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "wget", "g++", "gcc", "make", "clang")
    .pip_install("protenix==1.0.4")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .run_function(download_models, gpu="A10G")
)

app = App("protenix", image=image)


def _fasta_to_json(input_faa: str, name: str = "input") -> str:
    """Convert FASTA to Protenix JSON format.

    Expected FASTA format:
    >protein|A|description
    SEQUENCE
    >dna|B|description
    ATCG
    >ligand|D|CCD_NAME
    """
    import json
    import re

    rx = re.compile(r"^([^|]+)\|([^|]*)\|?(.*)$")
    sequences = []

    # Inline FASTA parsing
    current_id = None
    current_seq: list[str] = []
    for line in input_faa.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                sequences.append((current_id, "".join(current_seq)))
            current_id = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)
    if current_id is not None:
        sequences.append((current_id, "".join(current_seq)))

    json_sequences = []
    for seq_id, seq in sequences:
        match = rx.search(seq_id)
        entity_type = match.group(1).lower() if match else "protein"

        if entity_type not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {entity_type}. Must be one of {ENTITY_TYPES}")

        protenix_type = ENTITY_TYPE_MAP[entity_type]

        if entity_type == "ligand":
            entity = {protenix_type: {"ligand": seq, "count": 1}}
        else:
            entity = {protenix_type: {"sequence": seq, "count": 1}}

        json_sequences.append(entity)

    return json.dumps([{"name": name, "sequences": json_sequences}], indent=2)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def protenix(
    input_str: str,
    input_name: str = "input",
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    model_name: str = DEFAULT_MODEL,
) -> list:
    """Run Protenix on FASTA or JSON input and return output files.

    Args:
        input_str: FASTA or JSON content.
        input_name: Name for prediction job.
        seeds: Comma-separated seeds.
        use_msa: Use MSA.
        model_name: Model name.

    Returns:
        list[tuple[Path, bytes]]: Output files as (relative_path, bytes) tuples.
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        if input_str.strip().startswith(">"):
            json_str = _fasta_to_json(input_str, name=input_name)
            print("Converted FASTA to JSON:")
            print(json_str)
        else:
            json_str = input_str

        json_path = Path(in_dir) / "input.json"
        json_path.write_text(json_str)

        print(f"Running Protenix with model={model_name}, seeds={seeds}, use_msa={use_msa}")

        run(
            f'protenix pred --input "{json_path}" '
            f'--out_dir "{out_dir}" '
            f"--seeds {seeds} "
            f"--use_msa {str(use_msa).lower()} "
            f"--model_name {model_name} "
            f"--enable_cache true --enable_fusion true",
            shell=True,
            check=True,
        )

        return [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in Path(out_dir).glob("**/*")
            if out_file.is_file()
        ]


@app.local_entrypoint()
def main(
    input_faa: str | None = None,
    input_json: str | None = None,
    seeds: str = DEFAULT_SEEDS,
    use_msa: bool = True,
    model_name: str = DEFAULT_MODEL,
    use_mini: bool = False,
    out_dir: str | None = None,
    run_name: str | None = None,
):
    """Run Protenix predictions.

    Args:
        input_faa: Path to FASTA file.
        input_json: Path to JSON file (native Protenix format).
        seeds: Comma-separated seed values.
        use_msa: Whether to use MSA.
        model_name: Model name.
        use_mini: Use mini model variant.
        out_dir: Output directory.
        run_name: Run name subdirectory (timestamp if None).
    """
    from datetime import datetime

    assert input_faa or input_json, "Either input_faa or input_json required"

    if use_mini:
        model_name = "protenix_mini_default_v0.5.0"
        out_dir = out_dir or "./out/protenix_mini"
    else:
        out_dir = out_dir or "./out/protenix"

    input_path = input_faa or input_json
    input_name = Path(input_path).stem

    outputs = protenix.remote(
        input_str=open(input_path).read(),
        input_name=input_name,
        seeds=seeds,
        use_msa=use_msa,
        model_name=model_name,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        out_path = Path(out_dir_full) / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(out_content or b"")

    print(f"Output saved to: {out_dir_full}")
