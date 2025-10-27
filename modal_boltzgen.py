"""BoltzGen https://github.com/HannesStark/boltzgen

Example yaml file:
```yaml
entities:
  - protein:
      id: B
      sequence: 80..140
  - file:
      path: 6m1u.cif
      include:
        - chain:
            id: A
```

Example usage:
```bash
wget https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13.cif
wget https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13prot.yaml
modal run modal_boltzgen.py --input-yaml 1g13prot.yaml --protocol protein-anything --num-designs 2
```

Available protocols: protein-anything, peptide-anything, protein-small_molecule, nanobody-anything

Other useful options:
  --steps design inverse_folding folding    # Run specific steps only
  --cache /path/to/cache                    # Custom cache directory
  --devices 2                               # Number of GPUs to use
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 120))


def download_boltzgen_models():
    """Download all boltzgen models during image build to avoid runtime timeouts."""
    import subprocess

    # Download all artifacts to default cache location (~/.cache)
    print("Downloading boltzgen models...")
    subprocess.run(
        ["boltzgen", "download", "all"],
        check=True,
    )
    print("Model download complete")


image = (
    Image.debian_slim()
    .apt_install("git", "wget", "build-essential")
    .pip_install("torch>=2.4.1")
    .run_commands(
        "git clone https://github.com/HannesStark/boltzgen /root/boltzgen",
        "cd /root/boltzgen && git checkout 58c1eed2b07f00fd5263f78fe2821c80d6875699 && pip install -e .",
        gpu="a10g",
    )
    .run_function(
        download_boltzgen_models,
        gpu="a10g",
    )
)

app = App("boltzgen", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def boltzgen_run(
    yaml_str: str,
    yaml_name: str,
    additional_files: dict[str, bytes],
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    cache: str | None = None,
    devices: int | None = None,
    extra_args: str | None = None,
) -> list:
    """Run BoltzGen on a yaml specification.

    Args:
        yaml_str: YAML design specification as string
        yaml_name: Name of the yaml file
        additional_files: Dict of relative_path -> file_content for referenced files
        protocol: Design protocol (protein-anything, peptide-anything, etc.)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        cache: Custom cache directory path
        devices: Number of GPUs to use
        extra_args: Additional CLI arguments as string

    Returns:
        List of (path, content) tuples for all output files
    """
    from subprocess import run
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as in_dir, TemporaryDirectory() as out_dir:
        # Write yaml to file
        yaml_path = Path(in_dir) / yaml_name
        yaml_path.write_text(yaml_str)

        # Write any additional files (e.g., .cif files referenced in yaml)
        for rel_path, content in additional_files.items():
            file_path = Path(in_dir) / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)

        # Build command
        cmd = [
            "boltzgen",
            "run",
            str(yaml_path),
            "--output",
            out_dir,
            "--protocol",
            protocol,
            "--num_designs",
            str(num_designs),
        ]

        if steps:
            cmd.extend(["--steps"] + steps.split())
        if cache:
            cmd.extend(["--cache", cache])
        if devices:
            cmd.extend(["--devices", str(devices)])
        if extra_args:
            cmd.extend(extra_args.split())

        print(f"Running: {' '.join(cmd)}")
        run(cmd, check=True)

        # Collect all output files
        return [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in Path(out_dir).rglob("*")
            if out_file.is_file()
        ]


@app.local_entrypoint()
def main(
    input_yaml: str,
    protocol: str = "protein-anything",
    num_designs: int = 10,
    steps: str | None = None,
    cache: str | None = None,
    devices: int | None = None,
    extra_args: str | None = None,
    out_dir: str = "./out/boltzgen",
    run_name: str | None = None,
):
    """Run BoltzGen locally with results saved to out_dir.

    Args:
        input_yaml: Path to YAML design specification file
        protocol: Design protocol (protein-anything, peptide-anything, protein-small_molecule, nanobody-anything)
        num_designs: Number of designs to generate
        steps: Specific pipeline steps to run (e.g. "design inverse_folding")
        cache: Custom cache directory path
        devices: Number of GPUs to use
        extra_args: Additional CLI arguments as string
        out_dir: Local output directory
        run_name: Optional run name (defaults to timestamp)
    """
    import re
    from datetime import datetime

    yaml_path = Path(input_yaml)
    yaml_str = yaml_path.read_text()
    yaml_dir = yaml_path.parent

    # Find any file references in the yaml (path: something.cif)
    # File paths in yaml are relative to the yaml file location
    additional_files = {}
    for match in re.finditer(r"path:\s*([^\s\n]+)", yaml_str):
        ref_file = match.group(1)
        ref_path = yaml_dir / ref_file
        if ref_path.exists():
            additional_files[ref_file] = ref_path.read_bytes()
            print(f"Including referenced file: {ref_file}")

    outputs = boltzgen_run.remote(
        yaml_str=yaml_str,
        yaml_name=yaml_path.name,
        additional_files=additional_files,
        protocol=protocol,
        num_designs=num_designs,
        steps=steps,
        cache=cache,
        devices=devices,
        extra_args=extra_args,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        output_path = out_dir_full / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"\nResults saved to: {out_dir_full}")
