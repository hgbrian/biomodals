# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Run OmegaFold on an amino acid fasta file to produce a pdb file.

Mostly defunct now with boltz and chai.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from subprocess import run

import modal

# Configuration
GPU = os.environ.get("GPU", "a100")
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

# Create Modal app
app = modal.App("omegafold")

# Define the image
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "git+https://github.com/HeliXonProtein/OmegaFold.git"
    )
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * TIMEOUT,
)
def omegafold(input_fasta_content: bytes, fasta_name: str, subbatch_size: int) -> list[tuple[str, bytes]]:
    """Run OmegaFold protein structure prediction.

    Args:
        input_fasta_content: Content of the input FASTA file containing protein sequence
        fasta_name: Name of the input FASTA file
        subbatch_size: Batch size for model processing

    Returns:
        List of tuples containing (file_path, file_content) for generated PDB files
    """
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        # Create directories
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write input FASTA
        input_path = input_dir / fasta_name
        with open(input_path, "wb") as f:
            f.write(input_fasta_content)

        # Run OmegaFold
        run(
            f"omegafold --model 2 --subbatch_size {subbatch_size} {input_path} {output_dir}",
            shell=True,
            check=True,
        )

        # Collect results
        results = []
        for pdb_file in output_dir.glob("**/*.pdb"):
            out_file = str(pdb_file.relative_to(output_dir))
            with open(pdb_file, "rb") as f:
                results.append((out_file, f.read()))

        return results


@app.local_entrypoint()
def main(
    input_fasta: str,
    subbatch_size: int = 224,
    out_dir: str = "./out/omegafold",
    run_name: str | None = None
):
    """Local entrypoint for the Modal app to run OmegaFold.

    Args:
        input_fasta: Path to the input FASTA file
        subbatch_size: Batch size for model processing
        out_dir: Directory to save the output files
        run_name: Optional name for the run, used to create a subdirectory in out_dir.
                 If None, a timestamp-based name is used.
    """
    # Validate input file
    input_path = Path(input_fasta)
    assert input_path.suffix in (".faa", ".fasta"), f"not a fasta file: {input_path}"

    # Read input file content
    with open(input_path, "rb") as f:
        input_content = f.read()

    # Run OmegaFold remotely
    outputs = omegafold.remote(
        input_content,
        input_path.name,
        subbatch_size
    )

    # Create output directory with run name or timestamp
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    # Write outputs
    for out_file, out_content in outputs:
        out_path = out_dir_full / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as out:
            out.write(out_content)
        print(f"Saved output to {out_path}")
