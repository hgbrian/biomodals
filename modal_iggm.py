"""IgGM - A Generative Model for Functional Antibody and Nanobody Design
https://github.com/TencentAI4S/IgGM

CRITICAL FASTA FORMAT REQUIREMENTS:
- FASTA file must contain 1-3 sequences
- When using antigen: THE LAST sequence header MUST specify the antigen chain ID
- Use 'X' characters to mark regions for design (e.g., CDR loops, framework regions)
- Chain identifiers: >H (heavy), >L (light), >A (antigen chain A), >B (antigen chain B), etc.

Example FASTA formats:
```
# Nanobody only (no antigen)
>H
QVQLVESGGGLVQPGGSLRLSCAASGFTFSXXXXXXXXXXXXXXXXXXXTRV...

# Antibody with antigen (antigen chain A in PDB)
>H
VQLVESGGGLVQPGGSLRLSCAASXXXXXXYMNWVRQAPGKGLEWVSAIG...
>L
DIQMTQSPSSLSASVGDRVTITCXXXXXXWYQQKPGKAPKLLIYKASSLES...
>A
(This tells IgGM to use chain A from the antigen PDB file)

# Single chain with antigen (antigen chain B in PDB)
>H
QVQLVESGGGLVQPGGSLRLSCAASGFTFSXXXXXXXXXXXXXXXXXXXTRV...
>B
(This tells IgGM to use chain B from the antigen PDB file)
```

Example usage:
```
# Nanobody design with antigen
# FASTA: >H\nsequence...\n>A\n(empty, just specifies chain A)
uv run modal run modal_iggm.py --input-fasta nanobody.fasta --antigen antigen.pdb --epitope "41,42,43" --task design

# Antibody design with antigen
# FASTA: >H\nsequence...\n>L\nsequence...\n>A\n(empty, just specifies chain A)
uv run modal run modal_iggm.py --input-fasta antibody.fasta --antigen antigen.pdb --epitope "126,127,129" --task design

# Structure prediction only (no antigen)
uv run modal run modal_iggm.py --input-fasta antibody.fasta --task design
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "A10")
TIMEOUT = int(os.environ.get("TIMEOUT", 300))

REQUIRED_INPUT_COLS = ["seq_id", "seq"]
APPS_BUCKET_NAME = "apps"

VALID_TASKS = {
    "design",
    "inverse_design",
    "affinity_maturation",
    "fr_design",  # framework redesign
}


def download_models():
    """I am sure there is an easier way to do this."""
    import importlib.util

    pretrain_path = "/root/IgGM/IgGM/model/pretrain.py"
    spec = importlib.util.spec_from_file_location("pretrain", pretrain_path)
    pretrain = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pretrain)

    os.chdir("/root/IgGM")
    pretrain.load_model_hub("esm_ppi_650m_ab")
    pretrain.load_model_hub("antibody_design_trunk")
    pretrain.load_model_hub("antibody_inverse_design_trunk")
    pretrain.load_model_hub("antibody_fr_design_trunk")
    pretrain.load_model_hub("igso3_buffer")


image = (
    Image.micromamba(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.1.2+cu121",
        "torchvision==0.16.2+cu121",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "torch_geometric==2.5.2",
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        find_links="https://data.pyg.org/whl/torch-2.1.0+cu121.html",
    )
    .pip_install(
        "numpy==1.23.5",
        "pandas==2.2.2",
        "scipy==1.14.0",
        "scikit-learn==1.5.1",
        "matplotlib==3.9.1",
        "seaborn==0.13.2",
        "termcolor==3.1.0",
        "absl-py==2.1.0",
        "biopython==1.85",
        "polars==1.19.0",
        "openmm==8.3.1",
    )
    .pip_install("ml-collections")
    .micromamba_install("pdbfixer", channels=["conda-forge"])
    .run_commands("git clone https://github.com/TencentAI4S/IgGM.git", "mv IgGM /root/")
    .run_commands(
        # Replace torch.from_numpy with torch.tensor to avoid numpy conversion issues
        # error is "TypeError: expected np.ndarray (got numpy.ndarray)"!?
        "cd /root/IgGM && sed -i 's/torch.from_numpy(residue\\[atom_name\\].get_coord())/torch.tensor(residue[atom_name].get_coord(), dtype=torch.float32)/' IgGM/protein/parser/pdb_parser.py"
    )
    .run_function(download_models)
    .pip_install("prody==2.6.1")
    .pip_install("ipython")
)

app = App("iggm", image=image)


def merge_pdb_chains(pdb_str: str, chains_to_merge: list) -> str:
    """
    Merge multiple chains in a PDB structure into the first chain in the list.

    Parameters:
    -----------
    pdb_str : str
        PDB structure as a string
    chains_to_merge : list
        List of chain IDs to combine (e.g., ['B', 'C']).
        All chains will be merged into the first chain ID in the list.

    Returns:
    --------
    str
        Modified PDB structure as a string with chains combined
    """
    import prody as pr
    from io import StringIO

    if not chains_to_merge or len(chains_to_merge) < 2:
        raise ValueError("Need at least 2 chains to combine")

    structure = pr.parsePDBStream(StringIO(pdb_str))

    target_chain = chains_to_merge[0]  # First chain becomes the target

    # Collect atoms from all chains to combine
    chain_atoms = []
    current_resnum = 1

    for chain_id in chains_to_merge:
        chain = structure.select(f"chain {chain_id}")
        if chain is not None:
            # Get unique residue numbers and renumber sequentially
            unique_resnums = sorted(set(chain.getResnums()))
            resnum_mapping = {
                old: current_resnum + i for i, old in enumerate(unique_resnums)
            }

            # Apply new residue numbers and set chain ID to target
            new_resnums = [resnum_mapping[old] for old in chain.getResnums()]
            chain.setResnums(new_resnums)
            chain.setChids([target_chain] * len(chain))

            chain_atoms.append(chain)
            current_resnum += len(unique_resnums)

    if not chain_atoms:
        raise ValueError(
            f"None of the specified chains {chains_to_merge} found in structure"
        )

    # Combine all selected chains
    combined_chains = chain_atoms[0]
    for chain in chain_atoms[1:]:
        combined_chains = combined_chains + chain

    # Add other chains that weren't combined
    other_chains_selection = f"not chain {' '.join(chains_to_merge)}"
    other_chains = structure.select(other_chains_selection)

    if other_chains is not None:
        final_structure = other_chains + combined_chains
    else:
        final_structure = combined_chains

    # Convert back to string
    output_stream = StringIO()
    pr.writePDBStream(output_stream, final_structure)
    return output_stream.getvalue()


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def iggm(
    input_fasta_str: str,
    task: str,
    antigen_pdb_str: str | None = None,
    epitope: list[int] | None = None,
    fasta_origin_str: str | None = None,
    num_samples: int | None = None,
    relax: bool = False,
    max_antigen_size: int | None = None,
) -> list:
    """Runs IgGM on a fasta file and returns the outputs.

    Args:
        input_fasta_str: Input antibody/nanobody sequence
        task: Design task (design, inverse_design, affinity_maturation, etc.)
        antigen_pdb_str: Antigen structure (optional)
        epitope: List of epitope residue numbers (optional)
        fasta_origin_str: Original sequence for affinity maturation (optional)
        num_samples: Number of samples to generate (optional)
        relax: Whether to perform structure relaxation
        max_antigen_size: Maximum antigen length to consider
    """
    from io import StringIO
    from subprocess import run
    from tempfile import TemporaryDirectory
    import prody as pr

    assert task in VALID_TASKS, f"Task must be one of {VALID_TASKS}"

    with TemporaryDirectory() as work_dir:
        # Write input fasta
        fasta_path = Path(work_dir) / "input.fasta"
        fasta_path.write_text(input_fasta_str)

        # Build command
        cmd = ["python", "/root/IgGM/design.py", "--fasta", str(fasta_path)]

        if antigen_pdb_str:
            # Merge chains for IgGM if necessary
            structure_ = pr.parsePDBStream(StringIO(antigen_pdb_str))
            chains_ = structure_.getChids().tolist()
            uniq_chains_ = list(dict.fromkeys(chains_))
            if len(uniq_chains_) > 2:
                print(f"WARNING: combining chains {uniq_chains_[1:]} for IgGM")
                print(f"WARNING: chain {uniq_chains_[0]} is the binder")
                antigen_pdb_str = merge_pdb_chains(antigen_pdb_str, uniq_chains_[1:])

            antigen_path = Path(work_dir) / "antigen.pdb"
            antigen_path.write_text(antigen_pdb_str)
            cmd.extend(["--antigen", str(antigen_path)])

        # Add epitope residues if provided
        if epitope:
            cmd.extend(["--epitope"] + [str(res) for res in epitope])

        # Add original sequence for affinity maturation
        if fasta_origin_str:
            orig_path = Path(work_dir) / "original.fasta"
            orig_path.write_text(fasta_origin_str)
            cmd.extend(["--fasta_origin", str(orig_path)])

        # Add task
        cmd.extend(["--run_task", task])

        # Add other parameters
        if num_samples:
            cmd.extend(["--num_samples", str(num_samples)])

        if relax:
            cmd.append("--relax")

        if max_antigen_size:
            cmd.extend(["--max_antigen_size", str(max_antigen_size)])

        # Set output directory
        out_dir = Path(work_dir) / "output"
        out_dir.mkdir(exist_ok=True)
        cmd.extend(["--output", str(out_dir)])

        print(f"Running: {' '.join(cmd)}")

        # Run IgGM
        run(cmd, check=True, cwd="/root/IgGM")

        # Collect all output files
        return [
            (out_file.relative_to(out_dir), out_file.read_bytes())
            for out_file in out_dir.rglob("*")
            if out_file.is_file()
        ]


@app.local_entrypoint()
def main(
    input_fasta: str,
    task: str = "design",
    antigen: str | None = None,
    epitope: str | None = None,
    fasta_origin: str | None = None,
    num_samples: int | None = None,
    relax: bool = False,
    max_antigen_size: int | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/iggm",
):
    """Run IgGM locally via Modal."""
    from datetime import datetime

    # Read input files
    input_fasta_str = Path(input_fasta).read_text()
    antigen_pdb_str = Path(antigen).read_text() if antigen else None
    fasta_origin_str = Path(fasta_origin).read_text() if fasta_origin else None

    # Parse epitope residues
    epitope_list = None
    if epitope:
        epitope_list = [int(x.strip()) for x in epitope.split(",")]

    # Run IgGM
    outputs = iggm.remote(
        input_fasta_str=input_fasta_str,
        task=task,
        antigen_pdb_str=antigen_pdb_str,
        epitope=epitope_list,
        fasta_origin_str=fasta_origin_str,
        num_samples=num_samples,
        relax=relax,
        max_antigen_size=max_antigen_size,
    )

    # Save outputs
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        output_path = Path(out_dir_full) / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(out_content)

    print(f"Results saved to: {out_dir_full}")
