"""
MD_protein_ligand

Errors:
    OpenMMException: NonbondedForce: The cutoff distance cannot be greater than half the periodic box size. For more information, see
Solution:
    - Check to see if the PeriodicBox is too small (e.g., 0.1)
        - If it is, then it might be due to a CRYST line in the pdb file.
          You can delete all lines that are not ATOM, TER, END.

"""

import os
from datetime import datetime
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", None)  # no GPU or T4 for testing
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

GNINA_BIN_URL = "https://github.com/gnina/gnina/releases/download/v1.3/gnina"


def download_gnina_binary():
    import requests

    with requests.get(GNINA_BIN_URL, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open("/bin/gnina", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.chmod("/bin/gnina", 0o755)


image = (
    Image.micromamba(python_version="3.10")
    .apt_install("git", "wget")
    .micromamba_install(
        [
            "openmm=8.1.1",
            "openmmforcefields=0.12.0",
            "openmm-ml=1.1",
            "nnpops=0.6",
            "openff-toolkit=0.15.2",
            "pdbfixer=1.9",
            "rdkit=2023.03.1",
            "mdtraj=1.9.9",
            "plotly=4.9.0",
            "python-kaleido=0.2.1",
            "mdanalysis=2.5.0",
            "prody=2.4.0",
            "pymol-open-source==2.5.0",
            "pypdb=2.3",
        ],
        channels=["omnia", "plotly", "conda-forge"],
    )
    .run_function(download_gnina_binary)
    .add_local_python_source("MD_protein_ligand", copy=True)
    # maybe replace with local? i am not sure
    # .pip_install("git+https://github.com/hgbrian/MD_protein_ligand")
    # .run_commands("python -c 'from MD_protein_ligand import simulate'")
)

app = App("md_protein_ligand", image=image)

with image.imports():
    from MD_protein_ligand import simulate


@app.function(
    gpu=GPU,
    timeout=60 * TIMEOUT,
)
def simulate_md_ligand(
    pdb_id: str,
    pdb_contents: bytes | None,
    ligand_id: str,
    ligand_chain: str,
    use_pdb_redo: bool,
    num_steps: int | None,
    minimize_only: bool,
    use_solvent: bool,
    decoy_smiles: str | None,
    mutations: list | None,
    temperature: int,
    equilibration_steps: int,
    out_dir_root: str,
):
    """MD simulation of protein + ligand"""

    # Handle local PDB file vs PDB ID
    if pdb_contents is not None:
        # Create temporary file with PDB contents using a unique name
        import tempfile

        temp_fd, pdb_file_remote = tempfile.mkstemp(
            suffix=".pdb", prefix="pdb_", dir="/tmp"
        )
        try:
            with os.fdopen(temp_fd, "wb") as f:
                f.write(pdb_contents)
        except:
            os.close(temp_fd)  # Clean up if write fails
            raise
        # Extract the base name without .pdb extension for further processing
        if pdb_id.endswith(".pdb"):
            pdb_id = Path(pdb_id).stem
    else:
        pdb_file_remote = None

    out_id = f"{pdb_id}_{ligand_id}" if ligand_id else f"{pdb_id}"

    # Make the output directory unique to avoid conflicts when running multiple mutations
    import uuid

    unique_id = str(uuid.uuid4())[:8]
    out_dir = str(Path(out_dir_root) / f"{out_id}_{unique_id}")
    out_stem = str(Path(out_dir) / out_id)

    #
    # Mutate a residue and relax again.
    # Mutate prepared_files["pdb"] to ensure consistency
    # e.g., LEU-117-VAL-AB, following PDBFixer format (but adding chains)
    #
    for mutation in mutations or []:
        mut_from, mut_resn, mut_to, mut_chains = mutation.split("-")
        out_stem += f"_{mut_from}_{mut_resn}_{mut_to}_{mut_chains}"

    prepared_files = simulate.get_pdb_and_extract_ligand(
        pdb_file_remote if pdb_contents is not None else pdb_id,
        ligand_id,
        ligand_chain,
        out_dir=out_dir,
        use_pdb_redo=use_pdb_redo,
        mutations=mutations,
    )

    sim_files = simulate.simulate(
        prepared_files["pdb"],
        prepared_files.get("sdf", None),
        out_stem,
        num_steps,
        minimize_only=minimize_only,
        use_solvent=use_solvent,
        decoy_smiles=decoy_smiles,
        temperature=temperature,
        equilibration_steps=equilibration_steps,
    )

    # read in the output files
    return {
        out_name: (fname, open(fname, "rb").read() if Path(fname).exists() else None)
        for out_name, fname in (prepared_files | sim_files).items()
    }


@app.local_entrypoint()
def main(
    pdb_id: str,
    ligand_id: str | None = None,
    ligand_chain: str | None = None,
    use_pdb_redo: bool = False,
    num_steps: int | None = None,
    use_solvent: bool = False,
    decoy_smiles: str | None = None,
    mutations: str | None = None,
    temperature: int = 300,
    equilibration_steps: int = 200,
    run_name: str | None = None,
    out_dir: str = "./out/md_protein_ligand",
):
    """
    MD simulation of protein + ligand.

    :param pdb_id: String representing the PDB ID of the protein, or path to local PDB file.
    :param ligand_id: String representing the ligand ID (ID in PDB file).
    :param ligand_chain: String representing the ligand chain.
        You have to explicitly specify the chain, because otherwise >1 ligands can be merged!
    :param use_pdb_redo: Boolean, whether to use PDB redo. Default is False.
    :param num_steps: Integer representing the number of steps. Default is None.
    :param use_solvent: Boolean, whether to use solvent in the simulation. Default is False.
    :param decoy_smiles: String representing the decoy SMILES notation. Default is None.
    :param mutation: String representing the mutation in the format "ALA-117-VAL-AB".
        Follows PDBFixer notation, BUT you must include the chains to mutate too. Default is None.
    :param temperature: Integer representing the temperature in the simulation. Default is 300.
    :param equilibration_steps: Integer representing the number of equilibration steps. Default is 200.
    :param out_dir_root: String representing the root directory for output. Default is ".".
    """

    minimize_only = True if not num_steps else False
    if ligand_id is not None:
        assert ligand_chain is not None, "specify a ligand_chain, e.g., A"

    # Read PDB file contents if it's a local file
    pdb_contents = None
    if pdb_id.endswith(".pdb") and Path(pdb_id).exists():
        with open(pdb_id, "rb") as f:
            pdb_contents = f.read()

    outputs = simulate_md_ligand.remote(
        pdb_id,
        pdb_contents,
        ligand_id,
        ligand_chain,
        use_pdb_redo,
        num_steps,
        minimize_only,
        use_solvent,
        decoy_smiles,
        mutations.split(",") if mutations else None,
        temperature,
        equilibration_steps,
        ".",  # FIXFIX should be out_dir?
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs.values():
        (Path(out_dir_full) / out_file).parent.mkdir(parents=True, exist_ok=True)
        (Path(out_dir_full) / out_file).write_bytes(out_content or b"")

    print("Outputs:", {k: v[0] for k, v in outputs.items()})
