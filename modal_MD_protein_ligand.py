from datetime import datetime
from pathlib import Path
from typing import Union

from modal import Image, Mount, Stub

FORCE_BUILD = False
LOCAL_IN = "./in/md_protein_ligand"
LOCAL_OUT = "./out/md_protein_ligand"
REMOTE_IN = "/in"
GPU = "T4"

stub = Stub()

image = (Image
         .micromamba(python_version="3.10")
         .apt_install("git", "wget")
         .micromamba_install(["openmm=8.1.1", "openmmforcefields=0.12.0", "openmm-ml=1.1", "pdbfixer=1.9",
                              "rdkit=2023.03.1", "mdtraj=1.9.9", "plotly=4.9.0",
                              "openff-toolkit=0.15.2", "python-kaleido=0.2.1", "mdanalysis=2.5.0",
                              "prody=2.4.0", "pymol-open-source==2.5.0", "pypdb=2.3"],
                             channels=["omnia", "plotly", "conda-forge"])
         # maybe replace with local? i am not sure
         #.pip_install("git+https://github.com/hgbrian/MD_protein_ligand", force_build=FORCE_BUILD)
         #.run_commands("python -c 'from MD_protein_ligand import simulate'")
        )

with image.imports():
    from MD_protein_ligand import simulate


@stub.function(image=image, gpu=GPU, timeout=60*15,
               mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)])
def simulate_md_ligand(pdb_id:str, ligand_id:str, ligand_chain:str,
                       use_pdb_redo:bool, num_steps:Union[int,None], minimize_only:bool,
                       use_solvent:bool, decoy_smiles:Union[str,None], mutation:Union[str,None],
                       temperature:int, equilibration_steps:int, out_dir_root:str):
    """MD simulation of protein + ligand"""

    out_dir = str(Path(out_dir_root) / (f"{pdb_id}_{ligand_id}" if ligand_id else f"{pdb_id}"))
    out_stem = str(Path(out_dir) / (f"{pdb_id}_{ligand_id}" if ligand_id else f"{pdb_id}"))

    #
    # Mutate a residue and relax again.
    # Mutate prepared_files["pdb"] to ensure consistency
    # e.g., LEU-117-VAL-AB, following PDBFixer format (but adding chains)
    #
    if mutation is not None:
        mutate_from, mutate_resn, mutate_to, mutate_chains = mutation.split("-")
        out_stem = (f"{out_stem}_{mutate_from}_{mutate_resn}_{mutate_to}_{mutate_chains}")

    prepared_files = simulate.get_pdb_and_extract_ligand(pdb_id, ligand_id, ligand_chain,
                                                         out_dir=out_dir,
                                                         use_pdb_redo=use_pdb_redo,
                                                         mutation=(mutate_from, mutate_resn, mutate_to, mutate_chains)
                                                                   if mutation else None)

    sim_files = simulate.simulate(prepared_files["pdb"], prepared_files.get("sdf", None), out_stem, num_steps,
                                  minimize_only=minimize_only, use_solvent=use_solvent, decoy_smiles=decoy_smiles,
                                  temperature=temperature, equilibration_steps=equilibration_steps)

    # read in the output files
    return {out_name: (fname, open(fname, 'rb').read() if Path(fname).exists() else None)
            for out_name, fname in (prepared_files | sim_files).items()}


@stub.local_entrypoint()
def main(pdb_id:str, ligand_id:str=None, ligand_chain:str=None,
         use_pdb_redo:bool=False, num_steps:int=None,
         use_solvent:bool=False, decoy_smiles:str=None, mutation:str=None,
         temperature:int=300, equilibration_steps:int=200, out_dir_date=True, 
         out_dir_root:str="."):
    """
    MD simulation of protein + ligand.

    :param pdb_id: String representing the PDB ID of the protein.
    :param ligand_id: String representing the ligand ID (ID in PDB file).
    :param ligand_chain: String representing the ligand chain.
        You have to explicitly specify the chain, because otherwise >1 ligands can be merged!
    :param use_pdb_redo: Boolean, whether to use PDB redo. Default is False.
    :param num_steps: Integer representing the number of steps. Default is None.
    :param use_solvent: Boolean, whether to use solvent in the simulation. Default is True.
    :param decoy_smiles: String representing the decoy SMILES notation. Default is None.
    :param mutation: String representing the mutation in the format "ALA-117-VAL-AB".
        Follows PDBFixer notation, BUT you must include the chains to mutate too. Default is None.
    :param temperature: Integer representing the temperature in the simulation. Default is 300.
    :param equilibration_steps: Integer representing the number of equilibration steps. Default is 200.
    :param out_dir_root: String representing the root directory for output. Default is "out".
    """

    minimize_only = True if not num_steps else False
    if ligand_id is not None:
        assert ligand_chain is not None

    outputs = simulate_md_ligand.remote(pdb_id, ligand_id, ligand_chain,
                                        use_pdb_redo, num_steps, minimize_only,
                                        use_solvent, decoy_smiles, mutation,
                                        temperature, equilibration_steps, out_dir_root)

    out_dir_date = datetime.now().strftime("%Y%m%d%H%M")[2:] if out_dir_date else "."

    for (out_file, out_content) in outputs.values():
        if out_content:
            (Path(LOCAL_OUT) / out_dir_date / out_file).parent.mkdir(parents=True, exist_ok=True)
            with open(Path(LOCAL_OUT) / out_dir_date / out_file, 'wb') as out:
                out.write(out_content)

    print("Outputs:", {k:v[0] for k, v in outputs.items()})
