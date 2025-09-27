"""Performs protein-ligand molecular dynamics simulations using OpenMM.

This module provides functionalities for:
- Preparing protein and ligand molecules for simulation.
- Running simulations with explicit or implicit solvent.
- Analyzing trajectories for RMSD and binding affinity.
- Handling PDB file fixing, ligand extraction, and decoy generation.
- Interfacing with external tools like Gnina and OpenBabel for scoring and file conversion.

Key dependencies include OpenMM, RDKit, MDTraj, PDBFixer, and OpenFF Toolkit.
"""

import json
import os
import re
import subprocess
import time
from shutil import copy2
from tempfile import NamedTemporaryFile
from typing import Union
from pathlib import Path
from warnings import warn

import mdtraj as md
import numpy as np
import pandas as pd
import requests

import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter

from openff.toolkit.topology import Molecule
from openmm import app, unit, LangevinIntegrator, MonteCarloBarostat, Platform
from openmm.app import DCDReporter, Modeller, PDBFile, Simulation, StateDataReporter

from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms, rdShapeHelpers

# for convenience so I can use as a script or a module
try:
    from .extract_ligands import extract_ligand
except ImportError:
    from extract_ligands import extract_ligand

GNINA, GNINA_BIN = "gnina", "./bin/gnina"
OBABEL, OBABEL_BIN = "obabel", "obabel"
GNINA_LINUX_URL = "https://github.com/gnina/gnina/releases/download/v1.0.3/gnina"
binaries = {GNINA: GNINA_BIN, OBABEL: OBABEL_BIN}

PDB_PH = 7.4
PDB_TEMPERATURE = 300 * unit.kelvin
FRICTION_COEFF = 1.0 / unit.picosecond
# https://docs.openforcefield.org/projects/toolkit/en/stable/faq.html
# With step size of 2 femtoseconds
STEP_SIZE = 0.002 * unit.picosecond
SOLVENT_PADDING = 10.0 * unit.angstroms
BAROSTAT_PRESSURE = 1.0 * unit.atmospheres
BAROSTAT_FREQUENCY = 25
# "hydrogen-involving bond constraints" is recommended
FORCEFIELD_KWARGS = {
    "constraints": app.HBonds,
    "rigidWater": True,
    "removeCMMotion": False,
    "hydrogenMass": 4 * unit.amu,
}
FORCEFIELD_PROTEIN = "amber/ff14SB.xml"
FORCEFIELD_IMPLICIT_SOLVENT = "implicit/obc2.xml"
FORCEFIELD_SOLVENT = "amber/tip3p_standard.xml"
FORCEFIELD_SMALL_MOLECULE = "gaff-2.11"

# when I run a simulation with a protein and an SDF file,
# openmm calls the ligand "UNK" in the combined output PDB file
OPENMM_DEFAULT_LIGAND_ID = "UNK"


def _download_binary_if_missing(binary_name: str):
    """Downloads a binary if it's not found locally.

    Args:
        binary_name (str): The name of the binary to check and download (e.g., "gnina").
    """

    def _download(path, url):
        """Helper function to download a file from a URL.

        Args:
            path (str): The local path to save the downloaded file.
            url (str): The URL to download the file from.

        Returns:
            bool: True if download was successful.
        """
        print(f"Downloading {binary_name} binary (300Mb for gnina)")
        with requests.get(url, timeout=600, stream=True) as r:
            r.raise_for_status()
            os.makedirs(Path(path).parent, exist_ok=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        os.chmod(path, 0o755)
        print(f"Downloaded {binary_name} binary\n")
        return True

    if binary_name == GNINA and not Path(GNINA_BIN).exists():
        _download(GNINA_BIN, GNINA_LINUX_URL)


_download_binary_if_missing(GNINA)


def get_platform():
    """Checks whether a GPU platform is available and sets precision to mixed if so.

    Returns:
        openmm.Platform: The fastest available OpenMM platform, configured for mixed precision if GPU.
    """

    platform = max(
        (Platform.getPlatform(i) for i in range(Platform.getNumPlatforms())),
        key=lambda p: p.getSpeed(),
    )

    if platform.getName() == "CUDA" or platform.getName() == "OpenCL":
        platform.setPropertyDefaultValue("Precision", "mixed")
        print(f"Set precision for platform {platform.getName()} to mixed\n")

    return platform


def prepare_protein(
    in_pdb_file: str,
    out_pdb_file: str,
    minimize_pdb: bool = False,
    mutations: list | None = None,
) -> bool:
    """
    Prepare a protein for simulation using PDBFixer and optionally minimize it using OpenMM.

    This function fixes common issues in PDB files (missing residues, atoms, non-standard residues)
    and prepares them for simulation. It adds missing hydrogens according to PDB_PH.
    If `minimize_pdb` is True, it also minimizes the protein's energy.

    Args:
        in_pdb_file (str): Path to the input PDB file.
        out_pdb_file (str): Path to the output PDB file where the prepared protein will be saved.
        minimize_pdb (bool, optional): If True, minimize the PDB structure using OpenMM.
            Useful for docked poses, less so for crystal structures. Defaults to False.
        mutations (list[str] | None, optional): A list of mutations to apply. Each mutation
            is a string like "ALA-10-GLY-A" (OldAA-ResID-NewAA-ChainID). Defaults to None.

    Returns:
        bool: True if the function executes successfully. (Note: actual error handling might involve exceptions).

    Warnings:
        Issues a warning if DMSO is found in the PDB file, suggesting manual removal.
    """

    fixer = PDBFixer(filename=in_pdb_file)
    for mutation in mutations or []:
        mutation = mutation.split("-")
        for chain in mutation[3]:  # e.g., AB
            fixer.applyMutations([f"{mutation[0]}-{mutation[1]}-{mutation[2]}"], chain)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()

    print(
        f"# Preparing protein:\n"
        f"- Missing residues: {fixer.missingResidues}\n"
        f"- Atoms: {fixer.missingAtoms}\n"
        f"- Terminals: {fixer.missingTerminals}\n"
        f"- Non-standard: {fixer.nonstandardResidues}\n"
    )

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(PDB_PH)
    fixer.removeHeterogens(keepWater=False)

    for res in fixer.topology.residues():
        if res.name == "DMS":
            warn("DMSO found in PDB file. Maybe remove?")

    # Note in the process of adding missing residues, some weird proteins are created
    # .e.g, the PDB file could list the first N residues, but have no co-ordinates
    # Then some combination of PDBFixer and openmm place these residues but in a straight line
    with open(out_pdb_file, "w", encoding="utf-8") as out:
        PDBFile.writeFile(fixer.topology, fixer.positions, file=out, keepIds=True)

    if minimize_pdb is True:
        system_generator = SystemGenerator(forcefields=[FORCEFIELD_PROTEIN])
        system = system_generator.create_system(fixer.topology)
        integrator = LangevinIntegrator(PDB_TEMPERATURE, FRICTION_COEFF, STEP_SIZE)
        simulation = Simulation(fixer.topology, system, integrator)
        simulation.context.setPositions(fixer.positions)
        simulation.minimizeEnergy()

        with open(
            f"{Path(out_pdb_file).with_suffix('')}_minimized_no_ligand.pdb",
            "w",
            encoding="utf-8",
        ) as out:
            PDBFile.writeFile(
                fixer.topology,
                simulation.context.getState(
                    getPositions=True, enforcePeriodicBox=False
                ).getPositions(),
                file=out,
                keepIds=True,
            )

    return True


def get_pdb_and_extract_ligand(
    pdb_id: str,
    ligand_id: str | None = None,
    ligand_chain: str | None = None,
    out_dir: str = ".",
    use_pdb_redo: bool = False,
    minimize_pdb: bool = False,
    mutations: list | None = None,
) -> dict:
    """
    Downloads a PDB file, prepares it for MD, and extracts a specified ligand.

    This function fetches a PDB file (either from RCSB or PDB-REDO), prepares the protein
    structure using `prepare_protein`, and then extracts a specified ligand into SDF and SMILES format.

    Args:
        pdb_id (str): The 4-letter PDB ID or path to a local PDB file.
        ligand_id (str | None, optional): The 3-letter ID of the ligand to extract. If None,
            only protein preparation is performed. Defaults to None.
        ligand_chain (str | None, optional): The chain ID of the ligand. Defaults to None.
        out_dir (str, optional): The directory to save output files. Defaults to '.'.
        use_pdb_redo (bool, optional): If True, download from PDB-REDO instead of RCSB.
            Defaults to False.
        minimize_pdb (bool, optional): If True, minimize the protein structure during preparation.
            Defaults to False.
        mutations (list[str] | None, optional): A list of mutations to apply during protein
            preparation (passed to `prepare_protein`). Defaults to None.

    Returns:
        dict: A dictionary containing paths to the generated files.
            Expected keys: "original_pdb" (str), "pdb" (str, path to prepared protein PDB),
            "sdf" (str, path to ligand SDF, if `ligand_id` is provided),
            "smi" (str, SMILES string of the ligand, if `ligand_id` is provided).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if pdb_id.endswith(".pdb"):
        pdb_file = str(Path(out_dir) / f"{Path(pdb_id).stem}.pdb")
        copy2(pdb_id, pdb_file)
        pdb_id = Path(pdb_id).stem
    elif use_pdb_redo:
        pdb_file = str(Path(out_dir) / f"{pdb_id}_pdbredo.pdb")
        subprocess.run(
            f"wget -O {pdb_file} https://pdb-redo.eu/db/{pdb_id}/{pdb_id}_final.pdb",
            check=True,
            shell=True,
            capture_output=True,
        )
    else:
        pdb_file = str(Path(out_dir) / f"{pdb_id}.pdb")
        subprocess.run(
            f"wget -O {pdb_file} https://files.rcsb.org/download/{pdb_id}.pdb",
            check=True,
            shell=True,
            capture_output=True,
        )

    # FIXFIX is it ok to prepare_protein BEFORE extracting the ligand?
    prepared_pdb_file = str(Path(out_dir) / f"{pdb_id}_fixed.pdb")
    prepare_protein(
        pdb_file, prepared_pdb_file, minimize_pdb=minimize_pdb, mutations=mutations
    )

    if ligand_id is None:  # then extract nothing, just prepare the protein
        return {"original_pdb": pdb_file, "pdb": prepared_pdb_file}

    # _out_pdb_file is just the protein selection (not prepared for openmm)
    out_sdf_file = str(Path(out_dir) / f"{pdb_id}_{ligand_id}.sdf")
    _, _, out_sdf_smiles = extract_ligand(
        pdb_file, ligand_id, ligand_chain, out_pdb_file=None, out_sdf_file=out_sdf_file
    )

    return {
        "original_pdb": pdb_file,
        "pdb": prepared_pdb_file,
        "sdf": out_sdf_file,
        "smi": out_sdf_smiles,
    }


def make_decoy(reference_rmol, decoy_smiles, num_conformers=100):
    """
    Generates a 3D conformer for a decoy molecule that best matches a reference molecule.

    The decoy molecule is generated from a SMILES string, and multiple conformers are
    generated. The conformer with the smallest Tanimoto shape distance to the
    reference molecule is selected and aligned.

    Args:
        reference_rmol (rdkit.Chem.rdchem.Mol): The reference RDKit molecule.
        decoy_smiles (str): SMILES string for the decoy molecule.
        num_conformers (int, optional): Number of conformers to generate for the decoy.
            Defaults to 100.

    Returns:
        tuple[rdkit.Chem.rdchem.Mol, openff.toolkit.topology.Molecule, openff.toolkit.topology.Molecule.conformers]:
            A tuple containing:
            - The RDKit decoy molecule with the best conformer selected.
            - The OpenFF decoy molecule.
            - The coordinates of the best conformer.
    """

    # Convert SMILES to 3D structure
    decoy_rmol = Chem.MolFromSmiles(decoy_smiles)
    decoy_rmol = Chem.AddHs(decoy_rmol)

    # Generate conformers
    # AllChem.EmbedMolecule(decoy_rmol) # pretty sure this is unnecessary given EmbedMultipleConfs
    AllChem.EmbedMultipleConfs(decoy_rmol, numConfs=num_conformers)

    # TODO replace below with _transform_conformer_to_match_reference

    # Align each conformer to the original ligand
    centroid_rmol = rdMolTransforms.ComputeCentroid(reference_rmol.GetConformer())

    min_shape_dist = None
    for n, conformer in enumerate(decoy_rmol.GetConformers()):
        centroid_decoy = rdMolTransforms.ComputeCentroid(conformer)
        translation = centroid_rmol - centroid_decoy
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = translation
        rdMolTransforms.TransformConformer(conformer, transformation_matrix)

        shape_dist = rdShapeHelpers.ShapeTanimotoDist(
            reference_rmol, decoy_rmol, confId1=0, confId2=n
        )
        if min_shape_dist is None or shape_dist < min_shape_dist:
            best_conformer_n = n
            min_shape_dist = shape_dist

    # convert to openff Molecule
    decoy_mol = Molecule(decoy_rmol)
    best_conformer = decoy_mol.conformers[best_conformer_n]

    print(
        "best conformer??",
        best_conformer,
        decoy_rmol.GetConformer(best_conformer_n).GetPositions(),
    )
    return decoy_rmol, decoy_mol, best_conformer


def prepare_ligand_for_MD(mol_filename: str, is_sanitize: bool = True):
    """
    Prepares a ligand from a file for Molecular Dynamics (MD) simulation.

    This function reads an SDF or MOL file into RDKit, adds hydrogens,
    assigns chiral tags, and then converts it to an OpenFF Molecule object.

    Args:
        mol_filename (str): Path to the SDF or MOL file to read.
        is_sanitize (bool, optional): Whether to sanitize the molecule in RDKit.
            Some SDFs may not be sanitizable, which can be an issue for OpenMM.
            Defaults to True.

    Returns:
        tuple[rdkit.Chem.rdchem.Mol, openff.toolkit.topology.Molecule]: A tuple containing:
            - The RDKit molecule object (with hydrogens).
            - The OpenFF Molecule object created from the RDKit molecule.
    """
    ligand_rmol = Chem.MolFromMolFile(mol_filename, sanitize=is_sanitize)
    ligand_rmol = Chem.AddHs(ligand_rmol, addCoords=True)

    # Ensure the chiral centers are all defined
    Chem.AssignAtomChiralTagsFromStructure(ligand_rmol)
    return ligand_rmol, Molecule(ligand_rmol)


def prepare_system_generator(ligand_mol=None, use_solvent=False):
    """
    Prepares an OpenMM SystemGenerator object for MD simulation.

    This function initializes a SystemGenerator with appropriate force fields
    for protein and optionally a small molecule ligand and solvent.

    Args:
        ligand_mol (openff.toolkit.topology.Molecule | None, optional): The OpenFF ligand
            molecule to include in the system. Defaults to None.
        use_solvent (bool, optional): If True, configure the system for explicit solvent
            (e.g., TIP3P water). If False, implicit solvent (OBC2) is used.
            Defaults to False.

    Returns:
        openmmforcefields.generators.SystemGenerator: The configured SystemGenerator object.
    """

    # FIXFIX why is `molecules` not passed for use_solvent=False in tdudgeon/simulateComplex.py?
    # is there any harm if it is?
    system_generator = SystemGenerator(
        forcefields=(
            [FORCEFIELD_PROTEIN, FORCEFIELD_SOLVENT]
            if use_solvent
            else [FORCEFIELD_PROTEIN, FORCEFIELD_IMPLICIT_SOLVENT]
        ),
        small_molecule_forcefield=FORCEFIELD_SMALL_MOLECULE,
        molecules=[ligand_mol] if ligand_mol else [],
        forcefield_kwargs=FORCEFIELD_KWARGS,
    )

    return system_generator


def analyze_traj(
    traj_dcd: str,
    topol_in: str,
    output_traj_analysis: str,
    ligand_chain_id: str = "1",
    backbone_chain_id: str = "0",
) -> pd.DataFrame:
    """Analyzes a trajectory for RMSD of backbone and ligand using MDTraj.

    Calculates Root Mean Square Deviation (RMSD) for the ligand and protein backbone
    over a trajectory. The trajectory is first superimposed on its initial frame.
    Results are saved to a tab-separated file.

    Args:
        traj_dcd (str): Path to the input DCD trajectory file.
        topol_in (str): Path to the input topology file (e.g., PDB).
        output_traj_analysis (str): Path to save the RMSD analysis (tab-separated CSV).
        ligand_chain_id (str, optional): Chain ID of the ligand in the topology.
            Defaults to "1".
        backbone_chain_id (str, optional): Chain ID of the protein backbone in the topology.
            Defaults to "0".

    Returns:
        pandas.DataFrame: A DataFrame with columns 'time' (ps), 'rmsd_bck' (nm), and 'rmsd_lig' (nm).

    Warnings:
        Assumes ligand is chainid 1 and protein is chainid 0 by default. This might
        not be universally true depending on PDB file conventions.
    """

    traj = md.load(traj_dcd, top=topol_in)

    # re-output the trajectory with the ligand superimposed on the first frame
    traj.superpose(traj, 0)
    traj.save_dcd(traj_dcd)

    lig_atoms = traj.topology.select(f"chainid {ligand_chain_id}")
    rmsds_lig = md.rmsd(
        traj, traj, frame=0, atom_indices=lig_atoms, parallel=True, precentered=False
    )

    # and backbond constrains it to only the backbone atoms, not sidechains
    bb_atoms = traj.topology.select(f"chainid {backbone_chain_id} and backbone")
    rmsds_bck = md.rmsd(
        traj, traj, frame=0, atom_indices=bb_atoms, parallel=True, precentered=False
    )

    print(
        f"Topology:\n"
        f"- {traj.topology} with n_frames={traj.n_frames}\n"
        f"- {len(lig_atoms)} ligand atoms\n"
        f"- {len(bb_atoms)} backbone atoms"
    )

    df_traj = (
        pd.DataFrame([traj.time, rmsds_bck, rmsds_lig])
        .T.map(lambda x: round(x, 8))
        .rename(columns={0: "time", 1: "rmsd_bck", 2: "rmsd_lig"})
    )

    df_traj.to_csv(output_traj_analysis, sep="\t", index=False)

    return df_traj


def get_affinity(
    pdb_in: str,
    ligand_id: str,
    convert_to_pdbqt: bool = False,
    scoring_tool: str = GNINA,
) -> float:
    """Calculates predicted binding affinity using Gnina or a similar tool.

    Extracts protein and ligand from a PDB file, then uses a scoring tool
    (typically Gnina) to predict binding affinity. Lower scores indicate stronger binding.
    Optionally converts the protein to PDBQT format using OpenBabel first.

    Args:
        pdb_in (str): Path to the input PDB file containing the protein-ligand complex.
        ligand_id (str): Residue name (3-letter code) of the ligand within the PDB file.
        convert_to_pdbqt (bool, optional): If True, convert the protein part to PDBQT
            format using OpenBabel before scoring. This might handle some Gnina issues.
            Defaults to False.
        scoring_tool (str, optional): The command-line tool to use for scoring.
            Defaults to GNINA (expected to be "./bin/gnina").

    Returns:
        float: The predicted binding affinity (e.g., in kcal/mol).
    """
    gnina_affinity_pattern = r"Affinity:\s*([\-\.\d+]+)"

    with (
        NamedTemporaryFile("w", suffix="_ligand.pdb", delete=False) as gnina_ligand_pdb,
        NamedTemporaryFile(
            "w", suffix="_protein.pdb", delete=False
        ) as gnina_protein_pdb,
        NamedTemporaryFile(
            "w", suffix="_protein.pdbqt", delete=False
        ) as gnina_protein_pdbqt,
    ):
        for line in open(pdb_in, encoding="utf-8"):
            if (
                line.startswith("HETATM")
                and line[17:20] == ligand_id
                or line.startswith("CONECT")
            ):
                gnina_ligand_pdb.write(line)
            elif not line.startswith("HETATM"):
                gnina_protein_pdb.write(line)
        gnina_ligand_pdb.flush()
        gnina_ligand_pdb.seek(0)
        gnina_protein_pdb.flush()
        gnina_protein_pdb.seek(0)

        # convert pdb to pdbqt too to get flexible side chains? In theory MD sorts this out
        if convert_to_pdbqt:
            cmd = (
                f"{binaries[OBABEL]} {gnina_protein_pdb.name} -O {gnina_protein_pdbqt.name} && "
                f"{binaries[scoring_tool]} --cpu {max(1, os.cpu_count() - 1)} --score_only "
                f"-r {gnina_protein_pdbqt.name} -l {gnina_ligand_pdb.name}"
            )
        else:
            cmd = (
                f"{binaries[scoring_tool]} --cpu {max(1, os.cpu_count() - 1)} --score_only "
                f"-r {gnina_protein_pdb.name} -l {gnina_ligand_pdb.name}"
            )

        print(f"- Calculating score: {cmd}")
        gnina_out = subprocess.run(
            cmd, check=True, shell=True, capture_output=True
        ).stdout.decode("ascii")

    affinity = float(re.findall(gnina_affinity_pattern, gnina_out)[0])

    return affinity


def extract_pdbs_from_dcd(complex_pdb: str, trajectory_dcd: str) -> dict:
    """Extracts individual PDB snapshots from a DCD trajectory file.

    Uses MDAnalysis to load a trajectory and its corresponding topology (PDB),
    then writes each frame as a separate PDB file.

    Args:
        complex_pdb (str): Path to the PDB file corresponding to the topology of the trajectory
            (e.g., the initial complex PDB from OpenMM).
        trajectory_dcd (str): Path to the DCD trajectory file.

    Returns:
        dict[float, str]: A dictionary where keys are simulation time points (in picoseconds)
            and values are the file paths to the extracted PDB snapshot for that frame.
    """
    universe = mda.Universe(complex_pdb, trajectory_dcd)

    traj_pdbs = {}
    for ts in universe.trajectory:
        time_ps = round(ts.time, 2)
        traj_pdbs[time_ps] = (
            f"{Path(complex_pdb).parent / Path(complex_pdb).stem}_f{time_ps}.pdb"
        )
        with PDBWriter(traj_pdbs[time_ps]) as out_pdb:
            out_pdb.write(universe.atoms)

    return traj_pdbs


def simulate(
    pdb_in: str,
    mol_in: str,
    output: str,
    num_steps: int,
    use_solvent: bool = False,
    decoy_smiles: Union[str | None] = None,
    minimize_only: bool = False,
    temperature: float = PDB_TEMPERATURE,
    equilibration_steps: int = 200,
    reporting_interval: Union[int, None] = None,
    scoring_tool: str = GNINA,
) -> dict:
    """Runs a molecular dynamics simulation of a protein-ligand complex using OpenMM.

    This comprehensive function sets up and runs an MD simulation. It can handle:
    - Simulation in implicit or explicit solvent.
    - Use of a provided ligand or generation of a decoy.
    - Minimization-only runs or full MD simulations with equilibration.
    - Calculation of binding affinities before and after minimization, and along the trajectory.
    - Trajectory analysis for RMSD.

    Args:
        pdb_in (str): Path to the input PDB file for the protein.
        mol_in (str | None): Path to the input MOL/SDF file for the ligand. Can be None if `decoy_smiles` is provided.
        output (str): Prefix for all output file names (e.g., "sim_output/run1").
        num_steps (int): Number of simulation steps to perform after equilibration.
        use_solvent (bool, optional): If True, run with explicit solvent (TIP3P).
            If False, use implicit solvent (OBC2). Defaults to False.
        decoy_smiles (str | None, optional): SMILES string for a decoy ligand. If provided,
            `mol_in` might be used as a reference for generating the decoy's initial conformer.
            Defaults to None.
        minimize_only (bool, optional): If True, only perform energy minimization and stop.
            No MD steps will be run. Defaults to False.
        temperature (float, optional): Simulation temperature in Kelvin.
            Defaults to `PDB_TEMPERATURE` (300 K).
        equilibration_steps (int, optional): Number of equilibration steps before the main simulation.
            Defaults to 200.
        reporting_interval (int | None, optional): Interval (in steps) for writing trajectory frames
            and state data. If None, a reasonable default is chosen based on `num_steps`.
            Defaults to None.
        scoring_tool (str, optional): Command-line tool for binding affinity calculation (e.g., Gnina).
            Defaults to `GNINA`.

    Returns:
        dict: A dictionary containing paths to the major output files, such as:
            "complex_pdb" (str): Path to the PDB of the initial protein-ligand complex.
            "minimized_pdb" (str): Path to the PDB of the minimized complex.
            "affinity_tsv" (str): Path to a TSV file with binding affinities.
            "args_json" (str): Path to a JSON file saving the input arguments.
            If `minimize_only` is False, it also includes:
            "traj_dcd" (str): Path to the DCD trajectory file.
            "state_tsv" (str): Path to a TSV file with simulation state data (energy, temp).
            "analysis_tsv" (str): Path to a TSV file with RMSD analysis.
    """

    os.makedirs(Path(output).parent, exist_ok=True)
    output_complex_pdb = f"{output}_complex.pdb"
    output_minimized_pdb = f"{output}_minimized.pdb"
    output_affinity_tsv = f"{output}_affinity.tsv"
    if minimize_only is not True:
        output_traj_dcd = f"{output}_traj.dcd"
        output_state_tsv = f"{output}_state.tsv"
        output_analysis_tsv = f"{output}_analysis.tsv"
    output_args_json = f"{output}_args.json"
    json.dump(locals(), open(output_args_json, "w", encoding="utf-8"), indent=2)

    out_affinity = open(output_affinity_tsv, "w", encoding="utf-8")
    out_affinity.write("time_ps\taffinity\n")

    if num_steps is None:
        num_steps = 1

    # A reasonable number based on the number of steps unless it's >1M steps
    max_frames_to_report = 100
    reporting_interval = min(10_000, num_steps // max_frames_to_report)

    print(f"Processing {pdb_in} and {mol_in} with {num_steps} steps")

    # -------------------------------------------------------
    # Set up system
    #

    platform = get_platform()

    if mol_in is not None:
        print(f"# Preparing ligand:\n- {mol_in}\n")
        ligand_rmol, ligand_mol = prepare_ligand_for_MD(mol_in)
        ligand_conformer = ligand_mol.conformers[0]
        assert len(ligand_mol.conformers) == len(ligand_rmol.GetConformers()) == 1, (
            "reference ligand should have one conformer"
        )
    elif decoy_smiles is not None:
        ligand_rmol, ligand_mol, ligand_conformer = make_decoy(
            ligand_rmol, decoy_smiles
        )
        print(f"# Using decoy:\n- {ligand_mol}\n- {ligand_conformer}\n")
    else:
        ligand_rmol, ligand_mol, ligand_conformer = None, None, None

    # Initialize a SystemGenerator using the GAFF for the ligand and tip3p for the water.
    # Chat-GPT: To use a larger time step, artificially increase the mass of the hydrogens.
    print("# Preparing system")
    system_generator = prepare_system_generator(ligand_mol, use_solvent)

    # Use Modeller to combine the protein and ligand into a complex
    print("# Reading protein")
    protein_pdb = PDBFile(pdb_in)

    print("# Preparing complex")
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    print(f"- System has {modeller.topology.getNumAtoms()} atoms after adding protein")

    # This next bit is black magic.
    # Modeller needs topology and positions. Lots of trial and error found that this is what works to get
    # these from an openforcefield Molecule object that was created from a RDKit molecule.
    # The topology part is described in the openforcefield API but the positions part grabs the first
    # (and only) conformer and passes it to Modeller. It works. Don't ask why!
    # modeller.topology.setPeriodicBoxVectors([Vec3(x=8.461, y=0.0, z=0.0),
    # Vec3(x=0.0, y=8.461, z=0.0), Vec3(x=0.0, y=0.0, z=8.461)])
    if ligand_mol is not None:
        modeller.add(
            ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm()
        )
        print(
            f"- System has {modeller.topology.getNumAtoms()} atoms after adding ligand"
        )

    if use_solvent:
        # We use the 'padding' option to define the periodic box. The PDB file does not contain any
        # unit cell information so we just create a box that has a 10A padding around the complex.
        modeller.addSolvent(
            system_generator.forcefield, model="tip3p", padding=SOLVENT_PADDING
        )
        print(
            f"- System has {modeller.topology.getNumAtoms()} atoms after adding solvent"
        )

    # Output the complex with topology
    with open(output_complex_pdb, "w", encoding="utf-8") as out:
        PDBFile.writeFile(modeller.topology, modeller.positions, out)

    if ligand_mol is not None:
        affinity = get_affinity(
            output_complex_pdb, OPENMM_DEFAULT_LIGAND_ID, scoring_tool=scoring_tool
        )
        out_affinity.write(f"complex\t{affinity:.4f}\n")

    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    integrator = LangevinIntegrator(temperature, FRICTION_COEFF, STEP_SIZE)

    # This line is present in the WithSolvent.py version of the script but unclear why
    if use_solvent:
        system.addForce(
            MonteCarloBarostat(BAROSTAT_PRESSURE, temperature, BAROSTAT_FREQUENCY)
        )

    print(
        f"- Using Periodic box: {system.usesPeriodicBoundaryConditions()}\n"
        f"- Default Periodic box: {system.getDefaultPeriodicBoxVectors()}\n"
    )

    # -------------------------------------------------------
    # Run simulation
    #
    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)

    print("# Minimising ...")
    simulation.minimizeEnergy()

    # Write out the minimized PDB.
    # 'enforcePeriodicBox=False' is important otherwise the different components can end up in
    # different periodic boxes resulting in really strange looking output.
    with open(output_minimized_pdb, "w", encoding="utf-8") as out:
        PDBFile.writeFile(
            modeller.topology,
            context.getState(
                getPositions=True, enforcePeriodicBox=False
            ).getPositions(),
            file=out,
            keepIds=True,
        )

    if ligand_mol is not None:
        affinity = get_affinity(
            output_minimized_pdb, OPENMM_DEFAULT_LIGAND_ID, scoring_tool=scoring_tool
        )
        out_affinity.write(f"min\t{affinity:.4f}\n")
        out_affinity.flush()

    if minimize_only:
        return {
            "complex_pdb": output_complex_pdb,
            "minimized_pdb": output_minimized_pdb,
            "affinity_tsv": output_affinity_tsv,
            "args_json": output_args_json,
        }

    print("## Equilibrating ...")
    context.setVelocitiesToTemperature(temperature)
    simulation.step(equilibration_steps)

    # Run the simulation.
    # The enforcePeriodicBox arg to the reporters is important.
    # It's a bit counter-intuitive that the value needs to be False, but this is needed to ensure
    # that all parts of the simulation end up in the same periodic box when being output.
    # optional: simulation.reporters.append(PDBReporter(output_traj_pdb, reporting_interval,
    #                                       enforcePeriodicBox=False))
    simulation.reporters.append(
        DCDReporter(output_traj_dcd, reporting_interval, enforcePeriodicBox=False)
    )
    simulation.reporters.append(
        StateDataReporter(
            output_state_tsv,
            reporting_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )

    print(f"# Starting simulation with {num_steps} steps ...")
    time_0 = time.time()
    simulation.step(num_steps)
    time_1 = time.time()
    print(f"- Simulation complete in {time_1 - time_0} seconds at {temperature}K\n")

    # -------------------------------------------------------
    # Calculate affinities during the simulation
    #
    print("# Calculating affinities along trajectory...")
    traj_pdbs = extract_pdbs_from_dcd(output_complex_pdb, output_traj_dcd)
    if ligand_mol is not None:
        traj_affinities = {
            time_ps: get_affinity(
                traj_pdb, OPENMM_DEFAULT_LIGAND_ID, scoring_tool=scoring_tool
            )
            for time_ps, traj_pdb in traj_pdbs.items()
        }
        for time_ps, affinity in traj_affinities.items():
            out_affinity.write(f"{time_ps:.2f}\t{affinity:.4f}\n")

    print("# Running trajectory analysis...")
    _ = analyze_traj(output_traj_dcd, output_complex_pdb, output_analysis_tsv)

    # Fix the state data file: from csv to tsv
    (
        pd.read_csv(output_state_tsv, sep=",")
        .map(lambda x: round(x, 4) if isinstance(x, float) else x)
        .to_csv(output_state_tsv, sep="\t", index=False)
    )

    return {
        "complex_pdb": output_complex_pdb,
        "traj_dcd": output_traj_dcd,
        "minimized_pdb": output_minimized_pdb,
        "affinity_tsv": output_affinity_tsv,
        "args_json": output_args_json,
        "state_tsv": output_state_tsv,
        "analysis_tsv": output_analysis_tsv,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenMM protein-ligand simulation")
    parser.add_argument("pdb_in", type=str, help="Input PDB file path")
    parser.add_argument("mol_in", type=str, help="Input mol file path")
    parser.add_argument(
        "output", type=str, help="Output file name root, including path"
    )
    parser.add_argument("num_steps", type=int, help="Number of simulation steps")
    parser.add_argument("--use_solvent", action="store_true", help="Use solvent?")
    parser.add_argument(
        "--decoy_smiles",
        type=str,
        default=None,
        help="Use a decoy aligned to mol_in for simulation",
    )
    parser.add_argument(
        "--minimize_only", action="store_true", help="Only perform minimization"
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0, help="Temperature in Kelvin"
    )
    parser.add_argument(
        "--equilibration_steps", type=int, default=200, help="Equilibration steps"
    )
    parser.add_argument(
        "--reporting_interval", type=int, default=None, help="Reporting interval"
    )
    args = parser.parse_args()

    simulate(
        args.pdb_in,
        args.mol_in,
        args.output,
        args.num_steps,
        use_solvent=args.use_solvent,
        decoy_smiles=args.decoy_smiles,
        minimize_only=args.minimize_only,
        temperature=args.temperature,
        equilibration_steps=args.equilibration_steps,
        reporting_interval=args.reporting_interval,
    )
