"""AFDesign outputs the protein (chain A) and ligand (chain B) in one pdb.
(It also outputs >1 model.)
HAC-Net takes pdb and mol2 as input.
"""

from prody import parsePDB, writePDB
import subprocess
import sys

def extract_chain(pdb_file, select, output_file, model=1):
    """
    Extracts a specific chain from a specific model of a PDB file and saves it to a new file.
    :param pdb_file: Path to the PDB file.
    :param chain_id: Chain identifier to extract (e.g., 'B').
    :param output_file: Path to save the extracted chain.
    :param model: Model number to extract from (default is 1).
    """
    pdb = parsePDB(pdb_file, model=model)
    chain = pdb.select(select)
    writePDB(output_file, chain)

def convert_to_mol2(input_file, output_file):
    """
    Converts a PDB file to MOL2 format using Open Babel.
    :param input_file: Path to the input PDB file.
    :param output_file: Path to save the MOL2 file.
    """
    subprocess.run(["obabel", "-ipdb", input_file, "-omol2", "-O", output_file])

def convert_to_pdbqt(input_file, output_file):
    """
    Converts a PDB file to MOL2 format using Open Babel.
    :param input_file: Path to the input PDB file.
    :param output_file: Path to save the MOL2 file.
    """
    subprocess.run(["obabel", "-ipdb", input_file, "-opdbqt", "-O", output_file])

def main(pdb_file, chain_id='B', model=1):
    """
    Extracts a specified chain from a specified model of a PDB file and converts it to MOL2 format.
    :param pdb_file: Path to the PDB file.
    :param chain_id: Chain to extract and convert.
    :param model: Model number to extract from.
    """

    receptor_pdb = f'{pdb_file}_chain_not{chain_id}_model_{model}.pdb'
    ligand_pdb = f'{pdb_file}_chain_{chain_id}_model_{model}.pdb'
    extract_chain(pdb_file, f"not chain {chain_id}", receptor_pdb, model=model)
    extract_chain(pdb_file, f"chain {chain_id}", ligand_pdb, model=model)

    ligand_mol2 = f'{pdb_file}_chain_{chain_id}_model_{model}.mol2'
    ligand_pdbqt = f'{pdb_file}_chain_{chain_id}_model_{model}.pdbqt'
    convert_to_mol2(ligand_pdb, ligand_mol2)
    convert_to_pdbqt(ligand_pdb, ligand_pdbqt)

    # convert to pdbqt?
    print(f"Converted {pdb_file} (Chain {chain_id}, Model {model}) to {ligand_mol2}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pdb_file = sys.argv[1]
        chain_id = sys.argv[2] if len(sys.argv) > 2 else 'B'
        model = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        print(f"Extracting chain {chain_id} from model {model} of {pdb_file}")
        main(pdb_file, chain_id, model)
    else:
        print(f"Usage: {sys.argv[0]} pdb_file [chain_id] [model]", file=sys.stderr)
