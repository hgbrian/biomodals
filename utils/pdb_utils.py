from pathlib import Path
from subprocess import run

ALPHAFOLD_VERSION = "v4"

def get_pdb(pdb_code_or_file, biological_assembly=False, pdb_redo=False, out_dir="."):
    """Get a PDB file by code or by filename.
    Downloads to the current directory."""

    if biological_assembly is True and pdb_redo is True:
        raise AssertionError("Biological assembly is not available for pdb-redo files")

    if Path(pdb_code_or_file).is_file():
        out_path = Path(pdb_code_or_file).resolve()
    elif len(pdb_code_or_file) == 4:
        if pdb_redo:
            pdb_name = f"{pdb_code_or_file}_final.pdb"
            out_path = Path(out_dir) / Path(pdb_name)
            run(f"wget -qnc https://pdb-redo.eu/db/{pdb_code_or_file}/{pdb_name} -O {out_path}", shell=True, check=True)
        else:
            pdb_name = f"{pdb_code_or_file}.pdb{'1' if biological_assembly else ''}"
            out_path = Path(out_dir) / Path(pdb_name)
            run(f"wget -qnc https://files.rcsb.org/view/{pdb_name} -O {out_path}", shell=True, check=True)
    else:
        pdb_name = f"AF-{pdb_code_or_file}-F1-model_{ALPHAFOLD_VERSION}.pdb"
        out_path = Path(out_dir) / Path(pdb_name)
        run(f"wget -qnc https://alphafold.ebi.ac.uk/files/{pdb_name} -O {out_path}", shell=True, check=True)

    if not out_path.is_file():
        raise FileNotFoundError(f"{pdb_code_or_file} PDB file {out_path} does not exist")

    if out_path.stat().st_size < 1000:
        raise AssertionError(f"{pdb_code_or_file} PDB file {out_path} is too small, something went wrong, e.g., "
                             "pdb-redo will refuse poor quality pdbs")

    return str(out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Get a PDB file.')
    parser.add_argument('pdb_code', type=str, help='PDB code')
    parser.add_argument('--biological_assembly', type=str, default=False, help='get biological assembly version')
    parser.add_argument('--pdb_redo', type=str, default=True, help='use pdb redo instead of RCSB PDB')
    parser.add_argument('--out_dir', type=str, default='.', help='out dir')
    args = parser.parse_args()
    pdb_path = get_pdb(args.pdb_code,
                       biological_assembly=args.biological_assembly,
                       pdb_redo=args.pdb_redo,
                       out_dir=args.out_dir)
    print(f"Downloaded {pdb_path}")