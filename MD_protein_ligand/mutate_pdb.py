"""
Mutate a residue in a pdb file.

Requires pymol, which can be difficult to install:
!mamba install -c conda-forge pymol-open-source
"""

from pymol import cmd

AA_MAP_3_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_MAP_1_3 = {v:k for k, v in AA_MAP_3_1.items()}


def mutate_pdb(pdb_file:str, chains:str, res_num:int, aa:str, check_original_aa:str|None=None, rotamer:str|None=None) -> str:
    """
    This function mutates a specific residue in a pdb file and saves the modified structure.

    Parameters:
    pdb_file (str): The file path to the input pdb file.
    chains (str): The chains in the pdb file where the mutation should occur.
    res_num (int): The residue number where the mutation should occur.
    aa (str): The new amino acid (in three-letter format) to which the residue should be mutated.

    Returns:
    out_file (str): the mutated pdb file following the format: {pdb_file stem}_{chain}_{residue number}{mutated aa}.pdb
    """
    # important!! otherwise pymol holds onto state
    cmd.reinitialize()

    if len(aa) == 3:
        aa_3 = aa.upper()
        aa_1 = AA_MAP_3_1[aa_3]
    elif len(aa) == 1:
        aa_1 = aa.upper()
        aa_3 = AA_MAP_1_3[aa_1]
    else:
        raise ValueError(f"Invalid amino acid code {aa}")

    if check_original_aa is not None:
        if len(check_original_aa) == 3:
            check_original_aa = AA_MAP_3_1[check_original_aa.upper()]
        elif len(check_original_aa) == 1:
            check_original_aa = check_original_aa.upper()
        else:
            raise ValueError(f"Invalid amino acid code {check_original_aa}")

    cmd.load(pdb_file)
    cmd.wizard("mutagenesis")
    cmd.do("refresh_wizard")

    cmd.get_wizard().set_mode(aa_3)

    # if there are multiple chains then there may be more than one original_aa
    original_aas = set()
    for chain in chains:
        cmd.get_wizard().do_select(f"{chain}/{res_num}/")

        original_aa = []
        cmd.iterate(f"{chain}/{res_num}/", "original_aa.append(resn)", space={'original_aa': original_aa})
        original_aa = AA_MAP_3_1[original_aa[0].upper()]
        original_aas.add(original_aa)

        if check_original_aa is not None:
            assert check_original_aa == original_aa, f"Original amino acid is {original_aa}, not {check_original_aa}"

        # Optionally, select the rotamer. Expert use only!
        if rotamer is not None:
            cmd.frame(rotamer)

        cmd.get_wizard().apply()


    out_file = f"{pdb_file[:-4]}_{chains}_{''.join(original_aas)}{res_num}{aa_1}.pdb"
    cmd.save(out_file)

    return out_file


if __name__ == "__main__":
    print("test mutate_pdb")
    mutate_pdb("4O75.pdb", "A", 65, "ILE", check_original_aa="TYR")
