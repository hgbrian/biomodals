# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Designs protein binders, including cyclic peptides, using AFDesign on Modal.

Adapting the AFDesign colab for modal
https://colab.research.google.com/drive/1LHEbFMxMTGblSFmv83JBgH7I4TJt8E6M

- makes cyclic peptides by default
- 120 soft iters, 32 hard iters is recommended

Notes from the original colab:

# AfDesign - peptide binder design
For a given protein target and protein binder length, generate/hallucinate a protein binder
sequence AlphaFold thinks will bind to the target structure.
To do this, we maximize number of contacts at the interface and maximize pLDDT of the binder.

**WARNING**
1.   This notebook is in active development and was designed for demonstration purposes only.
2.   Using AfDesign as the only "loss" function for design might be a bad idea, you may find
     adversarial sequences (aka. sequences that trick AlphaFold).

Example:
```
modal run modal_afdesign.py --target-chain C --pdb in/afdesign/1igy.pdb
```
"""

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from subprocess import run
from tempfile import NamedTemporaryFile

from modal import Image, App

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 120))
DATA_DIR = "/"

warnings.simplefilter(action="ignore", category=FutureWarning)

image = (
    Image.micromamba()
    .apt_install("git", "wget", "aria2", "ffmpeg")
    .pip_install(
        "pdb-tools==2.4.8", "ffmpeg-python==0.2.0", "plotly==5.18.0", "kaleido==0.2.1"
    )
    .pip_install(
        "git+https://github.com/sokrypton/ColabDesign.git@v1.1.2", "jax[cuda12_pip]"
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign;"
        "mkdir /params"  # not sure which
    )
    .run_commands(
        "aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar;"
        "tar -xf alphafold_params_2022-12-06.tar -C /params"
    )
    .pip_install("matplotlib==3.8.1")
)

with image.imports():
    import numpy as np

    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.NeighborSearch import NeighborSearch
    from scipy.special import softmax


app = App("afdesign", image=image)


# ------------------------------------------------------------------------------
# BN added this function
#
def add_cyclic_offset(self):
    """Adds cyclic offset to connect N and C termini head to tail.

    This function modifies the model's internal offset matrix to enforce
    cyclization in peptide design.

    Args:
        self (colabdesign.af.model.af_model): The AFDesign model instance.

    Returns:
        None
    """

    def _cyclic_offset(L):
        """Calculates the cyclic offset matrix for a given length.

        Args:
            L (int): The length of the sequence or segment.

        Returns:
            np.ndarray: The cyclic offset matrix.
        """
        i = np.arange(L)
        ij = np.stack([i, i + L], -1)
        offset = i[:, None] - i[None, :]
        c_offset = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
        return np.sign(offset) * c_offset

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])

    if self.protocol == "binder":
        c_offset = _cyclic_offset(self._binder_len)
        offset[self._target_len :, self._target_len :] = c_offset

    if self.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in self._lengths:
            offset[Ln : Ln + L, Ln : Ln + L] = _cyclic_offset(L)
            Ln += L

    self._inputs["offset"] = offset


# ------------------------------------------------------------------------------
# BN added this function
#
class ResidueRangeSelect(Select):
    """Bio.PDB.Select class to accept residues within a specific range and chain."""
    def __init__(self, chain_ids, start, end):
        """Initializes the ResidueRangeSelect class.

        Args:
            chain_ids (list[str]): List of chain IDs to accept.
            start (int): Starting residue number to accept.
            end (int): Ending residue number to accept.
        """
        self.chain_ids = chain_ids
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        """Accepts a residue if it's within the specified range and chain.

        Args:
            residue (Bio.PDB.Residue.Residue): The residue to check.

        Returns:
            bool: True if the residue is accepted, False otherwise.
        """
        within_range = self.start <= residue.get_id()[1] <= self.end
        correct_chain = residue.parent.id in self.chain_ids
        return within_range and correct_chain


def extract_residues_from_pdb(pdb_file, chain_ids, start_residue, end_residue):
    """Extracts a specific range of residues from specified chains in a PDB file.

    Args:
        pdb_file (str): Path to the input PDB file.
        chain_ids (list[str]): List of chain IDs from which to extract residues.
        start_residue (int): Starting residue number.
        end_residue (int): Ending residue number.

    Returns:
        str: Path to a temporary PDB file containing the extracted residues.
    """
    # create a PDBParser object
    parser = PDBParser()

    # read the structure from a PDB file
    structure = parser.get_structure("my_protein", pdb_file)

    # create a PDBIO object
    io = PDBIO()

    # set the structure to be written
    io.set_structure(structure)

    # create a temporary file for output
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")

    # save the structure to the new PDB file, including only the specified range of residues
    io.save(temp_file.name, ResidueRangeSelect(chain_ids, start_residue, end_residue))

    return temp_file.name


# ------------------------------------------------------------------------------
# BN added this merging tool
#
def join_chains(pdb_file, target_chain, merge_chains):
    """Uses pdb-tools to combine specified chains in a PDB file into a single chain.

    Note: This might be unnecessary depending on AFDesign's capabilities.

    Args:
        pdb_file (str): Path to the input PDB file.
        target_chain (str): The new chain ID for the merged chain.
        merge_chains (list[str]): List of chain IDs to merge.

    Returns:
        str: Path to a temporary PDB file with merged chains.
    """
    with NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
        subprocess.run(
            f"pdb_selchain -{','.join(merge_chains)} {pdb_file} | "
            f"pdb_chain -{target_chain} | pdb_reres -1 > {tf.name}",
            shell=True,
            check=True,
        )
        return tf.name


# ------------------------------------------------------------------------------
# BN added this function
#
def get_nearby_residues(pdb_file, ligand_id, distance=8.0):
    """Reports the protein residues within a specified distance of a ligand.

    Args:
        pdb_file (str): Path to the PDB file.
        ligand_id (str): Residue name of the ligand (e.g., "LG1").
        distance (float, optional): Distance threshold in Angstroms. Defaults to 8.0.

    Returns:
        set[Bio.PDB.Residue.Residue]: A set of Biopython Residue objects that are near the ligand.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    # Get all atoms in the protein
    protein_atoms = [
        atom
        for atom in structure.get_atoms()
        if atom.parent.get_resname() != ligand_id and is_aa(atom.parent)
    ]

    # Get all atoms in the ligand
    ligand_atoms = [
        atom for atom in structure.get_atoms() if atom.parent.get_resname() == ligand_id
    ]

    # Create a NeighborSearch object
    ns = NeighborSearch(protein_atoms)

    # Find all protein atoms within `distance` of any ligand atom
    nearby_atoms = []
    for ligand_atom in ligand_atoms:
        nearby_atoms.extend(ns.search(ligand_atom.coord, distance))

    # Get the residues corresponding to these atoms
    nearby_residues = {atom.parent for atom in nearby_atoms}

    return nearby_residues


def get_pdb(pdb_code_or_file, biological_assembly=False, pdb_redo=False, out_dir="."):
    """Fetches a PDB file by code or uses a local filename, downloading if necessary.

    Downloads to `out_dir` (defaults to current directory). Can fetch from RCSB PDB,
    AlphaFold DB, or PDB-REDO.

    Args:
        pdb_code_or_file (str): PDB code (e.g., "1XYZ"), UniProt code (e.g., "P00760" for AFDB),
                                or path to a local PDB file.
        biological_assembly (bool, optional): If True, attempts to fetch the first biological
                                             assembly (e.g., "1XYZ.pdb1"). Defaults to False.
        pdb_redo (bool, optional): If True, attempts to fetch the PDB-REDO version if available.
                                   Defaults to False.
        out_dir (str, optional): Directory to download/output the PDB file. Defaults to ".".

    Returns:
        str: Path to the fetched or validated local PDB file.

    Raises:
        AssertionError: If `biological_assembly` and `pdb_redo` are both True, or if the
                        downloaded PDB file is too small (likely indicating an issue).
        FileNotFoundError: If the PDB file does not exist after attempting to fetch it.
    """
    ALPHAFOLD_VERSION = "v4"

    if biological_assembly is True and pdb_redo is True:
        raise AssertionError("Biological assembly is not available for pdb-redo files")

    if Path(pdb_code_or_file).is_file():
        out_path = Path(pdb_code_or_file).resolve()
    elif len(pdb_code_or_file) == 4:
        if pdb_redo:
            pdb_name = f"{pdb_code_or_file}_final.pdb"
            out_path = Path(out_dir) / Path(pdb_name)
            try:
                run(
                f"wget -qnc https://pdb-redo.eu/db/{pdb_code_or_file}/{pdb_name} -O {out_path}",
                shell=True,
                check=True,
                )
            except subprocess.CalledProcessError as e:
                print("Failed to find pdb-redo version. Using RSCB pdb.")
                pdb_redo = False
            else:
                return out_path

        if pdb_redo is False:
            pdb_name = f"{pdb_code_or_file}.pdb{'1' if biological_assembly else ''}"
            out_path = Path(out_dir) / Path(pdb_name)
            run(
                f"wget -qnc https://files.rcsb.org/view/{pdb_name} -O {out_path}",
                shell=True,
                check=True,
            )
    else:
        pdb_name = f"AF-{pdb_code_or_file}-F1-model_{ALPHAFOLD_VERSION}.pdb"
        out_path = Path(out_dir) / Path(pdb_name)
        run(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/{pdb_name} -O {out_path}",
            shell=True,
            check=True,
        )

    if not out_path.is_file():
        raise FileNotFoundError(
            f"{pdb_code_or_file} PDB file {out_path} does not exist"
        )

    if out_path.stat().st_size < 1000:
        raise AssertionError(
            f"{pdb_code_or_file} PDB file {out_path} is too small, something went wrong, e.g., "
            "pdb-redo will refuse poor quality pdbs"
        )

    return str(out_path)


# ------------------------------------------------------------------------------
# prep inputs
#


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
)
def afdesign(
    pdb_content: bytes | None,
    pdb_name: str,
    is_pdb_id: bool,
    target_chain: str,
    target_hotspot=None,
    target_flexible: bool = True,
    binder_len: int = 30,
    binder_seq=None,
    binder_chain=None,
    set_fixed_aas=None,
    cyclic_peptide: bool = True,
    use_multimer: bool = False,
    num_recycles: int = 3,
    num_models=2,
    pdb_redo: bool = True,
    soft_iters: int = 120,
    hard_iters: int = 32,
):
    """Designs protein binders using AFDesign on Modal, with options for cyclic peptides and various optimization strategies.

    Args:
        pdb_content (bytes | None): Content of the PDB file as bytes, or None if using PDB ID.
        pdb_name (str): Name/identifier for the PDB (for output naming).
        is_pdb_id (bool): True if pdb_name is a PDB/UniProt ID to download, False if using local content.
        target_chain (str): Chain(s) to design binder against. If multiple chains are provided (e.g., "AB"),
                            they will be merged into the first character of the string (e.g., "A").
        target_hotspot (str | None, optional): Restrict loss to predefined positions on target
                                               (e.g., "1-10,12,15"). Defaults to None.
        target_flexible (bool, optional): Allow backbone of target structure to be flexible.
                                          Defaults to True.
        binder_len (int, optional): Length of the binder to hallucinate. Defaults to 30.
        binder_seq (str | None, optional): If defined, will initialize design with this sequence.
                                           Defaults to None.
        binder_chain (str | None, optional): If defined, supervised loss is used (binder_len is ignored).
                                             Set it to the chain of the binder in the PDB file.
                                             Defaults to None.
        set_fixed_aas (str | None, optional): A string of amino acids of the same length as `binder_len`.
                                             Positions with specific amino acids will be fixed during design.
                                             Use 'X' or '-' for positions to be designed. Defaults to None.
        cyclic_peptide (bool, optional): Enforce cyclic peptide design. Defaults to True.
        use_multimer (bool, optional): Use alphafold-multimer for design. Defaults to False.
        num_recycles (int, optional): Number of recycles for the AlphaFold model. Defaults to 3.
        num_models (int, optional): Number of trained AlphaFold models to use during optimization (1-5).
                                    Defaults to 2.
        pdb_redo (bool, optional): If True, attempts to fetch the PDB-REDO version of the input PDB.
                                 Defaults to True.
        soft_iters (int, optional): Number of iterations for soft optimization. Defaults to 120.
        hard_iters (int, optional): Number of iterations for hard optimization. Defaults to 32.

    Returns:
        list[tuple[str, bytes]]: A list of tuples, where each tuple contains an output filename
                                 (e.g., for the log, HTML animation, PDB structure, sequence profile image)
                                 and its byte content.
    """

    from colabdesign import mk_afdesign_model, clear_mem
    from colabdesign.shared.utils import copy_dict
    from colabdesign.af.alphafold.common import residue_constants
    import plotly.express as px

    merge_chains = None
    if len(target_chain) > 1:
        print("merging chains", target_chain)
        merge_chains = list(target_chain)
        target_chain = target_chain[0]

    if binder_seq is not None:
        binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
        binder_len = len(binder_seq)
    print("binder_seq:", binder_seq, "binder_len:", binder_len)
    assert binder_len > 0, "binder_len must be > 0"

    num_models = 5 if num_models == "all" else int(num_models)

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td_in:
        if is_pdb_id:
            # Use the PDB ID directly for get_pdb function
            pdb_input = pdb_name
        else:
            # Write PDB content to temporary file
            temp_pdb_file = Path(td_in) / "input.pdb"
            temp_pdb_file.write_bytes(pdb_content)
            pdb_input = str(temp_pdb_file)

        x = {
            "pdb_filename": pdb_input,
            "chain": target_chain,
            "binder_len": binder_len,
            "binder_chain": binder_chain,
            "hotspot": target_hotspot,
            "use_multimer": use_multimer,
            "rm_target_seq": target_flexible,
        }

        # ------------------------------------------------------------------------------
        # BN added this to extract only chains A and B
        #
        _temp_pdb_file = get_pdb(x["pdb_filename"], pdb_redo=pdb_redo)

        if merge_chains is not None:
            _temp_pdb_file = join_chains(_temp_pdb_file, target_chain, merge_chains)
        x["pdb_filename"] = _temp_pdb_file

        # ------------------------------------------------------------------------------
        # BN add bias for Cysteine cyclic peptide
        #
        _bias = None
        if set_fixed_aas is not None:
            aa_order = residue_constants.restype_order
            assert (
                len(set_fixed_aas) == binder_len
            ), f"add_fixed_aas: {len(set_fixed_aas)} must be same length as binder_len: {binder_len}"
            assert len(aa_order.keys()) == 20, "restype_order has changed"
            _bias = np.zeros((binder_len, len(residue_constants.restype_order)))
            for n, aa in enumerate(set_fixed_aas):
                if aa in aa_order:
                    _bias[n, aa_order[aa]] = 1e16

        # TODO check this -- comes from the colab; something to do with redos?
        x_prev = None
        if "x_prev" not in dir() or x != x_prev:
            clear_mem()
            model = mk_afdesign_model(
                protocol="binder",
                use_multimer=x["use_multimer"],
                num_recycles=num_recycles,
                recycle_mode="sample",
                data_dir=DATA_DIR,
            )
            model.prep_inputs(**x, ignore_missing=False)
            # BN make cyclic peptide
            if cyclic_peptide:
                add_cyclic_offset(model)

            x_prev = copy_dict(x)
            print("target length:", model._target_len)
            print("binder length:", model._binder_len)
            # TODO check this, seems redundant
            binder_len = model._binder_len

        # ------------------------------------------------------------------------------
        # run AfDesign
        #
        # optimizer:
        # `pssm_semigreedy` - uses the designed PSSM to bias semigreedy opt. (Recommended)
        # `3stage` - gradient based optimization (GD) (logits → soft → hard)
        # `pssm` - GD optimize (logits → soft) to get a sequence profile (PSSM).
        # `semigreedy` - tries X random mutations, accepts those that decrease loss
        # `logits` - GD optimize logits inputs (continious)
        # `soft` - GD optimize softmax(logits) inputs (probabilities)
        # `hard` - GD optimize one_hot(logits) inputs (discrete)
        # WARNING: The output sequence from `pssm`,`logits`,`soft` is not one_hot.
        # To get a valid sequence use the other optimizers, or redesign the output backbone
        # with another protocol like ProteinMPNN.
        #

        optimizer: str = "pssm_semigreedy"  # @param ["pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"]

        # advanced GD settings
        GD_method: str = "sgd"  # @param ["adabelief", "adafactor", "adagrad", "adam", "adamw", "fromage", "lamb", "lars", "noisy_sgd", "dpsgd", "radam", "rmsprop", "sgd", "sm3", "yogi"]
        learning_rate: float = 0.1  # @param {type:"raw"}
        norm_seq_grad: bool = True
        dropout: bool = True

        # ------------------------------------------------------------------------------
        # BN added Cysteine cyclic peptide bias here
        #
        if _bias is not None:
            model.restart(seq=binder_seq, bias=_bias)
        else:
            model.restart(seq=binder_seq)

        model.set_optimizer(
            optimizer=GD_method, learning_rate=learning_rate, norm_seq_grad=norm_seq_grad
        )
        models = model._model_names[:num_models]

        flags = {"num_recycles": num_recycles, "models": models, "dropout": dropout}

        if optimizer == "3stage":
            model.design_3stage(120, 60, 10, **flags)
            pssm = softmax(model._tmp["seq_logits"], -1)

        if optimizer == "pssm_semigreedy":
            model.design_pssm_semigreedy(
                soft_iters=soft_iters, hard_iters=hard_iters, **flags
            )
            pssm = softmax(model._tmp["seq_logits"], 1)

        if optimizer == "semigreedy":
            model.design_pssm_semigreedy(0, 32, **flags)
            pssm = None

        if optimizer == "pssm":
            model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
            model.design_soft(32, num_models=1, **flags)
            flags.update({"dropout": False, "save_best": True})
            model.design_soft(10, num_models=num_models, **flags)
            pssm = softmax(model.aux["seq"]["logits"], -1)

        optimizer_funcs = {
            "logits": model.design_logits,
            "soft": model.design_soft,
            "hard": model.design_hard,
        }

        if optimizer in optimizer_funcs:
            optimizer_funcs[optimizer](120, num_models=1, ramp_recycles=True, **flags)
            flags.update({"dropout": False, "save_best": True})
            optimizer_funcs[optimizer](10, num_models=num_models, **flags)
            pssm = softmax(model.aux["seq"]["logits"], -1)

        model.save_pdb(f"{model.protocol}.pdb")

        # display hallucinated protein {run: "auto"}
        color: str = "pLDDT"  # @param ["chain", "pLDDT", "rainbow"]
        show_sidechains: bool = False  # @param {type:"boolean"}
        show_mainchains: bool = True  # @param {type:"boolean"}
        color_HP: bool = False  # @param {type:"boolean"}
        animate: bool = True  # @param {type:"boolean"}

        try:
            model.plot_pdb(
                show_sidechains=show_sidechains,
                show_mainchains=show_mainchains,
                color=color,
                color_HP=color_HP,
                animate=animate,
            )
        except Exception as e:
            print("requires jupyter:", e)

        # takes 30s+ so may not be worth it
        html_content = model.animate(dpi=100)

        out_name = f"{model.protocol}_{pdb_name}_{target_chain}_{model.get_seqs()[0]}_{round(model.get_loss()[-1], 2)}"
        model.save_pdb(f"{out_name}.pdb")

        # ------------------------------------------------------------------------------
        # BN added this
        # Add data into the REMARK section of the PDB file
        #
        pdb_txt = open(f"{out_name}.pdb").read()
        with open(f"{out_name}.pdb", "w") as out:
            for n, (k, v) in enumerate(model._tmp["best"]["aux"]["log"].items()):
                remark_text = f"{k}: {v}"
                remark_line = f"REMARK {n+1:<3} {remark_text:<69}\n"
                out.write(remark_line)
            out.write(pdb_txt)

        model.get_seqs()

        # ------------------------------------------------------------------------------
        # Amino acid probabilties
        #
        # Use residue_constants.restypes for amino acid alphabet
        if "pssm" in dir() and pssm is not None:
            fig = px.imshow(
                pssm.mean(0).T,
                labels=dict(x="positions", y="amino acids", color="probability"),
                y=residue_constants.restypes,
                zmin=0,
                zmax=1,
                template="simple_white",
            )
            fig.update_xaxes(side="top")
            fig.write_image(f"{out_name}.png")

        # plddt etc in here
        log = model._tmp["best"]["aux"]["log"]

        return [
            (f"{out_name}.log", str(log).encode("utf-8")),
            (f"{out_name}.html", html_content.encode("utf-8")),
            (f"{out_name}.pdb", open(f"{out_name}.pdb", "rb").read()),
            (f"{out_name}.png", open(f"{out_name}.png", "rb").read()),
        ]


@app.local_entrypoint()
def main(
    pdb: str,
    target_chain: str,
    target_hotspot: str | None = None,
    target_flexible: bool = True,
    binder_len: int = 12,
    binder_seq: str | None = None,
    binder_chain: str | None = None,
    set_fixed_aas: str | None = None,
    linear_peptide: bool = False,
    use_multimer: bool = False,
    num_recycles: int = 3,
    num_models: int = 2,
    use_rcsb_pdb: bool = False,
    soft_iters: int = 30,
    hard_iters: int = 6,
    num_parallel: int = 1,
    out_dir: str = "./out/afdesign",
    run_name: str | None = None,
):
    """Local entrypoint to run AFDesign predictions, potentially in parallel.

    Note: 120 soft iterations and 32 hard iterations are generally recommended for good results.

    Args:
        pdb (str): PDB code, UniProt code, or path to a PDB file.
        target_chain (str): Chain(s) to design binder against.
        target_hotspot (str | None, optional): Restrict loss to predefined positions on target.
                                               Defaults to None.
        target_flexible (bool, optional): Allow backbone of target structure to be flexible.
                                          Defaults to True.
        binder_len (int, optional): Length of the binder. Defaults to 12.
        binder_seq (str | None, optional): Initial sequence for the binder. Defaults to None.
        binder_chain (str | None, optional): Chain ID of the binder if using supervised loss.
                                             Defaults to None.
        set_fixed_aas (str | None, optional): String to fix amino acids at certain positions in the binder.
                                             Use 'X' for positions to be designed. Defaults to None.
        linear_peptide (bool, optional): If True, design a linear peptide instead of a cyclic one.
                                        Defaults to False (meaning cyclic by default).
        use_multimer (bool, optional): Use alphafold-multimer. Defaults to False.
        num_recycles (int, optional): Number of recycles. Defaults to 3.
        num_models (int, optional): Number of AlphaFold models to use (1-5). Defaults to 2.
        use_rcsb_pdb (bool, optional): If True, force fetching from RCSB PDB instead of PDB-REDO.
                                      Defaults to False (meaning PDB-REDO is preferred).
        soft_iters (int, optional): Number of soft optimization iterations. Defaults to 30.
        hard_iters (int, optional): Number of hard optimization iterations. Defaults to 6.
        num_parallel (int, optional): Number of parallel AFDesign runs to execute. Defaults to 1.

    Returns:
        None
    """

    assert hard_iters >= 2, "fails on hard_iters=1"

    from datetime import datetime

    # Check if input is a file path or PDB ID
    pdb_path = Path(pdb)
    if pdb_path.exists():
        # Local file - read content and pass as bytes
        pdb_content = pdb_path.read_bytes()
        pdb_name = pdb_path.stem
        is_pdb_id = False
    elif len(pdb) in [4, 5] and pdb.replace("-", "").isalnum():
        # Looks like a PDB ID or UniProt ID - pass as string to remote function
        pdb_content = None
        pdb_name = pdb
        is_pdb_id = True
    else:
        raise FileNotFoundError(f"PDB file not found and '{pdb}' doesn't look like a valid PDB/UniProt ID: {pdb}")

    # I can't figure out how to use kwargs with map so order is important
    pdb_redo = not use_rcsb_pdb
    cyclic_peptide = not linear_peptide
    args = tuple(
        (
            pdb_content,
            pdb_name,
            is_pdb_id,
            target_chain,
            target_hotspot,
            target_flexible,
            binder_len,
            binder_seq,
            binder_chain,
            set_fixed_aas,
            cyclic_peptide,
            use_multimer,
            num_recycles,
            num_models,
            pdb_redo,
            soft_iters,
            hard_iters,
        )
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    # use starmap to pass multiple args
    for outputs in afdesign.starmap([args for _ in range(num_parallel)]):
        for out_file, out_content in outputs:
            out_path = Path(out_dir_full) / out_file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_content:
                with open(out_path, "wb") as out:
                    out.write(out_content)
