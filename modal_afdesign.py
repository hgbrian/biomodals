"""
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
"""

import re
import subprocess
import tempfile
import warnings

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from utils.pdb_utils import get_pdb

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.NeighborSearch import NeighborSearch
from scipy.special import softmax

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from colabdesign.af.alphafold.common import residue_constants
import plotly.express as px

from modal import Image, Mount, Stub

LOCAL_IN = "in/afdesign"
LOCAL_OUT = "out/afdesign"
REMOTE_IN = "/in"
GPU = "a100"
DATA_DIR = "/"

warnings.simplefilter(action='ignore', category=FutureWarning)

stub = Stub()
image = (Image
        .debian_slim()
        .apt_install("git", "wget", "aria2", "ffmpeg")
        .pip_install("jax[cuda12_pip]", find_links="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
        .pip_install("pdb-tools==2.4.8", "ffmpeg-python==0.2.0", "plotly==5.18.0", "kaleido==0.2.1")
        .pip_install("git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
        .run_commands("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign;"
                        "mkdir /params")
        .run_commands("aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar;"
                        "tar -xf alphafold_params_2022-12-06.tar -C /params")
)


# ------------------------------------------------------------------------------
# BN added this function
#
def add_cyclic_offset(self):
    """add cyclic offset to connect N and C term head to tail"""
    def _cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i,i+L],-1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
        return np.sign(offset) * c_offset

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])
  
    if self.protocol == "binder":
        c_offset = _cyclic_offset(self._binder_len)
        offset[self._target_len:,self._target_len:] = c_offset
  
    if self.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in self._lengths:
            offset[Ln:Ln+L,Ln:Ln+L] = _cyclic_offset(L)
            Ln += L

    self._inputs["offset"] = offset


# ------------------------------------------------------------------------------
# BN added this function
#
class ResidueRangeSelect(Select):
    def __init__(self, chain_ids, start, end):
        self.chain_ids = chain_ids
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        within_range = self.start <= residue.get_id()[1] <= self.end
        correct_chain = residue.parent.id in self.chain_ids
        return within_range and correct_chain

def extract_residues_from_pdb(pdb_file, chain_ids, start_residue, end_residue):
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
    """Use pdb-tools to combine the pdb file into one chain.
    Probably unnecessary!"""
    with NamedTemporaryFile(suffix=".pdb", delete=False) as tf:
        subprocess.run(f"pdb_selchain -{','.join(merge_chains)} {pdb_file} | "
                       f"pdb_chain -{target_chain} | pdb_reres -1 > {tf.name}", shell=True, check=True)
        return tf.name

# ------------------------------------------------------------------------------
# BN added this function
#
def get_nearby_residues(pdb_file, ligand_id, distance=8.0):
    """Report the residues within `distance` of the ligand as a dict."""
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)

    # Get all atoms in the protein
    protein_atoms = [atom for atom in structure.get_atoms()
                     if atom.parent.get_resname() != ligand_id
                     and is_aa(atom.parent)]

    # Get all atoms in the ligand
    ligand_atoms = [atom for atom in structure.get_atoms() if atom.parent.get_resname() == ligand_id]

    # Create a NeighborSearch object
    ns = NeighborSearch(protein_atoms)

    # Find all protein atoms within `distance` of any ligand atom
    nearby_atoms = []
    for ligand_atom in ligand_atoms:
        nearby_atoms.extend(ns.search(ligand_atom.coord, distance))

    # Get the residues corresponding to these atoms
    nearby_residues = {atom.parent for atom in nearby_atoms}

    return nearby_residues

# ------------------------------------------------------------------------------
# prep inputs
#

@stub.function(image=image, gpu=GPU, timeout=60*120,
               mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)])
def afdesign(pdb:str, target_chain:str, target_hotspot=None, target_flexible:bool=True,
             binder_len:int=30, binder_seq=None, binder_chain=None,
             set_fixed_aas = None,
             cyclic_peptide:bool=True,
             use_multimer:bool = False,
             num_recycles:int = 3,
             num_models = 2,
             pdb_redo:bool=True,
             soft_iters:int=120,
             hard_iters:int=32):
    """
    pdb: enter PDB code or UniProt code (to fetch AlphaFoldDB model) or leave blink to upload your own
    target_chain: chain to design binder against
    target_hotspot: restrict loss to predefined positions on target (eg. "1-10,12,15")
    target_flexible: allow backbone of target structure to be flexible

    binder_len: length of binder to hallucination
    binder_seq: if defined, will initialize design with this sequence
    binder_chain: if defined, supervised loss is used (binder_len is ignored).
                  Set it to the chain of the binder in the PDB file?

    cyclic_peptide: enforce cyclic peptide

    use_multimer: use alphafold-multimer for design
    num_recycles: #@param ["0", "1", "3", "6"] {type:"raw"}
    num_models: number of trained models to use during optimization  #@param ["1", "2", "3", "4", "5", "all"]

    soft_iters: number of iterations for soft optimization
    hard_iters: number of iterations for hard optimization

    """

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

    if (Path(REMOTE_IN) / Path(pdb).relative_to(LOCAL_IN)).is_file():
        pdb = str(Path(REMOTE_IN) / Path(pdb).relative_to(LOCAL_IN))

    x = {"pdb_filename":pdb,
         "chain":target_chain,
         "binder_len":binder_len,
         "binder_chain":binder_chain,
         "hotspot":target_hotspot,
         "use_multimer":use_multimer,
         "rm_target_seq":target_flexible}

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
        assert len(set_fixed_aas) == binder_len, f"add_fixed_aas: {len(set_fixed_aas)} must be same length as binder_len: {binder_len}"
        assert len(aa_order.keys()) == 20, "restype_order has changed"
        _bias = np.zeros((binder_len, len(residue_constants.restype_order)))
        for n, aa in enumerate(set_fixed_aas):
            if aa in aa_order:
                _bias[n, aa_order[aa]] = 1e16

    # TODO check this -- comes from the colab; something to do with redos?
    x_prev = None
    if "x_prev" not in dir() or x != x_prev:
        clear_mem()
        model = mk_afdesign_model(protocol="binder",
                                  use_multimer=x["use_multimer"],
                                  num_recycles=num_recycles,
                                  recycle_mode="sample",
                                  data_dir=DATA_DIR
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

    optimizer:str = "pssm_semigreedy" #@param ["pssm_semigreedy", "3stage", "semigreedy", "pssm", "logits", "soft", "hard"]

    # advanced GD settings
    GD_method:str = "sgd" #@param ["adabelief", "adafactor", "adagrad", "adam", "adamw", "fromage", "lamb", "lars", "noisy_sgd", "dpsgd", "radam", "rmsprop", "sgd", "sm3", "yogi"]
    learning_rate:float = 0.1 #@param {type:"raw"}
    norm_seq_grad:bool = True
    dropout:bool = True

    # ------------------------------------------------------------------------------
    # BN added Cysteine cyclic peptide bias here
    #
    if _bias is not None:
        model.restart(seq=binder_seq, bias=_bias)
    else:
        model.restart(seq=binder_seq)

    model.set_optimizer(optimizer=GD_method,
                        learning_rate=learning_rate,
                        norm_seq_grad=norm_seq_grad)
    models = model._model_names[:num_models]

    flags = {"num_recycles":num_recycles,
             "models":models,
             "dropout":dropout}

    if optimizer == "3stage":
        model.design_3stage(120, 60, 10, **flags)
        pssm = softmax(model._tmp["seq_logits"],-1)

    if optimizer == "pssm_semigreedy":
        model.design_pssm_semigreedy(soft_iters=soft_iters, hard_iters=hard_iters, **flags)
        pssm = softmax(model._tmp["seq_logits"],1)

    if optimizer == "semigreedy":
        model.design_pssm_semigreedy(0, 32, **flags)
        pssm = None

    if optimizer == "pssm":
        model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
        model.design_soft(32, num_models=1, **flags)
        flags.update({"dropout":False,"save_best":True})
        model.design_soft(10, num_models=num_models, **flags)
        pssm = softmax(model.aux["seq"]["logits"],-1)

    O = {"logits":model.design_logits,
         "soft":model.design_soft,
         "hard":model.design_hard}

    if optimizer in O:
        O[optimizer](120, num_models=1, ramp_recycles=True, **flags)
        flags.update({"dropout":False, "save_best":True})
        O[optimizer](10, num_models=num_models, **flags)
        pssm = softmax(model.aux["seq"]["logits"],-1)

    model.save_pdb(f"{model.protocol}.pdb")

    # display hallucinated protein {run: "auto"}
    color:str = "pLDDT" #@param ["chain", "pLDDT", "rainbow"]
    show_sidechains:bool = False #@param {type:"boolean"}
    show_mainchains:bool = True #@param {type:"boolean"}
    color_HP:bool = False #@param {type:"boolean"}
    animate:bool = True #@param {type:"boolean"}

    model.plot_pdb(show_sidechains=show_sidechains,
                    show_mainchains=show_mainchains,
                    color=color, color_HP=color_HP, animate=animate)

    # takes 30s+ so may not be worth it
    html_content = model.animate(dpi=100)

    out_name = f"{model.protocol}_{Path(pdb).stem}_{target_chain}_{model.get_seqs()[0]}_{round(model.get_loss()[-1], 2)}"
    model.save_pdb(f"{out_name}.pdb")

    # ------------------------------------------------------------------------------
    # BN added this
    # Add data into the REMARK section of the PDB file
    # 
    pdb_txt = open(f"{out_name}.pdb").read()
    with open(f"{out_name}.pdb", 'w') as out:
        for n, (k, v) in enumerate(model._tmp["best"]["aux"]["log"].items()):
            remark_text = f"{k}: {v}"
            remark_line = f"REMARK {n+1:<3} {remark_text:<69}\n"
            out.write(remark_line)
        out.write(pdb_txt)

    model.get_seqs()

    # ------------------------------------------------------------------------------
    # Amino acid probabilties
    #
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    if "pssm" in dir() and pssm is not None:
      fig = px.imshow(pssm.mean(0).T,
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

    return [(f"{out_name}.log", str(log).encode("utf-8")),
            (f"{out_name}.html", html_content.encode("utf-8")),
            (f"{out_name}.pdb", open(f"{out_name}.pdb", "rb").read()),
            (f"{out_name}.png", open(f"{out_name}.png", "rb").read())]


@stub.local_entrypoint()
def main(pdb:str, target_chain:str,
         target_hotspot:str=None,
         target_flexible:bool=True,
         binder_len:int=12,
         binder_seq:str=None,
         binder_chain:str=None,
         set_fixed_aas:str=None,
         linear_peptide:bool=False,
         use_multimer:bool=False,
         num_recycles:int=3,
         num_models:int=2,
         use_rcsb_pdb:bool=False,
         soft_iters:int=30,
         hard_iters:int=6,
         num_parallel:int=1):
    """120 soft iters, 32 hard iters is recommended"""

    assert hard_iters >= 2, "fails on hard_iters=1"

    # I can't figure out how to use kwargs with map so order is important
    pdb_redo = not use_rcsb_pdb
    cyclic_peptide = not linear_peptide
    args = tuple((pdb, target_chain, target_hotspot, target_flexible,
                  binder_len, binder_seq, binder_chain, set_fixed_aas,
                  cyclic_peptide, use_multimer, num_recycles, num_models, pdb_redo,
                  soft_iters, hard_iters))

    # use starmap to pass multiple args
    for outputs in afdesign.starmap([args for _ in range(num_parallel)]):
        for (out_file, out_content) in outputs:
            out_path = Path(LOCAL_OUT) / out_file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_content:
                with open(out_path, 'wb') as out:
                    out.write(out_content)
