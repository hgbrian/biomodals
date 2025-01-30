"""Run AF2Rank

e.g.,
```
wget https://files.rcsb.org/download/4KRL.pdb
modal run modal_af2rank.py --input-pdb 4RKL.pdb
```

using AF2 multimer instead, and use both chains
```
modal run modal_af2rank.py --input-pdb 4KRL.pdb --model-name "model_1_multimer_v3" --chains "A,B"
```

"""

import os
from pathlib import Path

from modal import App, Image, Mount

GPU = os.environ.get("MODAL_GPU", "A10G")
TIMEOUT = os.environ.get("MODAL_TIMEOUT", 20 * 60)
LOCAL_MSA_DIR = "msas"
if not Path(LOCAL_MSA_DIR).exists():
    Path(LOCAL_MSA_DIR).mkdir(exist_ok=True)

image = (
    Image
    .micromamba()
    .apt_install("wget", "curl", "git", "g++")
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git@v1.1.1", "jax[cuda12_pip]")
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign",
        "mkdir params",
        "curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params",
        "mv params /root/",
        "wget -qnc https://zhanggroup.org/TM-score/TMscore.cpp",
        "g++ -static -O3 -ffast-math -lm -o TMscore TMscore.cpp",
        "cp TMscore /root/"
    )
    .pip_install("ipython")
)

app = App("af2rank", image=image)


with image.imports():
    #@title import libraries
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    import os

    import jax
    import matplotlib.pyplot as plt
    import numpy as np
    from colabdesign import clear_mem, mk_af_model
    from colabdesign.shared.utils import copy_dict
    from scipy.stats import spearmanr

    def tmscore(x,y):
        # save to dumpy pdb files
        for n,z in enumerate([x,y]): 
            out = open(f"{n}.pdb","w")
            for k,c in enumerate(z):
                out.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f   d%4.2f\n" 
                                        % (k+1,"CA","ALA","A",k+1,c[0],c[1],c[2],1,0))
            out.close()

        # pass to TMscore
        output = os.popen('./TMscore 0.pdb 1.pdb')

        # parse outputs
        parse_float = lambda x: float(x.split("=")[1].split()[0])
        o = {}
        for line in output:
            line = line.rstrip()
            if line.startswith("RMSD"): o["rms"] = parse_float(line)
            if line.startswith("TM-score"): o["tms"] = parse_float(line)
            if line.startswith("GDT-TS-score"): o["gdt"] = parse_float(line)
        
        return o
        
    def plot_me(scores, x="tm_i", y="composite", 
                title=None, diag=False, scale_axis=True, dpi=100, **kwargs):
        def rescale(a,amin=None,amax=None):
            a = np.copy(a)
            if amin is None: amin = a.min()
            if amax is None: amax = a.max()
            a[a < amin] = amin
            a[a > amax] = amax
            return (a - amin)/(amax - amin)

        plt.figure(figsize=(5,5), dpi=dpi)
        if title is not None: plt.title(title)
        x_vals = np.array([k[x] for k in scores])
        y_vals = np.array([k[y] for k in scores])
        c = rescale(np.array([k["plddt"] for k in scores]),0.5,0.9)
        plt.scatter(x_vals, y_vals, c=c*0.75, s=5, vmin=0, vmax=1, cmap="gist_rainbow",
                    **kwargs)
        if diag:
            plt.plot([0,1],[0,1],color="black")
        
        labels = {"tm_i":"TMscore of Input",
                  "tm_o":"TMscore of Output",
                  "tm_io":"TMscore between Input and Output",
                  "ptm":"Predicted TMscore (pTM)",
                  "i_ptm":"Predicted interface TMscore (ipTM)",
                  "plddt":"Predicted LDDT (pLDDT)",
                  "composite":"Composite"}

        plt.xlabel(labels.get(x,x));  plt.ylabel(labels.get(y,y))
        if scale_axis:
            if x in labels: plt.xlim(-0.1, 1.1)
            if y in labels: plt.ylim(-0.1, 1.1)
        
        print(spearmanr(x_vals,y_vals).correlation)

    class af2rank:
        def __init__(self, pdb, chain=None, model_name="model_1_ptm", model_names=None):
            self.args = {"pdb":pdb, "chain":chain,
                         "use_multimer":("multimer" in model_name),
                         "model_name":model_name,
                         "model_names":model_names}
            self.reset()

        def reset(self):
            self.model = mk_af_model(protocol="fixbb",
                                     use_templates=True,
                                     use_multimer=self.args["use_multimer"],
                                     debug=False,
                                     model_names=self.args["model_names"])
            
            self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
            self.model.set_seq(mode="wildtype")
            self.wt_batch = copy_dict(self.model._inputs["batch"])
            self.wt = self.model._wt_aatype

        def set_pdb(self, pdb, chain=None):
            if chain is None: chain = self.args["chain"]
            self.model.prep_inputs(pdb, chain=chain)
            self.model.set_seq(mode="wildtype")
            self.wt = self.model._wt_aatype

        def set_seq(self, seq):
            self.model.set_seq(seq=seq)
            self.wt = self.model._params["seq"][0].argmax(-1)

        def _get_score(self):
            score = copy_dict(self.model.aux["log"])

            score["plddt"] = score["plddt"]
            score["pae"] = 31.0 * score["pae"]
            score["rmsd_io"] = score.pop("rmsd",None)

            i_xyz = self.model._inputs["batch"]["all_atom_positions"][:,1]
            o_xyz = np.array(self.model.aux["atom_positions"][:,1])

            # TMscore to input/output
            if hasattr(self,"wt_batch"):
                n_xyz = self.wt_batch["all_atom_positions"][:,1]
                score["tm_i"] = tmscore(n_xyz,i_xyz)["tms"]
                score["tm_o"] = tmscore(n_xyz,o_xyz)["tms"]

            # TMscore between input and output
            score["tm_io"] = tmscore(i_xyz,o_xyz)["tms"]

            # composite score
            score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
            return score
        
        def predict(self, pdb=None, seq=None, chain=None, 
                    input_template=True, model_name=None,
                    rm_seq=True, rm_sc=True, rm_ic=False,
                    recycles=1, iterations=1,
                    output_pdb=None, extras=None, verbose=True):

            if model_name is not None:
                self.args["model_name"] = model_name
                if "multimer" in model_name: 
                    if not self.args["use_multimer"]:
                        self.args["use_multimer"] = True
                        self.reset()
                else:
                    if self.args["use_multimer"]:
                        self.args["use_multimer"] = False
                        self.reset()

            if pdb is not None: self.set_pdb(pdb, chain)
            if seq is not None: self.set_seq(seq)

            # set template sequence
            self.model._inputs["batch"]["aatype"] = self.wt

            # set other options
            self.model.set_opt(
                    template=dict(rm_ic=rm_ic),
                    num_recycles=recycles
            )
            self.model._inputs["rm_template"][:] = not input_template
            self.model._inputs["rm_template_sc"][:] = rm_sc
            self.model._inputs["rm_template_seq"][:] = rm_seq

            # "manual" recycles using templates
            ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
            for i in range(iterations):
                self.model.predict(models=self.args["model_name"], verbose=False)
                if i < iterations - 1:
                    self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
                else:
                    self.model._inputs["batch"]["all_atom_positions"] = ini_atoms
            
            score = self._get_score()
            if extras is not None:
                score.update(extras)

            if output_pdb is not None:
                self.model.save_pdb(output_pdb)
            
            if verbose:
                print_list = ["tm_i","tm_o","tm_io","composite","ptm","i_ptm","plddt","fitness","id"]
                print_score = lambda k: f"{k} {score[k]:.4f}" if isinstance(score[k],float) else f"{k} {score[k]}"
                print(*[print_score(k) for k in print_list if k in score])

            return score

@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
)
def run_af2rank(
    pdb_str: str,
    pdb_name: str|None = None,
    chains: str = "A",
    model_name: str = "model_1_ptm",
    num_recycles: int = 1,
    num_iterations: int = 1,
    mask_sequence: bool = False,
    mask_sidechains: bool = False,
    mask_interchain: bool = False,
):
    import json

    if pdb_name is None:
        pdb_name = "out.pdb"

    Path(in_pdb := "/tmp/in_af2rank/input.pdb")
    Path(in_pdb).parent.mkdir(parents=True, exist_ok=True)
    Path(in_pdb).write_text(pdb_str)

    Path(out_dir := "/tmp/out_af2rank").mkdir(parents=True, exist_ok=True)

    SETTINGS = {'rm_seq': mask_sequence, 'rm_sc': mask_sidechains, 'rm_ic': mask_interchain, 
                'recycles': num_recycles, 'iterations': num_iterations, 
                'model_name': model_name}
    print("settings:", SETTINGS)
    af = af2rank(in_pdb, chains, model_name=SETTINGS["model_name"])
    score = af.predict(pdb=in_pdb, **SETTINGS, extras={"id":in_pdb})
    
    results = SETTINGS | {"score": score, "chains": chains}
    open(Path(out_dir) / "results.json", "w").write(json.dumps(results))
    open(Path(out_dir) / f"{Path(pdb_name).stem}_af2rank.pdb", "w").write(pdb_str)

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if out_file.is_file()
    ]


@app.local_entrypoint()
def main(
    input_pdb: str,
    chains: str = "A",
    model_name: str = None,
    num_recycles: int = 1,
    num_iterations: int = 1,
    mask_sequence: bool = False,
    mask_sidechains: bool = False,
    mask_interchain: bool = False,
    out_dir = "./out/af2rank",
    run_name = None,
):
    # model_{model_num}_multimer_v3
    from datetime import datetime

    pdb_str = open(input_pdb).read()

    # model_{n}_ptm or model_{n}_multimer_v3
    if model_name is None:
        model_name = "model_1_ptm"

    outputs = run_af2rank.remote(
        pdb_str=pdb_str,
        pdb_name=Path(input_pdb).name,
        chains=chains,
        model_name=model_name,
        num_recycles=num_recycles,
        num_iterations=num_iterations,
        mask_sequence=mask_sequence,
        mask_sidechains=mask_sidechains,
        mask_interchain=mask_interchain,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / out_file).parent.mkdir(parents=True, exist_ok=True)
        with open((Path(out_dir_full) / out_file), "wb") as out:
            out.write(out_content or "")
