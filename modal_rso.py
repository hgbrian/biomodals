"""
adapted from https://github.com/coreyhowe999/RSO

Example:
```
modal run modal_rso.py --input-pdb ABC1.pdb --run-name ABC1 --binder-len 60
```
"""

import modal
import os
from datetime import datetime  # Add this import
from pathlib import Path

GPU = os.environ.get("MODAL_GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 180))

image = (
    modal.Image.debian_slim()
    .apt_install("wget", "git")
    .pip_install(
        "numpy",
        "pandas",
        "biopython",
        "jax[cuda]",
        "git+https://github.com/sokrypton/ColabDesign.git",
    )
    .run_commands(
        [
            "mkdir -p /root/params",
            "wget -P /root/params/ https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
            "tar -xvf /root/params/alphafold_params_2022-12-06.tar -C /root/params/",
            "rm /root/params/alphafold_params_2022-12-06.tar",
        ]
    )
)

app = modal.App("rso", image=image)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
)
def rso(pdb_name, pdb_str, traj_iters, binder_len, chain, hotspot=None, thresholds=None):
    # Import colabdesign modules here
    from colabdesign import mk_afdesign_model, clear_mem
    from colabdesign.mpnn import mk_mpnn_model
    import jax
    import jax.numpy as jnp
    from colabdesign.af.alphafold.common import residue_constants

    import pandas as pd

    pdb_path = str(Path("/tmp/in_rso") / pdb_name)
    Path(pdb_path).parent.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        # e.g. proper thresholds vs extremely permissive
        # thresholds = {"rmsd": 2, "plddt": 0.15, "pae": 0.4}
        thresholds = {"rmsd": 10, "plddt": 1, "pae": 1}

    with open(pdb_path, "w") as f:
        f.write(pdb_str)

    def add_rg_loss(self, weight=0.1):
        """add radius of gyration loss"""

        def loss_fn(inputs, outputs):
            xyz = outputs["structure_module"]
            ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]

            ca = ca[-self._binder_len :]

            rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
            rg_th = 2.38 * ca.shape[0] ** 0.365
            rg = jax.nn.elu(rg - rg_th)
            return {"rg": rg}

        self._callbacks["model"]["loss"].append(loss_fn)
        self.opt["weights"]["rg"] = weight

    # Remove all PDB files with 'binder_design' in the file name
    for pdb_file in Path(".").glob("**/*binder_design*.pdb"):
        pdb_file.unlink()

    #
    # AFDesign steps
    #
    clear_mem()
    af_model = mk_afdesign_model(protocol="binder")
    add_rg_loss(af_model)
    af_model.prep_inputs(pdb_filename=pdb_path, chain=chain, hotspot=hotspot, binder_len=binder_len)

    #
    # Adjust as needed
    #
    af_model.restart(mode=["gumbel", "soft"])
    af_model.set_weights(helix=-0.2, plddt=0.1, pae=0.1, rg=0.5, i_pae=5.0, i_con=2.0)
    af_model.design_logits(traj_iters)
    af_model.save_pdb("backbone.pdb")

    ### SEQ DESIGN AND FILTER ####

    binder_model = mk_afdesign_model(protocol="binder", use_multimer=True, use_initial_guess=True)
    monomer_model = mk_afdesign_model(protocol="fixbb")

    # binder_model.set_weights(i_pae=1.0)

    mpnn_model = mk_mpnn_model(weights="soluble")
    mpnn_model.prep_inputs(pdb_filename="backbone.pdb", chain="A,B", fix_pos="A", rm_aa="C")

    samples = mpnn_model.sample_parallel(8, temperature=0.01)
    monomer_model.prep_inputs(pdb_filename="backbone.pdb", chain="B")
    binder_model.prep_inputs(
        pdb_filename="backbone.pdb",
        chain="A",
        binder_chain="B",
        use_binder_template=True,
        rm_template_ic=True,
    )

    results_df = pd.DataFrame()

    # output results

    for j, seq in enumerate(samples["seq"]):
        print("Predicting binder only")
        monomer_model.predict(seq=seq[-binder_len:], num_recycles=3)
        if monomer_model.aux["losses"]["rmsd"] < thresholds["rmsd"]:
            print("Passed! Predicting binder with receptor using AF Multimer")
            binder_model.predict(seq=seq[-binder_len:], num_recycles=3)
            if monomer_model.aux["losses"]["plddt"] < thresholds["plddt"] and monomer_model.aux["losses"]["pae"] < thresholds["pae"]:
                binder_model.save_pdb(f"{Path(pdb_name).stem}_binder_design_{j}.pdb")
                results_df.loc[j, "pdb_id"] = f"{Path(pdb_name).stem}_binder_design_{j}.pdb"
                results_df.loc[j, "seq"] = seq[-binder_len:]
                for key in binder_model.aux["log"]:
                    results_df.loc[j, key] = binder_model.aux["log"][key]
                for weight in af_model.opt["weights"]:
                    results_df.loc[j, f"weights_{key}"] = weight
        else:
            print(f"Failed! RMSD: {monomer_model.aux['losses']['rmsd']} >= 2.0")

    results_df.to_csv("binder_design_scores.csv", index=False)

    return [
        (str(out_file), open(out_file, "rb").read())
        for out_file in Path(".").glob("**/*")
        if Path(out_file).is_file()
        if Path(out_file).suffix != ".npz"
    ]


@app.local_entrypoint()
def main(
    input_pdb: str,
    num_designs: int = 1,
    traj_iters: int = 100,
    binder_len: int = 80,
    chain: str = "A",
    hotspot: str|None = None,
    thresholds: str|None = None,
    out_dir="./out/rso",
    run_name=None,
):
    pdb_str = open(input_pdb).read()
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_outputs = rso.starmap(
        [(Path(input_pdb).name, pdb_str, traj_iters, binder_len, chain, hotspot, thresholds)
         for _ in range(num_designs)]
    )

    for bb_num, outputs in enumerate(all_outputs):
        for out_file, out_content in outputs:
            output_path = Path(out_dir) / (run_name or today) / str(bb_num) / out_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as out:
                out.write(out_content)
