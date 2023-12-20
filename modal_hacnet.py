from pathlib import Path
from datetime import datetime

from modal import Image, Mount, Stub

MODAL_IN = "./modal_in/hacnet"
MODAL_OUT = "./modal_out/hacnet"

stub = Stub()

image = (Image
         .micromamba(python_version="3.10")
         .apt_install(["git", "wget", "ffmpeg", "libsm6", "libxext6"])
         #.micromamba_install(["openbabel"], channels=["conda-forge"]) # openbabel is not available for python 3.10
         .micromamba_install(["pymol-open-source"], channels=["conda-forge"])
         .pip_install(["openbabel-wheel", "biopandas", "h5py", "matplotlib", "HACNet"])
         .pip_install(["torch==2.0.1"], index_url="https://download.pytorch.org/whl/cu118")
         .pip_install(["torch-geometric", "torch-scatter", "torch-sparse"], find_links="https://data.pyg.org/whl/torch-2.0.1+cu118.html")
         .run_commands("git clone https://github.com/gregory-kyro/HAC-Net.git")
         .run_commands("mkdir /content")
        )

@stub.function(image=image, gpu="T4", timeout=60*15,
               mounts=[Mount.from_local_dir(MODAL_IN, remote_path="/in")])
def run_hacnet(pdbs_ligands:list, verbose=False) -> dict:
    from HACNet.functions import predict_pkd

    # define xml file containing atomic features
    elements_xml = '/HAC-Net/HACNet/element_features.xml'
    # define 3D-CNN parameter file
    cnn_params = '/HAC-Net/HACNet/parameter_files/CNN_parameters.pt'
    # define GCN parameter file
    gcn0_params = '/HAC-Net/HACNet/parameter_files/GCN0_parameters.pt'
    # define other GCN parameter file
    gcn1_params = '/HAC-Net/HACNet/parameter_files/GCN1_parameters.pt'
    # define MLP parameter file
    mlp_params = '/HAC-Net/HACNet/parameter_files/MLP_parameters.pt'

    pkds = {}
    for protein, ligand in pdbs_ligands:
        pkd = predict_pkd(protein_pdb=f"/in/{Path(protein).name}",
                          ligand_mol2=f"/in/{Path(ligand).name}",
                          elements_xml=elements_xml,
                          cnn_params=cnn_params, gcn0_params=gcn0_params, gcn1_params=gcn1_params,
                          mlp_params=mlp_params, verbose=verbose)
        pkds[(protein, ligand)] = pkd

    return pkds

@stub.local_entrypoint()
def main(pdb:str, mol2:str, all_by_all:bool=False):
    if all_by_all:
        pdbs_ligands = [(_pdb.strip(), _mol2.strip()) for _pdb in pdb.split(",") for _mol2 in mol2.split(",") ]
    else:
        pdbs_ligands = [(_pdb.strip(), _mol2.strip()) for _pdb, _mol2 in zip(pdb.split(","), mol2.split(",")) ]


    pkds = run_hacnet.remote(pdbs_ligands)

    today = datetime.today().strftime("%Y%m%d")
    outfile = (Path(MODAL_OUT) / f"{today}_{'-'.join(Path(_pdb).name for _pdb in pdb.split(',')[0:1])}"
               f"_{'-'.join(Path(_mol2).name for _mol2 in mol2.split(',')[0:1])}_pkds.tsv")

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, 'w') as out:
        for (pdb, ligand), pkd in pkds.items():
            out.write(f"{Path(pdb).stem}\t{Path(ligand).stem}\t{round(float(pkd), 3)}\n")
