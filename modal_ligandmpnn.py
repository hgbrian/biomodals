# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""LigandMPNN (superseding ProteinMPNN) for protein design on Modal.

LigandMPNN: https://github.com/dauparas/LigandMPNN

- By default, calc_score is False, because it's quite slow.

## Example EGFR binder
- Design chain C but include chains A and C

```
modal run modal_ligandmpnn.py --input-pdb in/ligandmpnn/1IVO_edited.pdb --extract-chains AC \
--params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" \
--chains_to_design "C" --save_stats 1'
```

## Example EGFR binder
- Outputs will have only chain C
- 15 sequences total (3x5)

```
modal run modal_ligandmpnn.py --input-pdb in/ligandmpnn/1IVO_edited.pdb \
--params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" \
--parse_these_chains_only "C" --save_stats 1 --batch_size 3 --number_of_batches 5'
```

"""

import os
from pathlib import Path

import modal
from modal import App, Image

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

DEFAULT_PARAMS = '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" --save_stats 1'

image = (
    Image.micromamba(python_version="3.11")
    .apt_install(["git", "wget", "gcc", "g++", "libffi-dev"])
    .pip_install(
        [
            "biopython==1.79",
            "filelock==3.13.1",
            "fsspec==2024.3.1",
            "Jinja2==3.1.3",
            "MarkupSafe==2.1.5",
            "mpmath==1.3.0",
            "networkx==3.2.1",
            "numpy==1.23.5",
        ]
    )
    .pip_install(
        [
            "nvidia-cublas-cu12==12.1.3.1",
            "nvidia-cuda-cupti-cu12==12.1.105",
            "nvidia-cuda-nvrtc-cu12==12.1.105",
            "nvidia-cuda-runtime-cu12==12.1.105",
            "nvidia-cudnn-cu12==8.9.2.26",
            "nvidia-cufft-cu12==11.0.2.54",
            "nvidia-curand-cu12==10.3.2.106",
            "nvidia-cusolver-cu12==11.4.5.107",
            "nvidia-cusparse-cu12==12.1.0.106",
            "nvidia-nccl-cu12==2.19.3",
            "nvidia-nvjitlink-cu12==12.4.99",
            "nvidia-nvtx-cu12==12.1.105",
        ]
    )
    .pip_install(
        [
            "ProDy==2.4.1",
            "pyparsing==3.1.1",
            "scipy==1.12.0",
            "sympy==1.12",
            "torch==2.2.1",
            "triton==2.2.0",
            "typing_extensions==4.10.0",
            "ml-collections==0.1.1",
            "dm-tree==0.1.8",
        ]
    )
    .run_commands(
        "git clone https://github.com/dauparas/LigandMPNN.git"
        " && cd LigandMPNN"
        ' && bash get_model_params.sh "./model_params"'
    )
)

app = App("LigandMPNN", image=image)


def extract_chains_inplace(pdb_file: str, extract_chains: str) -> str:
    """Extract specific chains from a PDB file in-place.
    
    Args:
        pdb_file (str): Path to the PDB file.
        extract_chains (str): Chain identifiers to extract.
        
    Returns:
        str: Path to the modified PDB file.
    """
    from prody import parsePDB, writePDB

    chains = parsePDB(pdb_file, chain=extract_chains.replace(",", ""))
    writePDB(pdb_file, chains)
    return pdb_file


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def ligandmpnn(
    input_pdb_str: str,
    input_pdb_name: str,
    params_str: str | None = None,
    calc_score: bool = False,
    score_params_str: str | None = None,
    extract_chains: str | None = None,
) -> list[tuple[Path, bytes]]:
    """Runs LigandMPNN on a PDB input string.
    
    Args:
        input_pdb_str (str): PDB file content as a string.
        input_pdb_name (str): Name for the PDB file.
        params_str (str | None): Optional string of additional parameters for LigandMPNN.
        calc_score (bool): Whether to calculate scores for the output.
        score_params_str (str | None): Optional parameters for scoring.
        extract_chains (str | None): Chain identifiers to extract from the PDB.
        
    Returns:
        list[tuple[Path, bytes]]: A list of tuples containing output file paths and their byte content.
    """
    from subprocess import run

    out_dir = "./out"

    open(input_pdb_name, "w").write(input_pdb_str)
    if extract_chains is not None:
        input_pdb_name = extract_chains_inplace(input_pdb_name, extract_chains)

    # --------------------------------------------------------------------------
    # Run LigandMPNN
    # By default, use a protein model
    if params_str is None:
        params_str = DEFAULT_PARAMS

    cmd = f'python /LigandMPNN/run.py --pdb_path "{input_pdb_name}" --out_folder "{out_dir}" {params_str}'
    print(cmd)
    run(cmd, shell=True, check=True)

    # --------------------------------------------------------------------------
    # Score the output from LigandMPNN
    # Defaults from https://github.com/dauparas/LigandMPNN, not sure what some of these do
    #
    if calc_score:
        if score_params_str is None:
            score_params_str = (
                f' --seed 111 --model_type "protein_mpnn" --checkpoint_protein_mpnn {ckpt}'
                " --single_aa_score 1 --use_sequence 1 --batch_size 1 --number_of_batches 10"
            )

        for backbone in (Path(out_dir) / "backbones").glob("*.pdb"):
            score_params_str_ = score_params_str + f' --pdb_path "{backbone}"'

            cmd_score = f'python /LigandMPNN/score.py --out_folder "{out_dir}" {score_params_str_}'
            print(cmd_score)
            run(cmd_score, shell=True, check=True)

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if Path(out_file).is_file()
    ]


@app.local_entrypoint()
def main(
    input_pdb: str,
    params_str: str | None = None,
    calc_score: bool = False,
    score_params_str: str | None = None,
    extract_chains: str | None = None,
    run_name: str | None = None,
    out_dir: str = "./out/ligandmpnn",
):
    """Local entrypoint to run LigandMPNN predictions using Modal.
    
    Args:
        input_pdb (str): Path to an input PDB file.
        params_str (str | None): Optional string of additional parameters for LigandMPNN.
        calc_score (bool): Whether to calculate scores for the output.
        score_params_str (str | None): Optional parameters for scoring.
        extract_chains (str | None): Chain identifiers to extract from the PDB.
        run_name (str | None): Optional name for the run, used for organizing output files.
                               If None, a timestamp-based name is used.
        out_dir (str): Directory to save the output files. Defaults to "./out/ligandmpnn".
        
    Returns:
        None
    """
    from datetime import datetime

    input_pdb_str = open(input_pdb).read()

    outputs = ligandmpnn.remote(
        input_pdb_str, Path(input_pdb).name, params_str, calc_score, score_params_str, extract_chains
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
            out.write(out_content)
