"""Run AlphaFold2 / AF2-multimer.

- It runs only the first entry in a fasta file.
- If providing a complex, e.g., a binder and target pair,
  provide one sequence with the binder and target separated by ":"
  Provide the binder_len to get iPAE scoring
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("MODAL_GPU", "A10G")
TIMEOUT = os.environ.get("MODAL_TIMEOUT", 20 * 60)

image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install("colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold")
    .micromamba_install("kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"])
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a100",
    )
    .run_commands("python -m colabfold.download")
)

app = App("alphafold", image=image)


def score_af2m_binding(af2m_dict: str, binder_len: int, target_len: int = None) -> dict:
    """
    Calculate binding scores from AlphaFold2 multimer prediction results.
    The binder is assumed to be the first part of the sequence up to `binder_len`,
    with the target being the remainder, unless otherwise specified.

    Parameters:
    af_multimer_dict (str): From AlphaFold2 multimer JSON file
    binder_len (int): Length of the binder protein sequence.
    target_len (int): Length of the target protein sequence (optional)

    Returns:
    dict: A dictionary containing the following scores:
        - plddt_binder (float): Average pLDDT score for the binder.
        - plddt_target (float): Average pLDDT score for the target.
        - pae_binder (float): Average PAE score within the binder.
        - pae_target (float): Average PAE score within the target.
        - ipae (float): Average PAE score for the binder-target interaction.
    """

    import numpy as np

    target_end = (binder_len + target_len) if target_len is not None else None

    # --------------------------------------------------------------------------
    # pLDDT
    #
    plddt_array = np.array(af2m_dict["plddt"])

    plddt_binder = np.mean(plddt_array[:binder_len])
    plddt_target = np.mean(plddt_array[binder_len:target_end])

    # --------------------------------------------------------------------------
    # PAE
    #
    pae_array = np.array(af2m_dict["pae"])

    pae_binder = np.mean(pae_array[:binder_len, :binder_len])
    pae_target = np.mean(pae_array[binder_len:target_end, binder_len:target_end])
    ipae = np.mean(
        [
            np.mean(pae_array[:binder_len, binder_len:target_end]),
            np.mean(pae_array[binder_len:target_end, :binder_len]),
        ]
    )

    return {
        "plddt_binder": float(plddt_binder),
        "plddt_target": float(plddt_target),
        "pae_binder": float(pae_binder),
        "pae_target": float(pae_target),
        "ipae": float(ipae),
    }


@app.function(image=image, gpu=GPU, timeout=TIMEOUT)
def alphafold(
    fasta_name: str,
    fasta_str: str,
    models: list[int] = None,
    num_recycles: int = 3,
    binder_len: int = None,
    target_len: int = None,
    return_all_files: bool = False,
):
    import json
    import zipfile
    from colabfold.batch import get_queries, run
    from colabfold.download import default_data_dir

    if models is None:
        models = [1]

    in_dir = "/tmp/in_af"
    out_dir = "/tmp/out_af"
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(in_dir) / fasta_name, "w") as f:
        f.write(fasta_str)

    queries, is_complex = get_queries(in_dir)

    run(
        queries=queries,
        result_dir=out_dir,
        use_templates=False,
        num_relax=0,
        relax_max_iterations=200,
        msa_mode="MMseqs2 (UniRef+Environmental)",
        model_type="auto",
        num_models=len(models),
        num_recycles=num_recycles,
        model_order=models,
        is_complex=is_complex,
        data_dir=default_data_dir,
        keep_existing_results=False,
        rank_by="auto",
        pair_mode="unpaired+paired",
        stop_at_score=100,
        zip_results=True,
        user_agent="colabfold/google-colab-batch",
    )

    # --------------------------------------------------------------------------
    # If binder_len is supplied, evaluate binder-target score using iPAE
    #
    if binder_len is not None:
        results_zip = list(Path(out_dir).glob("**/*.zip"))
        assert len(results_zip) == 1, f"unexpected zip output: {results_zip}"

        with zipfile.ZipFile(results_zip[0], "a") as zip_ref:
            json_files = [f for f in zip_ref.namelist() if Path(f).suffix == ".json"]

            for json_file in json_files:
                json_data = json.loads(zip_ref.read(json_file))

                if "plddt" in json_data and "pae" in json_data:
                    prefix = json_file.split(".")[0]
                    af2m_scores = score_af2m_binding(json_data, binder_len, target_len)
                    scores_json = json.dumps(af2m_scores, indent=2)
                    zip_ref.writestr(f"{prefix}.af2m_scores.json", scores_json)

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*.*")
        if (return_all_files or Path(out_file).suffix == ".zip")
    ]


@app.local_entrypoint()
def main(
    input_fasta: str,
    models: str = "1",
    num_recycles: int = 1,
    binder_len: int = None,
    target_len: int = None,
    local_out: str = ".",
    return_all_files: bool = False,
):
    fasta_str = open(input_fasta).read()
    models = [int(model) for model in models.split(",")]

    outputs = alphafold.remote(
        fasta_name=Path(input_fasta).name,
        fasta_str=fasta_str,
        models=models,
        num_recycles=num_recycles,
        binder_len=binder_len,
        target_len=target_len,
        return_all_files=return_all_files,
    )

    for out_file, out_content in outputs:
        (Path(local_out) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open((Path(local_out) / Path(out_file)), "wb") as out:
                out.write(out_content)
