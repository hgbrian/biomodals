# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs AlphaFold2 or AF2-multimer predictions using ColabFold on Modal.

Limitations:
- It requires only one entry in a fasta file.
- If providing a complex, e.g., a binder and target pair,
  Provide the target first, then N binders after, separated by ":"
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", "A10G")
TIMEOUT = os.environ.get("TIMEOUT", 20)

image = (
    Image.micromamba(python_version="3.11")
    .apt_install("wget", "git")
    .pip_install(
        "colabfold[alphafold-minus-jax]@git+https://github.com/sokrypton/ColabFold@a134f6a8f8de5c41c63cb874d07e1a334cb021bb"
    )
    .micromamba_install(
        "kalign2=2.04", "hhsuite=3.3.0", channels=["conda-forge", "bioconda"]
    )
    .run_commands(
        'pip install --upgrade "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
        gpu="a10g",
    )
    .run_commands("python -m colabfold.download")
)

app = App("alphafold", image=image)


def score_af2m_binding(
    af2m_dict: dict, target_len: int, binders_len: list[int]
) -> dict:
    """Calculates binding scores from AlphaFold2 multimer prediction results.

    The target is assumed to be the first part of the sequence, followed by one or more binders.

    Args:
        af2m_dict (dict): Dictionary loaded from an AlphaFold2 multimer JSON output file (usually contains 'plddt' and 'pae' keys).
        target_len (int): Length of the target protein sequence.
        binders_len (list[int]): List of lengths for each binder protein sequence.

    Returns:
        dict: A dictionary containing various scores:
            - "plddt_binder" (dict[int, float]): Average pLDDT for each binder, keyed by binder index (0-based).
            - "plddt_target" (float): Average pLDDT for the target.
            - "pae_binder" (dict[int, float]): Average PAE within each binder, keyed by binder index.
            - "pae_target" (float): Average PAE within the target.
            - "ipae" (dict[int, float]): Average interface PAE between the target and each binder, keyed by binder index.
            - "ipae_binder" (dict[int, list[float]]): Per-residue interface PAE scores for each binder interacting with the target, keyed by binder index.
    """

    import numpy as np

    plddt_array = np.array(af2m_dict["plddt"])
    pae_array = np.array(af2m_dict["pae"])

    assert len(plddt_array) == len(pae_array) == target_len + sum(binders_len)

    plddt_target = np.mean(plddt_array[:target_len])
    pae_target = np.mean(pae_array[:target_len, :target_len])

    plddt_binder = {}
    pae_binder = {}
    ipae = {}
    ipae_binder = {}

    current_pos = target_len
    for binder_n, binder_len in enumerate(binders_len):
        binder_start, binder_end = current_pos, current_pos + binder_len

        # --------------------------------------------------------------------------
        # pLDDT; binder
        #

        plddt_binder[binder_n] = np.mean(plddt_array[binder_start:binder_end])

        # --------------------------------------------------------------------------
        # PAE; binder vs itself; mean target<>binder; target<>binder separately
        #
        pae_binder[binder_n] = np.mean(
            pae_array[binder_start:binder_end, binder_start:binder_end]
        )
        ipae[binder_n] = np.mean(
            [
                np.mean(pae_array[:target_len, binder_start:binder_end]),
                np.mean(pae_array[binder_start:binder_end, :target_len]),
            ]
        )

        ipae_binder[binder_n] = np.mean(
            [
                np.mean(pae_array[:target_len, binder_start:binder_end], axis=0),
                np.mean(pae_array[binder_start:binder_end, :target_len], axis=1),
            ],
            axis=0,
        )
        current_pos += binder_len

    return {
        "plddt_binder": {k: float(v) for k, v in plddt_binder.items()},
        "plddt_target": float(plddt_target),
        "pae_binder": {k: float(v) for k, v in pae_binder.items()},
        "pae_target": float(pae_target),
        "ipae": {k: float(v) for k, v in ipae.items()},
        "ipae_binder": {
            k: [float(ipae_b) for ipae_b in ipae_binder[k]]
            for k, v in ipae_binder.items()
        },
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT * 60,
)
def alphafold(
    fasta_name: str,
    fasta_str: str,
    models: list[int] | None = None,
    num_recycles: int = 3,
    num_relax: int = 0,
    use_templates: bool = False,
    use_precomputed_msas: bool = False,
    return_all_files: bool = False,
):
    """Runs AlphaFold2/ColabFold prediction on Modal.

    Args:
        fasta_name (str): Name of the FASTA file (e.g., "protein.fasta").
        fasta_str (str): Content of the FASTA file as a string.
        models (list[int], optional): List of model numbers to run (1-5). Defaults to [1].
        num_recycles (int, optional): Number of recycles for the model. Defaults to 3.
        num_relax (int, optional): Number of relaxation steps (0 means no Amber relaxation,
                                   1 means relax top model). Defaults to 0.
        use_templates (bool, optional): Whether to use PDB templates during prediction. Defaults to False.
        use_precomputed_msas (bool, optional): If True, attempts to copy MSAs from a mounted
                                               directory ("/msas") into the output directory to reuse them.
                                               Defaults to False.
        return_all_files (bool, optional): If True, returns all generated files. If False,
                                           only returns the main ZIP file containing predictions.
                                           Defaults to False.

    Returns:
        list[tuple[Path, bytes]]: A list of tuples, where each tuple contains the relative output
                                  file path (typically a zip file or specific requested files)
                                  and its byte content.
    """
    import json
    import subprocess
    import zipfile
    from colabfold.batch import get_queries, run
    from colabfold.download import default_data_dir

    if models is None:
        models = [1]

    in_dir = "/tmp/in_af"
    out_dir = "/tmp/out_af"
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # saves the colabfold server, speeds things up
    if use_precomputed_msas:
        subprocess.run(f"cp -r /msas/* {out_dir}", shell=True)

    with open(Path(in_dir) / fasta_name, "w") as f:
        f.write(fasta_str)

    header = fasta_str.splitlines()[0]
    fasta_seq = "".join(seq.strip() for seq in fasta_str.splitlines()[1:])
    if header[0] != ">" or any(aa not in "ACDEFGHIKLMNPQRSTVWY:" for aa in fasta_seq):
        raise AssertionError(f"invalid fasta:\n{fasta_str}")

    queries, is_complex = get_queries(in_dir)

    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    run(
        queries=queries,
        result_dir=out_dir,
        use_templates=use_templates,
        num_relax=num_relax,
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
    if ":" in fasta_seq:  # then it is a multimer
        target_len = len(fasta_seq.split(":")[0])
        binders_len = [len(b_seq) for b_seq in fasta_seq.split(":")[1:]]

        results_zip = list(Path(out_dir).glob("**/*.zip"))
        assert len(results_zip) == 1, f"unexpected zip output: {results_zip}"

        with zipfile.ZipFile(results_zip[0], "a") as zip_ref:
            json_files = [f for f in zip_ref.namelist() if Path(f).suffix == ".json"]

            for json_file in json_files:
                json_data = json.loads(zip_ref.read(json_file))

                if "plddt" in json_data and "pae" in json_data:
                    prefix = Path(json_file).with_suffix("")
                    af2m_scores = score_af2m_binding(json_data, target_len, binders_len)
                    scores_json = json.dumps(af2m_scores, indent=2)
                    zip_ref.writestr(f"{prefix}.af2m_scores.json", scores_json)
                    break

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if (return_all_files or Path(out_file).suffix == ".zip")
        if Path(out_file).is_file()
    ]


@app.local_entrypoint()
def main(
    input_fasta: str,
    models: str | None = None,
    num_recycles: int = 1,
    num_relax: int = 0,
    out_dir: str = "./out/alphafold",
    use_templates: bool = False,
    use_precomputed_msas: bool = False,
    return_all_files: bool = False,
    run_name: str | None = None,
):
    """Local entrypoint for running AlphaFold2 predictions.

    This function prepares the inputs, calls the remote `alphafold` Modal function,
    and saves the output files locally.

    Args:
        input_fasta (str): Path to the input FASTA file.
        models (list[int], optional): List of AlphaFold2 model numbers to run (1-5).
                                      Can be a comma-separated string if passed via CLI.
                                      Defaults to [1].
        num_recycles (int, optional): Number of recycles for the model. Defaults to 1.
        num_relax (int, optional): Number of Amber relaxation steps (0 for none, 1 for top model).
                                   Defaults to 0.
        out_dir (str, optional): Directory to save the output files. Defaults to ".".
        use_templates (bool, optional): Whether to use PDB templates. Defaults to False.
        use_precomputed_msas (bool, optional): Whether to use precomputed MSAs. Defaults to False.
        return_all_files (bool, optional): Whether to return all generated files from the remote
                                           function or just the primary zip. Defaults to False.

    Returns:
        None
    """
    from datetime import datetime

    fasta_str = open(input_fasta).read()
    if isinstance(models, str):
        models = [int(model) for model in models.split(",")]
    elif models is None:
        models = [1]

    outputs = alphafold.remote(
        fasta_name=Path(input_fasta).name,
        fasta_str=fasta_str,
        models=models,
        num_recycles=num_recycles,
        num_relax=num_relax,
        use_templates=use_templates,
        use_precomputed_msas=use_precomputed_msas,
        return_all_files=return_all_files,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        (Path(out_dir_full) / Path(out_file)).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open((Path(out_dir_full) / Path(out_file)), "wb") as out:
                out.write(out_content)
