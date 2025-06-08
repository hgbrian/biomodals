# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESM2 predict masked amino acid.

Input a fasta with format:
>1
MA<mask>GMT

Returns a tsv file of most probably amino acids.
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("GPU", None)
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

def download_model():
    import esm

    _model, _alphabet = esm.pretrained.esm2_t33_650M_UR50D()

image = (
    Image.micromamba(python_version="3.10")
    .apt_install(["git", "wget", "gcc", "g++", "libffi-dev"])
    .pip_install(
        ["torch==1.13.1+cu117"], index_url="https://download.pytorch.org/whl/cu117"
    )
    .pip_install(["fair-esm"])
    .pip_install(["pandas", "matplotlib"])
    .run_function(download_model, gpu=GPU)
)

app = App("esm2_predict_masked", image=image)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def esm2(fasta_name: str, fasta_str: str, make_figures: bool = False):
    from tempfile import TemporaryDirectory
    import torch
    import esm
    import matplotlib.pyplot as plt
    import pandas as pd

    with TemporaryDirectory() as td_out:

        # Load ESM-2 model
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disables dropout for deterministic results

        assert fasta_str.startswith(">"), f"{fasta_name} is not a fasta file"

        data = []
        for entry in fasta_str[1:].split("\n>"):
            label, _, seq = entry.partition("\n")
            seq = seq.replace("\n", "").strip()
            data.append((label, seq))

        _batch_labels, _batch_strs, batch_tokens = batch_converter(data)

        results_list = []
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)

        for i, (label, seq) in enumerate(data):
            # Find the position of the mask token for this sequence
            mask_position = (batch_tokens[i] == alphabet.mask_idx).nonzero(as_tuple=True)[
                0
            ][0]

            # Get logits for the masked position
            logits = results["logits"][i, mask_position]

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=0)

            # Get the top 5 predictions
            top_probs, top_indices = probs.topk(5)

            all_probs, all_indices = probs.sort(descending=True)
            for prob, idx in zip(all_probs, all_indices):
                aa = alphabet.get_tok(idx)
                results_list.append((i, label, aa, round(float(prob), 4)))

            # Get the best prediction
            best_prediction = alphabet.get_tok(top_indices[0])
            best_probability = top_probs[0].item()
            print(
                f"\nBest prediction for '{label}': {best_prediction} {best_probability}\n"
            )

            if make_figures:
                # Visualize the contact map
                plt.figure(figsize=(10, 10))
                plt.matshow(results["contacts"][i].cpu())
                plt.title(f"Contact Map for {label}")
                plt.colorbar()
                plt.savefig(f"{td_out}/{fasta_name}.contact_map_{label}.png")
                plt.close()

        df = pd.DataFrame(results_list, columns=["seq_n", "label", "aa", "prob"])
        df.to_csv(Path(td_out) / f"{fasta_name}.results.tsv", sep="\t", index=None)

        print(results_list)
        return [
            (out_file.relative_to(td_out), open(out_file, "rb").read())
            for out_file in Path(td_out).glob("**/*.*")
        ]


@app.local_entrypoint()
def main(
    input_faa: str,
    make_figures: bool = False,
    out_dir: str = "./out/esm2_predict_masked",
    run_name: str | None = None,
):
    from datetime import datetime

    fasta_str = open(input_faa).read()

    outputs = esm2.remote(Path(input_faa).name, fasta_str, make_figures)

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]

    for out_file, out_content in outputs:
        output_path = Path(out_dir) / (run_name or today) / Path(out_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(output_path, "wb") as out:
                out.write(out_content)
