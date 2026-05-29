# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""ESMFold2 (Biohub) structure prediction.

https://github.com/Biohub/esm  https://huggingface.co/biohub/ESMFold2

Single-sequence (or multi-chain complex) folding with the Biohub ESMFold2
model. No API token required; weights are baked into the image at build time.

Input is a FASTA. Header type tags `protein|`, `dna|`, `rna|`, `ligand|`
are honored when present; otherwise sequences are treated as protein.
Ligand sequences are interpreted as SMILES.

```
printf '>protein|name=insulin\\nGIVEQCCTSICSLYQLENYCN\\n' > test_esmfold2.faa
modal run modal_esmfold2.py --input-faa test_esmfold2.faa

# Multi-entity complex with a ligand
printf '>protein|A\\nMKTAYIAKQRQISFVKSHFSRQ\\n>ligand|B\\nN[C@@H](Cc1ccc(O)cc1)C(=O)O\\n' > complex.faa
modal run modal_esmfold2.py --input-faa complex.faa --num-diffusion-samples 3
```
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("MODAL_GPU", "A100-40GB")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", 30))

ESMFOLD2_HF_REPO = "biohub/ESMFold2"
ESMFOLD2_HF_REVISION = "1afea82e432079d9af2ebd71d1e4c339ecca2ff0"
ESMFOLD2_GIT_REF = "c94ed8d"
ESMFOLD2_CACHE_DIR = "/root/.cache/huggingface"


def _download_models():
    """Pre-download ESMFold2 weights into the image cache at build time."""
    from huggingface_hub import snapshot_download

    Path(ESMFOLD2_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=ESMFOLD2_HF_REPO,
        revision=ESMFOLD2_HF_REVISION,
        cache_dir=ESMFOLD2_CACHE_DIR,
        allow_patterns=["*.safetensors", "*.bin", "*.json", "*.pkl", "*.txt", "*.model"],
    )


image = (
    Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "HF_HOME": ESMFOLD2_CACHE_DIR,
        "HF_HUB_CACHE": ESMFOLD2_CACHE_DIR,
    })
    .uv_pip_install(
        f"esm @ git+https://github.com/Biohub/esm.git@{ESMFOLD2_GIT_REF}",
        "xformers",
        "huggingface_hub",
    )
    .run_function(_download_models)
)

app = App("esmfold2", image=image)


def _fasta_iter(s: str):
    """Yield (header, sequence) tuples from a FASTA string."""
    from io import StringIO
    from itertools import groupby

    with StringIO(s) as fh:
        groups = (x[1] for x in groupby(fh, lambda line: line.startswith(">")))
        for header_lines in groups:
            header = next(header_lines)[1:].strip()
            seq = "".join(s.strip() for s in next(groups))
            yield header, seq


def _fasta_to_input(fasta_str: str):
    """Build a StructurePredictionInput from a FASTA string."""
    from esm.models.esmfold2 import (
        DNAInput,
        LigandInput,
        ProteinInput,
        StructurePredictionInput,
    )
    try:
        from esm.models.esmfold2 import RNAInput
    except ImportError:
        RNAInput = None

    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    sequences: list = []
    for n, (seq_id, seq) in enumerate(_fasta_iter(fasta_str)):
        first = seq_id.split("|")[0].lower() if "|" in seq_id else ""
        cid = chain_letters[n] if n < len(chain_letters) else f"chain_{n}"
        if first == "ligand":
            sequences.append(LigandInput(id=cid, smiles=seq))
        elif first == "dna":
            sequences.append(DNAInput(id=cid, sequence=seq))
        elif first == "rna":
            if RNAInput is None:
                raise ValueError("RNA inputs not supported by this esm version")
            sequences.append(RNAInput(id=cid, sequence=seq))
        else:
            sequences.append(ProteinInput(id=cid, sequence=seq))
    return StructurePredictionInput(sequences=sequences)


@app.function(timeout=TIMEOUT * 60, gpu=GPU)
def esmfold2(
    fasta_name: str,
    fasta_str: str,
    seed: int = 42,
    num_diffusion_samples: int = 1,
    num_sampling_steps: int = 50,
    num_loops: int = 3,
) -> list[tuple[Path, bytes]]:
    """Fold a FASTA with ESMFold2; return per-sample CIF + scores JSON."""
    import json
    import time

    from esm.models.esmfold2 import ESMFold2InputBuilder
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    t0 = time.time()
    model = ESMFold2Model.from_pretrained(
        ESMFOLD2_HF_REPO,
        revision=ESMFOLD2_HF_REVISION,
        local_files_only=True,
    ).cuda().eval()
    print(f"[esmfold2] Model loaded in {time.time() - t0:.1f}s")

    spi = _fasta_to_input(fasta_str)
    print(f"[esmfold2] {len(spi.sequences)} entities, samples={num_diffusion_samples}, steps={num_sampling_steps}")

    t0 = time.time()
    result = ESMFold2InputBuilder().fold(
        model, spi,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
        complex_id=fasta_name,
    )
    print(f"[esmfold2] fold() done in {time.time() - t0:.1f}s")

    samples = result if isinstance(result, list) else [result]

    def _to_py(v):
        if v is None:
            return None
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): _to_py(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_py(x) for x in v]
        return v

    outputs: list[tuple[Path, bytes]] = []
    for idx, sample in enumerate(samples):
        complex_obj = getattr(sample, "complex", sample)
        cif_str = complex_obj.to_mmcif()
        plddt_list = _to_py(complex_obj.plddt) or []
        mean_plddt = sum(plddt_list) / len(plddt_list) if plddt_list else 0.0
        scores = {
            "plddt": mean_plddt,
            "plddt_per_token": plddt_list,
            "num_tokens": len(plddt_list),
            "sample_index": idx,
            "ptm": _to_py(getattr(sample, "ptm", None)),
            "iptm": _to_py(getattr(sample, "iptm", None)),
            "chain_pair_iptm": _to_py(getattr(sample, "pair_chains_iptm", None)),
        }
        outputs.append((Path(f"{fasta_name}_sample_{idx}.cif"), cif_str.encode()))
        outputs.append((Path(f"{fasta_name}_sample_{idx}_scores.json"),
                        json.dumps(scores, indent=2).encode()))

    return outputs


@app.local_entrypoint()
def main(
    input_faa: str,
    seed: int = 42,
    num_diffusion_samples: int = 1,
    num_sampling_steps: int = 50,
    num_loops: int = 3,
    out_dir: str = "./out/esmfold2",
    run_name: str | None = None,
):
    from datetime import datetime

    fasta_str = Path(input_faa).read_text()
    fasta_name = Path(input_faa).stem

    outputs = esmfold2.remote(
        fasta_name=fasta_name,
        fasta_str=fasta_str,
        seed=seed,
        num_diffusion_samples=num_diffusion_samples,
        num_sampling_steps=num_sampling_steps,
        num_loops=num_loops,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    for out_file, out_content in outputs:
        target = out_dir_full / out_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(out_content)

    print(f"Results saved to: {out_dir_full}")
