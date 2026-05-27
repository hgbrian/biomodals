# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Rosetta energy scoring via tmol (GPU-accelerated beta_nov2016).

https://github.com/uw-ipd/tmol

Scores PDB structures with the tmol implementation of the Rosetta beta_nov2016
energy function on GPU. By default runs an L-BFGS Cartesian relax before scoring.
Optionally repacks side chains with FASPR first (requires `modal_faspr.py` deployed:
`modal deploy modal_faspr.py`).

```
# Score a single PDB (Cartesian relax on, FASPR repack on)
modal run modal_tmol.py --input-pdb structure.pdb

# Score without relax / without FASPR
modal run modal_tmol.py --input-pdb structure.pdb --no-relax --no-faspr

# Score every PDB in a directory
modal run modal_tmol.py --input-dir ./pdbs/
```

Output: results.tsv (one row per structure, columns: key, total_energy,
pre_relax_energy, elapsed_ms, plus weighted per-term energies).
"""

import os
from pathlib import Path

from modal import App, Image

GPU = os.environ.get("MODAL_GPU", "a10g")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", 10))

TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.22/"
    "tmol-0.1.22%2Bcu12torch2.10cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

image = (
    Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(TMOL_WHEEL)
)

app = App("tmol", image=image)


def _score_pose(sfxn, scorer, pose_stack) -> tuple[float, dict[str, float]]:
    """Return (total_energy, {term_name: weighted_energy})."""
    total_energy = scorer(pose_stack.coords).item()
    terms = {}
    try:
        all_st = sfxn.all_score_types()
        wt = sfxn.weights_tensor()
        unweighted = scorer.unweighted_scores(pose_stack.coords)
        for i, st in enumerate(all_st):
            w = wt[i].item()
            if w != 0:
                terms[st.name] = unweighted[i, 0].item() * w
    except Exception:
        pass
    return total_energy, terms


@app.function(timeout=TIMEOUT * 60, max_containers=100, gpu=GPU)
def score_structure(param: dict) -> dict:
    """Score a PDB with tmol beta_nov2016.

    `param` keys: key, pdb_text, relax (bool), relax_steps (int), faspr (bool).
    """
    import tempfile
    import time

    import tmol

    pdb_text = param["pdb_text"]
    do_relax = param.get("relax", True)
    relax_steps = param.get("relax_steps", 10)
    do_faspr = param.get("faspr", True)

    if do_faspr:
        try:
            import modal

            faspr_fn = modal.Function.from_name("faspr", "faspr")
            outputs = faspr_fn.remote(input_pdb_str=pdb_text)
            pdb_text = outputs[0][1].decode()
        except Exception as e:
            print(f"FASPR failed for {param.get('key', '?')}, scoring without repacking: {e}")

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        f.write(pdb_text)
        pdb_path = f.name

    try:
        import torch

        t0 = time.perf_counter()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose_stack = tmol.pose_stack_from_pdb(pdb_path, device=device)
        sfxn = tmol.beta2016_score_function(device)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

        pre_energy = None
        if do_relax:
            pre_energy, _ = _score_pose(sfxn, scorer, pose_stack)

            torch.set_grad_enabled(True)
            pose_stack.coords.requires_grad_(True)
            optimizer = torch.optim.LBFGS(
                [pose_stack.coords], lr=0.1, max_iter=20, line_search_fn="strong_wolfe",
            )
            for _ in range(relax_steps):
                def closure():
                    optimizer.zero_grad()
                    E = scorer(pose_stack.coords)
                    E.backward()
                    return E
                optimizer.step(closure)
            pose_stack.coords.requires_grad_(False)
            scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

        total_energy, terms = _score_pose(sfxn, scorer, pose_stack)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = {"key": param["key"], "total_energy": total_energy, "elapsed_ms": elapsed_ms}
        if pre_energy is not None:
            result["pre_relax_energy"] = pre_energy
        result.update(terms)
        return result
    except Exception as e:
        return {"key": param["key"], "error": str(e)}


@app.local_entrypoint()
def main(
    input_pdb: str | None = None,
    input_dir: str | None = None,
    relax: bool = True,
    relax_steps: int = 10,
    faspr: bool = True,
    out_dir: str = "./out/tmol",
    run_name: str | None = None,
):
    from datetime import datetime

    if not input_pdb and not input_dir:
        raise ValueError("Provide --input-pdb or --input-dir")

    pdb_files: list[Path] = []
    if input_pdb:
        pdb_files.append(Path(input_pdb))
    if input_dir:
        pdb_files.extend(sorted(Path(input_dir).glob("*.pdb")))
    if not pdb_files:
        raise ValueError("No PDB files found")

    params = [
        {
            "key": p.stem,
            "pdb_text": p.read_text(),
            "relax": relax,
            "relax_steps": relax_steps,
            "faspr": faspr,
        }
        for p in pdb_files
    ]

    print(f"Scoring {len(params)} structure(s)...")
    results = list(score_structure.map(params))

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    # Collect all columns across results
    all_cols: list[str] = ["key", "total_energy", "pre_relax_energy", "elapsed_ms"]
    seen = set(all_cols)
    for r in results:
        for k in r:
            if k not in seen and k not in ("error",):
                all_cols.append(k)
                seen.add(k)

    tsv_path = out_dir_full / "results.tsv"
    with tsv_path.open("w") as f:
        f.write("\t".join(all_cols) + "\n")
        for r in results:
            if "error" in r:
                print(f"  FAILED {r['key']}: {r['error']}")
                continue
            row = [str(r.get(c, "")) for c in all_cols]
            f.write("\t".join(row) + "\n")

    n_ok = sum(1 for r in results if "error" not in r)
    print(f"\nWrote {tsv_path}")
    print(f"Done: {n_ok}/{len(params)} succeeded")
