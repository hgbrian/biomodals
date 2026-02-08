"""Functional tests: run each Modal app with its quickest README example.

WARNING: These tests are SLOW, use GPU cloud resources, and COST MONEY.
They are excluded from the default test run.

Run all:    uv run run_tests.py tests/test_quick_runs.py
Run one:    uv run run_tests.py tests/test_quick_runs.py -k af2rank

Approximate run times (GPU):
    1-2 min       esm2, pdb2png, anarci, nextflow, minimap2
    2-5 min       af2rank, chai1, boltz, boltzgen, iggm, ligandmpnn
    5-10 min      alphafold, diffdock, md_protein_ligand, afdesign, rso
    10-30 min     germinal
    30-60 min     bindcraft
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _modal_run(*args, timeout=600, extra_withs=None):
    """Run a modal app and assert it succeeds."""
    cmd = ["uv", "run", "--with", "modal"]
    for w in (extra_withs or []):
        cmd += ["--with", w]
    cmd += ["modal", "run", *args]
    print(f"\n  CMD: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        f"FAILED (rc={result.returncode}):\n"
        f"stdout (last 1500):\n{result.stdout[-1500:]}\n"
        f"stderr (last 1500):\n{result.stderr[-1500:]}"
    )
    return result


def _download(url, filename):
    """Download a file to REPO_ROOT if it doesn't exist."""
    path = REPO_ROOT / filename
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _ensure_file(filename, content):
    """Write a small test input file if it doesn't exist."""
    path = REPO_ROOT / filename
    if not path.exists():
        path.write_text(content)
    return path


# ---- Working apps ----

def test_af2rank():
    _download("https://files.rcsb.org/download/1YWI.pdb", "1YWI.pdb")
    _modal_run("modal_af2rank.py", "--input-pdb", "1YWI.pdb", "--run-name", "test")


def test_minimap2():
    _download(
        "https://gist.githubusercontent.com/hgbrian/56787d9b3ce2e68f698ac94d537340d8/raw/mito.fasta",
        "mito.fasta",
    )
    _download(
        "https://gist.githubusercontent.com/hgbrian/802d8094bb4fed435bbb93a8c9092ee2/raw/mito_reads.fastq",
        "mito_reads.fastq",
    )
    _modal_run(
        "modal_minimap2.py",
        "--input-ref-fasta", "mito.fasta",
        "--input-reads-fastq", "mito_reads.fastq",
    )


def test_esm2():
    _ensure_file("test_esm2.faa", ">1\nMA<mask>GMT\n")
    _modal_run("modal_esm2_predict_masked.py", "--input-faa", "test_esm2.faa")


def test_pdb2png():
    _download("https://files.rcsb.org/download/1YWI.pdb", "1YWI.pdb")
    _modal_run(
        "modal_pdb2png.py",
        "--input-pdb", "1YWI.pdb",
        "--protein-zoom", "0.8",
        "--protein-color", "240,200,190",
    )


def test_anarci():
    _ensure_file(
        "test_anarci.faa",
        ">test_anarci\nDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLESGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT\n",
    )
    _modal_run("modal_anarci.py", "--input-fasta", "test_anarci.faa")


def test_nextflow():
    _modal_run("modal_nextflow_example.py")


def test_chai1():
    _ensure_file(
        "test_chai1.faa",
        ">protein|name=insulin\n"
        "MAWTPLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQGSGVPSRFSGSKDASTNAGLLLISG"
        "LQSEDEADYYCAIWYSSTS\n"
        ">RNA|name=rna\n"
        "ACUGACUGGAAGUCCCCGUAGUACCCGACG\n"
        ">ligand|name=caffeine\n"
        "N[C@@H](Cc1ccc(O)cc1)C(=O)O\n",
    )
    _modal_run("modal_chai1.py", "--input-faa", "test_chai1.faa")


def test_md_protein_ligand():
    pdb = _download("https://files.rcsb.org/download/5O45.pdb", "5O45.pdb")
    chainA = REPO_ROOT / "5O45_chainA.pdb"
    if not chainA.exists():
        lines = [l for l in pdb.read_text().splitlines(keepends=True) if l.startswith("ATOM") and l[21] == "A"]
        chainA.write_text("".join(lines))
    _modal_run("modal_md_protein_ligand.py", "--pdb-id", "5O45_chainA.pdb", timeout=900)


def test_boltz():
    _ensure_file(
        "test_boltz.yaml",
        "sequences:\n    - protein:\n        id: A\n        sequence: TDKLIFGKGTRVTVEP\n",
    )
    _modal_run(
        "modal_boltz.py", "--input-yaml", "test_boltz.yaml",
        "--params-str", "--seed 42",
    )


def test_boltzgen():
    _download(
        "https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13.cif",
        "1g13.cif",
    )
    _ensure_file(
        "1g13prot.yaml",
        "entities:\n  - protein:\n      id: C\n      sequence: 80..140\n  - file:\n      path: 1g13.cif\n      include:\n        - chain:\n            id: A\n",
    )
    _modal_run("modal_boltzgen.py", "--input-yaml", "1g13prot.yaml", "--num-designs", "1")


def test_iggm():
    _ensure_file(
        "test_iggm.faa",
        ">H\n"
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS\n"
        ">L\n"
        "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT\n"
        ">A\n",
    )
    _download("https://files.rcsb.org/download/5O45.pdb", "5O45.pdb")
    _modal_run(
        "modal_iggm.py",
        "--input-fasta", "test_iggm.faa",
        "--antigen", "5O45.pdb",
        "--epitope", "19,20,21",
    )


# ---- Slow apps (>10 min) ----

@pytest.mark.slow
def test_bindcraft():
    _download(
        "https://raw.githubusercontent.com/martinpacesa/BindCraft/refs/heads/main/example/PDL1.pdb",
        "PDL1.pdb",
    )
    _modal_run(
        "modal_bindcraft.py",
        "--input-pdb", "PDL1.pdb",
        "--number-of-final-designs", "1",
        timeout=3600,
    )


@pytest.mark.slow
def test_germinal():
    pdb = _download("https://files.rcsb.org/download/5O45.pdb", "5O45.pdb")
    chainA = REPO_ROOT / "5O45_chainA.pdb"
    if not chainA.exists():
        lines = [l for l in pdb.read_text().splitlines(keepends=True) if l.startswith("ATOM") and l[21] == "A"]
        chainA.write_text("".join(lines))
    _ensure_file(
        "target_example.yaml",
        'target_name: "5O45"\n'
        'target_pdb_path: "5O45_chainA.pdb"\n'
        'target_chain: "A"\n'
        'binder_chain: "B"\n'
        'target_hotspots: "A19,A20,A21,A22"\n'
        "length: 129\n",
    )
    _modal_run(
        "modal_germinal.py",
        "--target-yaml", "target_example.yaml",
        "--max-trajectories", "1",
        "--max-passing-designs", "1",
        timeout=3600,
        extra_withs=["PyYAML"],
    )


# ---- Medium apps (5-10 min) ----

def test_alphafold():
    _download("https://www.rcsb.org/fasta/entry/3NIT", "3NIT.faa")
    _modal_run("modal_alphafold.py", "--input-fasta", "3NIT.faa", timeout=900)


def test_boltz_msa():
    _ensure_file(
        "test_boltz.yaml",
        "sequences:\n    - protein:\n        id: A\n        sequence: TDKLIFGKGTRVTVEP\n",
    )
    _modal_run(
        "modal_boltz.py", "--input-yaml", "test_boltz.yaml",
        "--params-str", "--use_msa_server --seed 42",
    )


def test_diffdock():
    _download("https://files.rcsb.org/download/1IGY.pdb", "1IGY.pdb")
    _download(
        "https://gist.github.com/hgbrian/393ec799893cbf518f3084847c17cb2d/raw/1IGY_example.mol2",
        "1IGY_example.mol2",
    )
    _modal_run(
        "modal_diffdock.py",
        "--pdb-file", "1IGY.pdb",
        "--mol2-file", "1IGY_example.mol2",
    )


def test_ligandmpnn():
    _download("https://files.rcsb.org/download/1IVO.pdb", "1IVO.pdb")
    _modal_run(
        "modal_ligandmpnn.py",
        "--input-pdb", "1IVO.pdb",
        "--extract-chains", "AC",
        "--params-str", '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt" --chains_to_design "C" --save_stats 1',
    )


def test_afdesign():
    _modal_run(
        "modal_afdesign.py",
        "--pdb", "4MZK",
        "--target-chain", "A",
        "--soft-iters", "2",
        "--hard-iters", "2",
        "--binder-len", "6",
    )


def test_rso():
    pdb = _download("https://files.rcsb.org/download/5O45.pdb", "5O45.pdb")
    chainA = REPO_ROOT / "5O45_chainA.pdb"
    if not chainA.exists():
        lines = [l for l in pdb.read_text().splitlines(keepends=True) if l.startswith("ATOM") and l[21] == "A"]
        chainA.write_text("".join(lines))
    _modal_run(
        "modal_rso.py",
        "--input-pdb", "5O45_chainA.pdb",
        "--num-designs", "1",
        "--traj-iters", "10",
        "--binder-len", "30",
    )


def test_mber():
    _download("https://files.rcsb.org/download/7STF.pdb", "7STF.pdb")
    _modal_run("modal_mber.py", "--target-pdb", "7STF.pdb", "--target-name", "PDL1", timeout=900)
