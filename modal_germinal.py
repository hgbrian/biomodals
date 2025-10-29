"""
Germinal: Efficient generation of epitope-targeted de novo antibodies
https://github.com/SantiagoMille/germinal/

Germinal is a pipeline for designing de novo antibodies against specified epitopes on target proteins.
The pipeline follows a 3-step process: hallucination based on ColabDesign, selective sequence redesign
with AbMPNN, and cofolding with a structure prediction model.

## Setup and minimal test:
```bash
# Get the PD-L1 structure from RCSB PDB
curl -O https://files.rcsb.org/download/5O45.pdb

# Extract chain A only
grep "^ATOM.*\ A\ " 5O45.pdb > 5O45_chainA.pdb

# Create target configuration file
cat > target_example.yaml << 'EOF'
target_name: "5O45"
target_pdb_path: "5O45_chainA.pdb"
target_chain: "A"
binder_chain: "B"
target_hotspots: "A19,A20,A21,A22"
length: 129
EOF

# Run minimal test (1 trajectory, 1 passing design)
uvx --with PyYAML modal run modal_germinal.py --target-yaml target_example.yaml --max-trajectories 1 --max-passing-designs 1
```

## Production usage:
```bash
# Basic VHH design against PD-L1
modal run modal_germinal.py --target-yaml target.yaml

# scFv design with custom parameters
modal run modal_germinal.py --target-yaml target.yaml --run-type scfv --max-trajectories 500
```
"""

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "H100")
TIMEOUT = int(os.environ.get("TIMEOUT", 180))  # 3 hours default for antibody design

REQUIRED_TARGET_COLS = [
    "target_name",
    "target_pdb_path",
    "target_chain",
    "binder_chain",
    "target_hotspots",
    "length",
]

GERMINAL_VOLUME_NAME = "germinal-models"
GERMINAL_MODEL_VOLUME = Volume.from_name(GERMINAL_VOLUME_NAME, create_if_missing=True)

# Cache directories for models
AF_PARAMS_DIR = f"/{GERMINAL_VOLUME_NAME}/alphafold_params"


def download_models():
    """Download required models and parameters for Germinal"""
    import subprocess

    Path(AF_PARAMS_DIR).mkdir(parents=True, exist_ok=True)

    # Download AlphaFold-Multimer parameters if not present
    if not Path(AF_PARAMS_DIR + "/params_model_1_multimer_v3.npz").exists():
        print("Downloading AlphaFold-Multimer parameters...")
        subprocess.run(
            [
                "wget",
                "-q",
                "-O",
                f"{AF_PARAMS_DIR}/alphafold_params_2022-12-06.tar",
                "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar",
            ],
            check=True,
        )

        subprocess.run(
            [
                "tar",
                "-xf",
                f"{AF_PARAMS_DIR}/alphafold_params_2022-12-06.tar",
                "-C",
                AF_PARAMS_DIR,
            ],
            check=True,
        )


def patch_germinal_code():
    """Patch Germinal code to fix bugs and add features"""
    import re

    # Patch 1: Add entropy logging to design.py
    design_py = Path("/tmp/germinal/germinal/design/design.py")
    content = design_py.read_text()
    content = content.replace(
        "str(softmax_ipae),",
        'str(softmax_ipae), " / entropy:", str(af_model._tmp["best"]["mean_soft_pseudo"]), f"(threshold: {seq_entropy_threshold})",',
    )
    design_py.write_text(content)
    print("✓ Patched design.py: added entropy logging")

    # Patch 2: Fix PDB headers for DSSP in utils.py
    utils_py = Path("/tmp/germinal/germinal/utils/utils.py")
    content = utils_py.read_text()
    # Find def get_dssp and insert header fix code before it
    pattern = r"(def get_dssp\([^)]*\):)"
    header_fix = """    # Fix PDB headers for DSSP
    from pathlib import Path as _Path
    if _Path(pdb_file).exists():
        with open(pdb_file) as _f: _content = _f.read()
        if not _content.startswith("HEADER"):
            _h = ["HEADER    PROTEIN                                 26-SEP-25   GERM", "TITLE     GERMINAL ANTIBODY DESIGN", "COMPND    MOL_ID: 1;", "COMPND   2 MOLECULE: ANTIBODY;", "SOURCE    MOL_ID: 1;", "SOURCE   2 ORGANISM_SCIENTIFIC: SYNTHETIC;"]
            with open(pdb_file, "w") as _f: _f.write("\\n".join(_h) + "\\n" + _content)

"""
    content = re.sub(pattern, header_fix + r"\1", content)
    utils_py.write_text(content)
    print("✓ Patched utils.py: added PDB header fix for DSSP")

    # Patch 3: Wrap PyRosetta filter calls in try-except
    # Match the code block that calls filter_utils.run_filters with final_filters
    # This is the code that can throw PyRosetta DAlphaBall geometry errors
    run_germinal_py = Path("/tmp/germinal/run_germinal.py")
    content = run_germinal_py.read_text()

    # Pattern: Find the filter_metrics assignment through utils.clear_memory
    # that uses final_filters (not initial_filters)
    # Match the whole lines with their full indentation
    pattern = re.compile(
        r"^(                filter_metrics, filter_results, accepted, final_struct = \(\n"
        r"                    filter_utils\.run_filters\(\n"
        r"                        [^\n]+,\n"  # mpnn_trajectory
        r"                        [^\n]+,\n"  # run_settings
        r"                        [^\n]+,\n"  # target_settings
        r"                        final_filters,\n"  # Key: this is final_filters not initial
        r".*?"  # Everything in between
        r"^                utils\.clear_memory\(clear_jax=False\)\n)",
        re.MULTILINE | re.DOTALL,
    )

    match = pattern.search(content)
    if match:
        original_block = match.group(1)

        # Step 1: Add 4 spaces to the beginning of every line (like sed does)
        indented_lines = []
        for line in original_block.split("\n"):
            if line:  # Add 4 spaces to non-empty lines
                indented_lines.append("    " + line)
            else:  # Keep empty lines empty
                indented_lines.append(line)
        indented_block = "\n".join(indented_lines)

        # Step 2: Wrap with try-except
        # try: has 16 spaces (same level as original filter_metrics line)
        # indented_block now has 20 spaces (16 + 4)
        # except and its contents use 16 spaces base
        wrapped = (
            "                try:\n"  # 16 spaces
            f"{indented_block}\n"  # Now has 20 spaces on first line
            "                except RuntimeError as e:\n"  # 16 spaces
            '                    if "surf_vol.cc" in str(e) or "DALPHABALL" in str(e):\n'  # 20 spaces
            '                        print(f"WARNING: Skipping filters due to DAlphaBall geometry bug (known PyRosetta issue)")\n'  # 24 spaces
            "                        continue\n"  # 24 spaces
            "                    else:\n"  # 20 spaces
            "                        raise\n"  # 24 spaces
        )

        content = content.replace(original_block, wrapped)
        run_germinal_py.write_text(content)
        print("✓ Patched run_germinal.py: added try-except for PyRosetta geometry bugs")
    else:
        print("⚠ Warning: Could not find filter_utils.run_filters pattern to patch")


image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "wget",
        "git",
        "gcc",
        "g++",
        "build-essential",
        "curl",
        "aria2",
        "ffmpeg",
        "procps",
        "zlib1g-dev",
        "libhdf5-dev",
        "pkg-config",
        "clang",
        "llvm",  # Add this
        "llvm-dev",  # Add this
        "libc++-dev",  # Add this
        "libc++abi-dev",  # Add this
        "dssp",  # Add DSSP for protein secondary structure analysis
    )
    .pip_install(
        [
            "polars==1.19.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "biopython>=1.79",
            "torch>=2.0.0",
            "dm-haiku==0.0.10",
            "chex>=0.1.7",
            "optax>=0.2.4",
            "flax>=0.7.0",
            "iglm",
            "colabfold[alphafold]",
            "scipy==1.10.1",
        ]
    )
    .run_commands(
        "mkdir -p /tools/llvm/bin",
        "ln -sf /usr/bin/llvm-ar /tools/llvm/bin/llvm-ar",
        "ln -sf /usr/bin/llvm-nm /tools/llvm/bin/llvm-nm",
        "ln -sf /usr/bin/llvm-ranlib /tools/llvm/bin/llvm-ranlib",
    )
    .pip_install("mdtraj==1.9.9")
    .run_commands(
        "pip install 'jax[cuda12_pip]==0.5.3' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    .pip_install(
        "https://west.rosettacommons.org/pyrosetta/release/release/PyRosetta4.Release.python310.ubuntu.wheel/pyrosetta-2025.37+release.df75a9c48e-cp310-cp310-linux_x86_64.whl"
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign"
    )
    .run_commands(
        "git clone https://github.com/SantiagoMille/germinal.git /tmp/germinal",
        "cd /tmp/germinal && git checkout 88d7f85aeb78684b05f872ec524255535ad15106",
    )
    .pip_install("cvxopt==1.3.2")
    .run_commands("cd /tmp/germinal && pip install -e .")
    .run_function(patch_germinal_code)
    .pip_install("git+https://github.com/chaidiscovery/chai-lab.git")
    .pip_install("dm-haiku==0.0.13")
    .pip_install("joblib==1.5.2")
    .run_commands(
        """find /usr/local/lib/python3.10/site-packages/haiku -name "*.py" -exec sed -i 's/jax\\.linear_util/jax._src.linear_util/g' {} +"""
    )
    .run_commands(
        "pip install 'jax[cuda12_pip]==0.5.3' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    .run_function(
        download_models,
        gpu="A100",
        volumes={f"/{GERMINAL_VOLUME_NAME}": GERMINAL_MODEL_VOLUME},
    )
)

app = App("germinal", image=image)


@app.function(
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={f"/{GERMINAL_VOLUME_NAME}": GERMINAL_MODEL_VOLUME},
)
def run_germinal_design(
    target_yaml_content: str,
    pdb_content: bytes,
    run_type: str = "vhh",
    max_trajectories: int = 100,
    max_passing_designs: int = 10,
    experiment_name: str = "germinal_design",
    original_pdb_path: str = "pdbs/target.pdb",
) -> dict:
    """
    Run Germinal antibody design pipeline

    Args:
        target_yaml_content: Target YAML file content as string
        pdb_content: PDB file content as bytes
        run_type: "vhh" or "scfv"
        max_trajectories: Maximum number of design trajectories
        max_passing_designs: Maximum number of passing designs to return
        experiment_name: Name for the experiment
        original_pdb_path: Original PDB path from the YAML (to replace with absolute)

    Returns:
        Dictionary containing design results and metrics
    """
    import tempfile
    import subprocess
    import shutil
    import yaml
    from pathlib import Path

    # Setup temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Parse YAML to get target_pdb_path
        target_data = yaml.safe_load(target_yaml_content)

        # Write PDB file
        pdb_path = temp_path / target_data["target_pdb_path"]
        pdb_path.parent.mkdir(parents=True, exist_ok=True)
        pdb_path.write_bytes(pdb_content)

        # Create pdbs directory and copy reference structures from Germinal
        pdbs_dir = temp_path / "pdbs"
        pdbs_dir.mkdir(exist_ok=True)

        # Copy required reference structures from Germinal
        germinal_pdbs_path = Path("/tmp/germinal/pdbs")
        for ref_pdb in ["nb.pdb", "scfv.pdb"]:
            src_pdb = germinal_pdbs_path / ref_pdb
            dst_pdb = pdbs_dir / ref_pdb
            if src_pdb.exists():
                shutil.copy2(src_pdb, dst_pdb)
                print(f"Copied reference structure: {ref_pdb}")
            else:
                print(f"Warning: Reference structure not found: {ref_pdb}")

        # Create config directories first
        (temp_path / "configs").mkdir(parents=True, exist_ok=True)
        (temp_path / "configs" / "run").mkdir(parents=True, exist_ok=True)
        (temp_path / "configs" / "target").mkdir(parents=True, exist_ok=True)
        (temp_path / "configs" / "filter" / "initial").mkdir(
            parents=True, exist_ok=True
        )
        (temp_path / "configs" / "filter" / "final").mkdir(parents=True, exist_ok=True)

        # Copy and modify the original Germinal main config
        germinal_configs_path = Path("/tmp/germinal/configs")
        original_config_path = germinal_configs_path / "config.yaml"
        main_config_path = temp_path / "configs" / "config.yaml"
        shutil.copy2(original_config_path, main_config_path)

        # Read and modify the config to update paths and parameters
        with open(main_config_path, "r") as f:
            content = f.read()

        # Update the config with our parameters
        content = content.replace('project_dir: "."', f'project_dir: "{temp_path}"')
        content = content.replace(
            "max_trajectories: 10000", f"max_trajectories: {max_trajectories}"
        )
        content = content.replace(
            "max_passing_designs: 100", f"max_passing_designs: {max_passing_designs}"
        )
        content = content.replace(
            'experiment_name: "germinal_run"', f'experiment_name: "{experiment_name}"'
        )
        content = content.replace(
            'af_params_dir: ""', f'af_params_dir: "{AF_PARAMS_DIR}"'
        )
        content = content.replace(
            'dssp_path: "params/dssp"', 'dssp_path: "/tmp/germinal/params/dssp"'
        )
        content = content.replace(
            'dalphaball_path: "params/DAlphaBall.gcc"',
            'dalphaball_path: "/tmp/germinal/params/DAlphaBall.gcc"',
        )

        with open(main_config_path, "w") as f:
            f.write(content)

        # Copy actual Germinal config files instead of creating our own

        # Copy run config from Germinal
        shutil.copy2(
            germinal_configs_path / "run" / f"{run_type.lower()}.yaml",
            temp_path / "configs" / "run" / f"{run_type.lower()}.yaml",
        )

        # Use "pdl1" as target name to avoid Hydra config issues
        target_config_final_path = temp_path / "configs" / "target" / "pdl1.yaml"

        # Get the absolute PDB path
        absolute_pdb_path = str((temp_path / target_data["target_pdb_path"]).resolve())

        # Write the user's YAML directly, just replace the PDB path with absolute path
        updated_yaml_content = target_yaml_content.replace(
            original_pdb_path, absolute_pdb_path
        )

        with open(target_config_final_path, "w") as f:
            f.write(updated_yaml_content)

        target_name = "pdl1"

        # Copy filter configs from Germinal and remove @package annotations
        initial_filter_src = germinal_configs_path / "filter" / "initial" / "vhh.yaml"
        initial_filter_dst = temp_path / "configs" / "filter" / "initial" / "vhh.yaml"
        shutil.copy2(initial_filter_src, initial_filter_dst)

        final_filter_src = germinal_configs_path / "filter" / "final" / "vhh.yaml"
        final_filter_dst = temp_path / "configs" / "filter" / "final" / "vhh.yaml"
        shutil.copy2(final_filter_src, final_filter_dst)

        # Run Germinal using subprocess (the proper way)
        try:
            import subprocess
            import os

            # Change to temp directory
            original_dir = os.getcwd()
            os.chdir(temp_path)

            print("Running Germinal via subprocess...")

            # First check JAX device availability
            print("Checking JAX GPU setup...")
            subprocess.run("nvidia-smi")
            jax_check = subprocess.run(
                [
                    "python",
                    "-c",
                    "import jax; print(f'JAX devices: {jax.devices()} {jax.default_backend()}'); print(f'JAX GPU available: {len(jax.devices()) > 0 and jax.devices()[0].device_kind == \"gpu\"}')",
                ],
                capture_output=True,
                text=True,
            )
            print(f"JAX check output: {jax_check.stdout}")
            if jax_check.stderr:
                print(f"JAX check stderr: {jax_check.stderr}")

            # Run Germinal (patched at image build time)
            cmd = [
                "python",
                "/tmp/germinal/run_germinal.py",
                "--config-path",
                str(temp_path / "configs"),
                "--config-name",
                "config",
                f"run={run_type}",
                f"target={target_name}",
                f"max_trajectories={max_trajectories}",
                f"max_passing_designs={max_passing_designs}",
                f"experiment_name={experiment_name}",
                f"af_params_dir={AF_PARAMS_DIR}",
                "dalphaball_path=/tmp/germinal/params/DAlphaBall.gcc",
                "dssp_path=/tmp/germinal/params/dssp",
            ]

            # Set environment variable for detailed Hydra errors
            env = os.environ.copy()
            env["HYDRA_FULL_ERROR"] = "1"

            print(f"Command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=TIMEOUT * 60, env=env
            )

            os.chdir(original_dir)

            print(f"Return code: {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")

            if result.returncode != 0:
                return {
                    "error": f"Germinal failed with return code {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "experiment_name": experiment_name,
                    "target_name": target_data["target_name"],
                    "status": "failed",
                }

            # Look for results in the expected output directory
            results_dir = temp_path / "results" / experiment_name
            if results_dir.exists():
                # Parse results from CSV files
                import pandas as pd

                all_trajectories_file = results_dir / "all_trajectories.csv"
                accepted_file = results_dir / "accepted" / "designs.csv"

                results_data = {
                    "experiment_name": experiment_name,
                    "run_type": run_type,
                    "target_name": target_data["target_name"],
                    "status": "completed",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "results_directory": str(results_dir),
                    "output_files": {},
                }

                # Copy all files from results directory
                import os

                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(results_dir)
                        try:
                            # Read all files as text
                            with open(file_path, "r") as f:
                                results_data["output_files"][str(relative_path)] = f.read()
                        except Exception as e:
                            results_data["output_files"][str(relative_path)] = (
                                f"<error reading file: {e}>"
                            )

                # Try to parse CSV files if they exist and have content
                if all_trajectories_file.exists():
                    try:
                        df_all = pd.read_csv(all_trajectories_file)
                        results_data["total_trajectories"] = len(df_all)
                        results_data["all_trajectories"] = df_all.to_dict("records")
                    except pd.errors.EmptyDataError:
                        results_data["total_trajectories"] = 0
                        results_data["all_trajectories"] = []
                        results_data["csv_parse_error"] = (
                            "all_trajectories.csv is empty"
                        )

                if accepted_file.exists():
                    try:
                        df_accepted = pd.read_csv(accepted_file)
                        results_data["passing_designs"] = len(df_accepted)
                        results_data["accepted_designs"] = df_accepted.to_dict(
                            "records"
                        )
                    except pd.errors.EmptyDataError:
                        results_data["passing_designs"] = 0
                        results_data["accepted_designs"] = []
                        results_data["csv_parse_error"] = "designs.csv is empty"
                else:
                    results_data["passing_designs"] = 0
                    results_data["accepted_designs"] = []

                return results_data
            else:
                return {
                    "error": "Results directory not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "experiment_name": experiment_name,
                    "target_name": target_data["target_name"],
                    "status": "failed",
                }

        except subprocess.TimeoutExpired:
            return {
                "error": f"Germinal timed out after {TIMEOUT} minutes",
                "experiment_name": experiment_name,
                "target_name": target_data["target_name"],
                "status": "timeout",
            }
        except Exception as e:
            return {
                "error": str(e),
                "experiment_name": experiment_name,
                "target_name": target_data["target_name"],
                "status": "failed",
            }
        finally:
            if "original_dir" in locals():
                os.chdir(original_dir)


@app.local_entrypoint()
def main(
    target_yaml: str,
    run_type: str = "vhh",
    max_trajectories: int = 100,
    max_passing_designs: int = 10,
    experiment_name: str = "germinal_design",
    run_name: str | None = None,
    out_dir: str = "./out/germinal",
):
    """
    Run Germinal antibody design locally

    Args:
        target_yaml: Path to target configuration YAML file
        run_type: Type of antibody design ("vhh" or "scfv")
        max_trajectories: Maximum number of design trajectories
        max_passing_designs: Maximum number of passing designs
        experiment_name: Name for the experiment
        run_name: Optional name for the run directory
        out_dir: Output directory base path
    """
    import json
    import yaml
    from pathlib import Path
    from datetime import datetime

    # Load target configuration
    target_yaml_path = Path(target_yaml)
    if not target_yaml_path.exists():
        raise FileNotFoundError(f"Target YAML file not found: {target_yaml}")

    # Read YAML content and parse it
    target_yaml_content = target_yaml_path.read_text()
    target_data = yaml.safe_load(target_yaml_content)

    # Validate required fields
    missing_fields = [
        field for field in REQUIRED_TARGET_COLS if field not in target_data
    ]
    if missing_fields:
        raise ValueError(f"Missing required target fields: {missing_fields}")

    # Load PDB file
    original_pdb_path = target_data["target_pdb_path"]
    pdb_path = Path(original_pdb_path)
    if not pdb_path.is_absolute():
        pdb_path = target_yaml_path.parent / pdb_path

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    pdb_content = pdb_path.read_bytes()

    print(
        f"Running Germinal {run_type.upper()} design for target: {target_data['target_name']}"
    )
    print(f"Max trajectories: {max_trajectories}")
    print(f"Max passing designs: {max_passing_designs}")

    # Run design
    results = run_germinal_design.remote(
        target_yaml_content=target_yaml_content,
        pdb_content=pdb_content,
        run_type=run_type,
        max_trajectories=max_trajectories,
        max_passing_designs=max_passing_designs,
        experiment_name=experiment_name,
        original_pdb_path=original_pdb_path,
    )

    # Set up output directory
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)
    out_dir_full.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    output_path = out_dir_full / "germinal_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Save output files if they exist
    if "output_files" in results and results["output_files"]:
        print("Saving output files:")
        for file_path, content in results["output_files"].items():
            local_file_path = out_dir_full / file_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            if content.startswith("<"):  # Error markers
                print(f"  - {file_path}: {content}")
            else:
                with open(local_file_path, "w") as f:
                    f.write(content)
                print(f"  - {local_file_path}")

    if "error" in results:
        print(f"Design failed: {results['error']}")
        return

    print(f"Design completed successfully!")
    print(f"Total trajectories: {results['total_trajectories']}")
    print(f"Passing designs: {results['passing_designs']}")
    if "success_rate" in results:
        print(f"Success rate: {results['success_rate']:.2%}")
    else:
        total = results.get("total_trajectories", 0)
        passing = results.get("passing_designs", 0)
        if total > 0:
            print(f"Success rate: {passing / total:.2%}")
        else:
            print(f"Success rate: N/A (no trajectories completed)")
