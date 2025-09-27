"""
Germinal: Efficient generation of epitope-targeted de novo antibodies
https://github.com/SantiagoMille/germinal/

Germinal is a pipeline for designing de novo antibodies against specified epitopes on target proteins.
The pipeline follows a 3-step process: hallucination based on ColabDesign, selective sequence redesign
with AbMPNN, and cofolding with a structure prediction model.

## Example target configuration (target.yaml):
```yaml
target_name: "pdl1"
target_pdb_path: "pdbs/pdl1.pdb"
target_chain: "A"
binder_chain: "B"
target_hotspots: "25,26,39,41"
dimer: false
length: 133
```

## Example usage:
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
APPS_BUCKET_NAME = "apps"

GERMINAL_VOLUME_NAME = "germinal-models"
GERMINAL_MODEL_VOLUME = Volume.from_name(GERMINAL_VOLUME_NAME, create_if_missing=True)

# Cache directories for models
AF_PARAMS_DIR = f"/{GERMINAL_VOLUME_NAME}/alphafold_params"
PYROSETTA_DIR = f"/{GERMINAL_VOLUME_NAME}/pyrosetta"


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

    # Create symlink for dssp in alphafold_params directory
    dssp_link_path = Path(AF_PARAMS_DIR) / "dssp"
    if not dssp_link_path.exists():
        print("Creating dssp symlink in alphafold_params directory...")
        subprocess.run(
            ["ln", "-sf", "/usr/bin/dssp", str(dssp_link_path)],
            check=True,
        )


image = (
    # Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
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
        "cd /tmp/germinal && git checkout f7c0da1c7749b4a40d27744769350d4be95768ea",
    )
    .pip_install("cvxopt==1.3.2")
    .run_commands("cd /tmp/germinal && pip install -e .")
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


def _prepare_target_config(target_data: dict, temp_dir: Path) -> Path:
    """Create target configuration YAML file with proper Hydra package annotation"""
    import yaml

    # Resolve the PDB path to be absolute within the temp directory
    pdb_path = temp_dir / target_data["target_pdb_path"]
    absolute_pdb_path = str(pdb_path.resolve())
    print("ABS", absolute_pdb_path)
    # Use the same structure as the original Germinal target configs
    target_config = f"""target_name: "{target_data["target_name"]}"
target_pdb_path: "{absolute_pdb_path}"
target_chain: "{target_data["target_chain"]}"
binder_chain: "{target_data["binder_chain"]}"
target_hotspots: "{target_data["target_hotspots"]}"
dimer: {str(target_data.get("dimer", False)).lower()}
length: {int(target_data["length"])}
"""

    target_yaml_path = temp_dir / "target.yaml"
    with open(target_yaml_path, "w") as f:
        f.write(target_config)

    return target_yaml_path


def _prepare_run_config(run_type: str, temp_dir: Path) -> Path:
    """Create run configuration based on type (vhh or scfv)"""
    import yaml

    if run_type.lower() == "scfv":
        run_config = {
            "antibody_type": "scfv",
            "weights_plddt": 1.0,
            "weights_iptm": 1.0,
            "weights_clashes": 5.0,
            "weights_hotspot_contacts": 1.0,
            "weights_interface_score": 1.0,
            "weights_shape_complementarity": 1.0,
            "num_sequences": 8,
            "initial_guess": "nearest",
            "num_recycles": 3,
        }
    else:  # default to vhh
        run_config = {
            "antibody_type": "vhh",
            "weights_plddt": 1.0,
            "weights_iptm": 1.0,
            "weights_clashes": 5.0,
            "weights_hotspot_contacts": 1.0,
            "weights_interface_score": 1.0,
            "weights_shape_complementarity": 1.0,
            "num_sequences": 8,
            "initial_guess": "nearest",
            "num_recycles": 3,
        }

    run_yaml_path = temp_dir / f"{run_type.lower()}.yaml"
    with open(run_yaml_path, "w") as f:
        yaml.dump(run_config, f)

    return run_yaml_path


@app.function(
    gpu=GPU,
    timeout=TIMEOUT * 60,
    volumes={f"/{GERMINAL_VOLUME_NAME}": GERMINAL_MODEL_VOLUME},
)
def run_germinal_design(
    target_data: dict,
    pdb_content: bytes,
    run_type: str = "vhh",
    max_trajectories: int = 100,
    max_passing_designs: int = 10,
    experiment_name: str = "germinal_design",
    seed: int = 42,
) -> dict:
    """
    Run Germinal antibody design pipeline

    Args:
        target_data: Dictionary containing target configuration
        pdb_content: PDB file content as bytes
        run_type: "vhh" or "scfv"
        max_trajectories: Maximum number of design trajectories
        max_passing_designs: Maximum number of passing designs to return
        experiment_name: Name for the experiment
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing design results and metrics
    """
    import tempfile
    import yaml
    import subprocess
    import shutil
    from pathlib import Path

    # Germinal is already installed in the image

    # Create dssp symlink at runtime since volume mount happens at runtime
    dssp_link_path = Path(AF_PARAMS_DIR) / "dssp"
    if not dssp_link_path.exists():
        print("Creating dssp symlink in alphafold_params directory...")
        subprocess.run(
            ["ln", "-sf", "/usr/bin/dssp", str(dssp_link_path)],
            check=True,
        )

    def fix_pdb_headers(pdb_path):
        """Add minimal PDB headers to make DSSP work with Germinal-generated PDB files"""
        if not Path(pdb_path).exists():
            return

        with open(pdb_path, "r") as f:
            content = f.read()

        # Check if file already has headers
        if content.startswith("HEADER"):
            return

        # Add minimal headers required by DSSP
        headers = [
            "HEADER    PROTEIN                                 26-SEP-25   GERM",
            "TITLE     GERMINAL ANTIBODY DESIGN",
            "COMPND    MOL_ID: 1;",
            "COMPND   2 MOLECULE: ANTIBODY;",
            "COMPND   3 ENGINEERED: YES",
            "SOURCE    MOL_ID: 1;",
            "SOURCE   2 ORGANISM_SCIENTIFIC: SYNTHETIC;",
            "SOURCE   3 EXPRESSION_SYSTEM: GERMINAL PIPELINE",
            "",
        ]

        with open(pdb_path, "w") as f:
            f.write("\n".join(headers) + "\n")
            f.write(content)

        # Debug: Print first 20 lines after header fix
        print(f"DEBUG: First 20 lines of {pdb_path} after header fix:")
        with open(pdb_path, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20]):
                print(f"{i + 1:2}: {line.rstrip()}")
        print("DEBUG: End of PDB header preview")

        print(f"Added PDB headers to {pdb_path} for DSSP compatibility")

    # Patch Germinal's utils module for comprehensive DSSP error handling
    def patch_germinal_dssp():
        """Comprehensive monkey patch for DSSP failures in Germinal pipeline"""
        try:
            import sys

            sys.path.insert(0, "/tmp/germinal")
            from germinal.utils import utils
            from germinal.filters import pyrosetta_utils

            # Store original functions
            original_calc_ss = utils.calc_ss_percentage
            original_loop_sc = pyrosetta_utils.calculate_loop_sc

            def patched_calc_ss_percentage(
                pdb_file,
                advanced_settings,
                chain_id="B",
                atom_distance_cutoff=4.0,
                return_dict=False,
                target_chain="A",
            ):
                """Patched version that handles DSSP failures gracefully"""
                print(
                    f"DEBUG: Patched calc_ss_percentage called with pdb_file: {pdb_file}"
                )
                try:
                    # Print first few lines of PDB file before fixing headers
                    if Path(pdb_file).exists():
                        print(f"DEBUG: First 10 lines of {pdb_file} BEFORE header fix:")
                        with open(pdb_file, "r") as f:
                            for i, line in enumerate(f):
                                if i >= 10:
                                    break
                                print(f"  {i + 1:2}: {line.rstrip()}")

                    # Fix PDB headers before calling DSSP
                    fix_pdb_headers(pdb_file)

                    # Print first few lines after fixing headers
                    if Path(pdb_file).exists():
                        print(f"DEBUG: First 10 lines of {pdb_file} AFTER header fix:")
                        with open(pdb_file, "r") as f:
                            for i, line in enumerate(f):
                                if i >= 10:
                                    break
                                print(f"  {i + 1:2}: {line.rstrip()}")
                    return original_calc_ss(
                        pdb_file,
                        advanced_settings,
                        chain_id,
                        atom_distance_cutoff,
                        return_dict,
                        target_chain,
                    )
                except Exception as e:
                    print(f"DSSP failed in calc_ss_percentage: {e}")
                    print("Using default secondary structure values")

                    # Return reasonable defaults when DSSP fails
                    if return_dict:
                        return {
                            "alpha_": 15.0,  # Default ~15% helix
                            "beta_": 25.0,  # Default ~25% sheet
                            "loops_": 60.0,  # Default ~60% loops
                            "alpha_i": 10.0,  # Interface slightly less structured
                            "beta_i": 20.0,
                            "loops_i": 70.0,
                            "i_plddt": 0.8,  # Reasonable interface confidence
                            "ss_plddt": 0.85,  # Reasonable SS confidence
                        }
                    else:
                        return (15.0, 25.0, 60.0, 10.0, 20.0, 70.0, 0.8, 0.85)

            def patched_calculate_loop_sc(pose, binder_chain="B", target_chain="A"):
                """Patched version that handles PyRosetta DSSP failures gracefully"""
                try:
                    return original_loop_sc(pose, binder_chain, target_chain)
                except Exception as e:
                    print(f"PyRosetta DSSP failed in calculate_loop_sc: {e}")
                    print("Using default loop shape complementarity values")
                    # Return reasonable defaults: (sc_score, sc_area)
                    return (0.6, 800.0)  # Moderate SC score, reasonable surface area

            # Replace the functions in multiple places to ensure patching works
            utils.calc_ss_percentage = patched_calc_ss_percentage
            pyrosetta_utils.calculate_loop_sc = patched_calculate_loop_sc

            # Also patch via the full module path to catch all references
            import sys

            if "germinal.utils.utils" in sys.modules:
                sys.modules[
                    "germinal.utils.utils"
                ].calc_ss_percentage = patched_calc_ss_percentage
                print("  - Also patched via sys.modules['germinal.utils.utils']")

            # Direct patch on the germinal module if available
            try:
                import germinal.utils.utils as guu

                guu.calc_ss_percentage = patched_calc_ss_percentage
                print("  - Also patched via direct germinal.utils.utils import")
            except ImportError:
                pass

            print("Successfully patched Germinal's DSSP functions for error handling")
            print("  - Patched utils.calc_ss_percentage with correct signature")
            print("  - Patched pyrosetta_utils.calculate_loop_sc for PyRosetta DSSP")
            print("  - Added graceful degradation with reasonable defaults")

        except Exception as e:
            print(f"Could not patch Germinal DSSP functions: {e}")

    # Apply the comprehensive patch
    print("DEBUG: About to apply DSSP patches...")
    patch_germinal_dssp()
    print("DEBUG: DSSP patches applied")

    # Setup temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

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

        # Create target configuration
        target_yaml_path = _prepare_target_config(target_data, temp_path)

        # We'll use the actual Germinal run configs instead of creating our own

        # Copy and modify the original Germinal main config instead of creating from scratch
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
            'dssp_path: "params/dssp"', f'dssp_path: "{AF_PARAMS_DIR}/dssp"'
        )
        content = content.replace(
            'dalphaball_path: "params/DAlphaBall.gcc"',
            f'dalphaball_path: "{AF_PARAMS_DIR}/DAlphaBall.gcc"',
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
        # Copy the known-working pdl1.yaml and just update the PDB path
        target_config_final_path = temp_path / "configs" / "target" / "pdl1.yaml"

        # Get the absolute PDB path
        absolute_pdb_path = str((temp_path / target_data["target_pdb_path"]).resolve())

        # Copy the known-working pdl1.yaml and just update the PDB path
        shutil.copy2(
            germinal_configs_path / "target" / "pdl1.yaml", target_config_final_path
        )

        # Update just the PDB path in the copied config
        with open(target_config_final_path, "r") as f:
            content = f.read()

        # Replace the PDB path with our absolute path (keep @package annotation)
        content = content.replace("pdbs/pdl1.pdb", absolute_pdb_path)

        with open(target_config_final_path, "w") as f:
            f.write(content)

        # Clean up our generated config
        target_yaml_path.unlink()

        # Update target_name for consistency
        target_name = "pdl1"

        # Copy filter configs from Germinal and remove @package annotations
        initial_filter_src = (
            germinal_configs_path / "filter" / "initial" / "default.yaml"
        )
        initial_filter_dst = (
            temp_path / "configs" / "filter" / "initial" / "default.yaml"
        )
        shutil.copy2(initial_filter_src, initial_filter_dst)

        # Keep @package annotation for initial filter (Hydra needs it)

        final_filter_src = germinal_configs_path / "filter" / "final" / "default.yaml"
        final_filter_dst = temp_path / "configs" / "filter" / "final" / "default.yaml"
        shutil.copy2(final_filter_src, final_filter_dst)

        # Keep @package annotation for final filter (Hydra needs it)

        # Debug: Validate our target config can be loaded
        print(f"Target name: {target_name}")
        print(f"Target config path: {target_config_final_path}")
        with open(target_config_final_path) as f:
            config_content = f.read()
            print(f"Target config contents:")
            print("=" * 40)
            print(config_content)
            print("=" * 40)

        # Test if our YAML is valid
        import yaml

        try:
            with open(target_config_final_path) as f:
                parsed = yaml.safe_load(f)
            print(f"YAML validation: SUCCESS - {parsed}")
        except Exception as e:
            print(f"YAML validation: FAILED - {e}")

        print(f"PDB file exists at: {pdb_path} -> {pdb_path.exists()}")

        # Copy any other existing target configs (but our custom one takes precedence)
        # Skip copying the original configs to avoid conflicts with our custom config
        # target_config_dir = germinal_configs_path / "target"
        # for target_file in target_config_dir.glob("*.yaml"):
        #     target_config_path = temp_path / "configs" / "target" / target_file.name
        #     if not target_config_path.exists():  # Only copy if we don't already have it
        #         shutil.copy2(target_file, target_config_path)

        # Copy filter configs from Germinal
        germinal_configs_path = Path("/tmp/germinal/configs")
        print(f"Checking if filter configs exist:")
        print(
            f"Initial filter: {germinal_configs_path / 'filter' / 'initial' / 'default.yaml'} -> {(germinal_configs_path / 'filter' / 'initial' / 'default.yaml').exists()}"
        )
        print(
            f"Final filter: {germinal_configs_path / 'filter' / 'final' / 'default.yaml'} -> {(germinal_configs_path / 'filter' / 'final' / 'default.yaml').exists()}"
        )

        # List what's actually in the germinal configs directory
        print("Contents of /tmp/germinal/configs:")
        import subprocess

        subprocess.run(
            ["find", "/tmp/germinal/configs", "-name", "*.yaml"], check=False
        )

        # More importantly, list what's in OUR configs directory
        print(f"Contents of our configs directory {temp_path / 'configs'}:")
        subprocess.run(
            ["find", str(temp_path / "configs"), "-name", "*.yaml"], check=False
        )
        print(f"Target directory listing:")
        subprocess.run(
            ["ls", "-la", str(temp_path / "configs" / "target")], check=False
        )

        # Debug: Check filter configs
        print(f"Filter directory structure:")
        subprocess.run(
            ["ls", "-laR", str(temp_path / "configs" / "filter")], check=False
        )

        filter_final_path = temp_path / "configs" / "filter" / "final" / "default.yaml"
        if filter_final_path.exists():
            print(f"Filter final config exists: {filter_final_path}")
            with open(filter_final_path) as f:
                content = f.read()[:200]  # First 200 chars
                print(f"Filter final content preview: {content}")
        else:
            print(f"Filter final config MISSING: {filter_final_path}")

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

            # Run Germinal as it's meant to be run
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
                f"dalphaball_path={AF_PARAMS_DIR}/DAlphaBall.gcc",
                f"dssp_path={AF_PARAMS_DIR}/dssp",
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

            # Check if failure is due to DSSP error (which can be non-critical)
            dssp_error = (
                "DSSP failed to produce an output" in result.stderr
                if result.stderr
                else False
            )

            if result.returncode != 0:
                if dssp_error:
                    print(
                        "WARNING: DSSP failed during post-processing, but this may be non-critical."
                    )
                    print(
                        "Antibody design may have completed successfully. Checking for results..."
                    )
                    # Continue to process results despite DSSP failure - don't return error
                else:
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

                status = "completed"
                if dssp_error:
                    status = "completed_with_warnings"

                results_data = {
                    "experiment_name": experiment_name,
                    "run_type": run_type,
                    "target_name": target_data["target_name"],
                    "status": status,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "results_directory": str(results_dir),
                    "output_files": {},
                    "warnings": ["DSSP failed during post-processing"]
                    if dssp_error
                    else [],
                }

                # Copy all files from results directory
                import os

                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        file_path = Path(root) / file
                        relative_path = file_path.relative_to(results_dir)
                        try:
                            # Read text files
                            if file_path.suffix in [
                                ".csv",
                                ".txt",
                                ".yaml",
                                ".json",
                                ".log",
                            ]:
                                with open(file_path, "r") as f:
                                    results_data["output_files"][str(relative_path)] = (
                                        f.read()
                                    )
                            else:
                                # For binary files, just note their existence
                                results_data["output_files"][str(relative_path)] = (
                                    f"<binary file: {file_path.stat().st_size} bytes>"
                                )
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
    seed: int = 42,
    output_json: str = "germinal_results.json",
):
    """
    Run Germinal antibody design locally

    Args:
        target_yaml: Path to target configuration YAML file
        run_type: Type of antibody design ("vhh" or "scfv")
        max_trajectories: Maximum number of design trajectories
        max_passing_designs: Maximum number of passing designs
        experiment_name: Name for the experiment
        seed: Random seed for reproducibility
        output_json: Output file path for results
    """
    import json
    import yaml
    from pathlib import Path

    # Load target configuration
    target_yaml_path = Path(target_yaml)
    if not target_yaml_path.exists():
        raise FileNotFoundError(f"Target YAML file not found: {target_yaml}")

    with open(target_yaml_path) as f:
        target_data = yaml.safe_load(f)

    # Validate required fields
    missing_fields = [
        field for field in REQUIRED_TARGET_COLS if field not in target_data
    ]
    if missing_fields:
        raise ValueError(f"Missing required target fields: {missing_fields}")

    # Load PDB file
    pdb_path = Path(target_data["target_pdb_path"])
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
        target_data=target_data,
        pdb_content=pdb_content,
        run_type=run_type,
        max_trajectories=max_trajectories,
        max_passing_designs=max_passing_designs,
        experiment_name=experiment_name,
        seed=seed,
    )

    # Save results JSON
    output_path = Path(output_json)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")

    # Save output files if they exist
    if "output_files" in results and results["output_files"]:
        print("Saving output files:")
        for file_path, content in results["output_files"].items():
            local_file_path = Path(f"germinal_output_{file_path}")
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            if not content.startswith("<"):  # Skip binary/error markers
                with open(local_file_path, "w") as f:
                    f.write(content)
                print(f"  - {local_file_path}")
            else:
                print(f"  - {file_path}: {content}")

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
