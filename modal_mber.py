"""MBER (Manifold Binder Engineering and Refinement) for VHH nanobody design

mBER is an open-source protein design framework for antibody binder design that leverages
structure templates and sequence conditioning through AlphaFold-Multimer.

Example usage:
```
# Design VHH against a PDB target
modal run modal_mber.py --target-pdb 7stf.pdb --target-name PDL1 --output-dir ./out/mber

# Design VHH with custom masked sequence (* = positions to design)
modal run modal_mber.py --target-pdb target.pdb --target-name MyTarget \
    --masked-binder-seq "EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS" \
    --output-dir ./out/mber

# Design against specific chains and hotspots
modal run modal_mber.py --target-pdb target.pdb --target-name MyTarget \
    --chains A,B --target-hotspot-residues A110,B120

# Multi-chain with non-contiguous numbering (prevents folding as single chain)
modal run modal_mber.py --target-pdb target.pdb --target-name MyTarget \
    --chains A,B --chain-offsets B:200 --target-hotspot-residues A6
```

Key parameters:
- masked_binder_seq: VHH sequence with * for positions to design (default: designs CDR1, CDR2, CDR3)
- chains: Which PDB chains to process (default: "A", use "A,B" for multi-chain if needed)
- target_hotspot_residues: Specific residues to target (e.g., "A110,A120")
- chain_offsets: Renumber chains to create breaks (e.g., "B:200" makes chain B start at residue 200)

References:
- GitHub: https://github.com/manifoldbio/mber-open
- Preprint: https://www.biorxiv.org/content/10.1101/2025.09.26.678877v1
"""

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "A100")
TIMEOUT = int(os.environ.get("TIMEOUT", 120))  # 2 hours default

# Use existing germinal-models volume that has AlphaFold2 weights
GERMINAL_VOLUME_NAME = "germinal-models"
GERMINAL_MODEL_VOLUME = Volume.from_name(GERMINAL_VOLUME_NAME, create_if_missing=True)


def download_nanobody_models():
    """Download NanoBodyBuilder2 models to default location.

    Models are saved to ~/.mber/nbb2_weights (the same location used at runtime).
    """
    import os
    from ImmuneBuilder import NanoBodyBuilder2

    print("Downloading NanoBodyBuilder2 models...")

    # Use the same default location that will be used at runtime
    weights_dir = os.path.expanduser("~/.mber/nbb2_weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Download models to the default directory
    _ = NanoBodyBuilder2(model_ids=[1, 2, 3, 4], weights_dir=weights_dir)
    print(f"NanoBodyBuilder2 models downloaded successfully to {weights_dir}")


image = (
    Image.micromamba(python_version="3.11")
    .apt_install("git", "wget", "tar", "build-essential", "gcc", "g++")
    .micromamba_install(
        "openmm==8.0.0",
        "pdbfixer==1.9",
        "anarci",
        "ffmpeg",
        channels=["conda-forge", "bioconda"],
    )
    .run_commands(
        "git clone https://github.com/manifoldbio/mber-open.git /tmp/mber-open",
        # Patch numpy indexing issue in plm_utils.py
        "sed -i 's/flex_pos = template_data\\.get_flex_pos(as_array=True) - 1/flex_pos = (template_data.get_flex_pos(as_array=True) - 1).astype(int)/g' /tmp/mber-open/src/mber/utils/plm_utils.py",
        # Disable animation generation in trajectory.py (comment out the line)
        "sed -i 's/design_state.trajectory_data.animated_trajectory = self.af_model.animate()/# design_state.trajectory_data.animated_trajectory = self.af_model.animate()  # Disabled to save time/g' /tmp/mber-open/src/mber/core/modules/trajectory.py",
        # Patch template module to skip fetching if template_pdb is already set
        "sed -i 's/def _process_target(self, design_state: DesignState) -> DesignState:/def _process_target(self, design_state: DesignState) -> DesignState:\\n        if design_state.template_data.template_pdb:\\n            self._log(\"Using provided PDB content, skipping fetch\")\\n            return design_state/g' /tmp/mber-open/src/mber/core/modules/template.py",
        # Add missing fields to BinderData class
        "sed -i '/esm_score: float = None/a\\    hbond: float = None\\n    salt_bridge: float = None\\n    ptm_energy: float = None' /tmp/mber-open/src/mber/core/data/state.py",
        # Patch evaluation module to save hbond, salt_bridge, ptm_energy
        """sed -i '/relax_rmsd=relax_data\\["rmsd"\\],/a\\            hbond=evaluation_metrics.get("hbond"),\\n            salt_bridge=evaluation_metrics.get("salt_bridge"),\\n            ptm_energy=evaluation_metrics.get("ptm_energy"),' /tmp/mber-open/src/mber/core/modules/evaluation.py""",
        # Add debug output to truncation to see what's happening with chains
        """sed -i '/for chain_id, mask in inclusion_masks.items():/a\\            print(f"DEBUG truncation: Processing chain {chain_id}, mask sum={sum(mask)}/{len(mask)}", flush=True)' /tmp/mber-open/src/mber/core/truncation.py""",
        # Fix chain ordering to be deterministic - sort chains alphabetically
        """sed -i 's/for chain_id, mask in inclusion_masks.items():/for chain_id, mask in sorted(inclusion_masks.items()):/g' /tmp/mber-open/src/mber/core/truncation.py""",
        # Debug hotspot processing
        """sed -i '/for hotspot in hotspots:/a\\            print(f"DEBUG: Checking hotspot {hotspot.chain}{hotspot.residue}", flush=True)' /tmp/mber-open/src/mber/core/truncation.py""",
        # Debug region chains
        """sed -i '/self.region_chains = list/a\\        print(f"DEBUG: Region chains: {self.region_chains}", flush=True)' /tmp/mber-open/src/mber/core/truncation.py""",
        # Debug optimize_truncation results
        """sed -i '/results = {}/a\\        print(f"DEBUG optimize_truncation: Processing {len(self.region_chains)} chains", flush=True)' /tmp/mber-open/src/mber/core/truncation.py""",
        """sed -i '/results\\[chain_id\\] = (inclusion_mask, F\\[pos\\]\\[current_state\\])/a\\            print(f"DEBUG optimize_truncation: Chain {chain_id} - kept {sum(inclusion_mask)}/{len(inclusion_mask)} residues", flush=True)' /tmp/mber-open/src/mber/core/truncation.py""",
        # Install JAX and pin numpy<2.0 for openmm compatibility
        "pip install jax[cuda12]==0.5.2 'numpy<2.0'",
        # Install PyTorch from CUDA 12.8 index
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
        # Then install mBER (requirements.txt will use already-installed numpy)
        "cd /tmp/mber-open && pip install -e .",
        "cd /tmp/mber-open/protocols && pip install -e .",
    )
    .pip_install("boto3==1.40.42", "prody==2.6.1", "numpy<2.0")
    # Download NanoBodyBuilder2 models during image build (v5 - use default ~/.mber/nbb2_weights)
    .run_function(download_nanobody_models, gpu="T4")
)

app = App("mber", image=image)


# Default VHH sequence with CDR regions masked for design (use * for positions to design)
DEFAULT_VHH_MASKED = "EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"


def renumber_pdb_chain(pdb_content: bytes, chain_id: str, offset: int) -> bytes:
    """Renumber residues in a specific chain by adding an offset.

    Args:
        pdb_content: PDB file content as bytes
        chain_id: Chain ID to renumber
        offset: Offset to add to residue numbers

    Returns:
        Modified PDB content with renumbered chain
    """
    lines = pdb_content.decode("utf-8").split("\n")
    result = []

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            # PDB format: chain is at position 21, resnum is at positions 22-26
            if len(line) > 26 and line[21] == chain_id:
                try:
                    old_resnum = int(line[22:26].strip())
                    new_resnum = old_resnum + offset
                    # Reconstruct line with new residue number
                    new_line = line[:22] + f"{new_resnum:>4}" + line[26:]
                    result.append(new_line)
                except ValueError:
                    result.append(line)
            else:
                result.append(line)
        else:
            result.append(line)

    return "\n".join(result).encode("utf-8")


def build_region_string(pdb_content: bytes, chains: str) -> str:
    """Build region specification string from PDB content and chain list.

    Args:
        pdb_content: PDB file content as bytes
        chains: Comma-separated list of chain IDs (e.g., "A,B")

    Returns:
        Region string in format "A:1-10,B:1-170"
    """
    import prody as pr
    from io import StringIO

    pdb_str = pdb_content.decode("utf-8")
    chains_list = [c.strip() for c in chains.split(",")]

    structure = pr.parsePDBStream(StringIO(pdb_str))
    region_parts = []

    for chain_id in chains_list:
        chain = structure.select(f"chain {chain_id}")
        if chain is not None:
            resnums = sorted(set(chain.getResnums()))
            min_res = min(resnums)
            max_res = max(resnums)
            region_parts.append(f"{chain_id}:{min_res}-{max_res}")
            print(f"Chain {chain_id}: residues {min_res}-{max_res}")
        else:
            print(f"WARNING: Chain {chain_id} not found in PDB")

    region_str = ",".join(region_parts)
    print(f"Using region specification: {region_str}")

    # Verify chains
    for chain_id in chains_list:
        chain = structure.select(f"chain {chain_id}")
        if chain:
            print(f"  Chain {chain_id} verified: {len(chain)} atoms")
        else:
            print(f"  WARNING: Chain {chain_id} not found!")

    return region_str


def merge_pdb_chains(pdb_str: str, chains_to_merge: list) -> str:
    """Merge multiple chains in a PDB structure into the first chain in the list.

    Copied from modal_iggm.py for consistency.
    """
    import prody as pr
    from io import StringIO

    print(f"merge_pdb_chains called with chains: {chains_to_merge}")

    if not chains_to_merge or len(chains_to_merge) < 2:
        raise ValueError("Need at least 2 chains to combine")

    structure = pr.parsePDBStream(StringIO(pdb_str))
    target_chain = chains_to_merge[0]

    chain_atoms = []
    current_resnum = 1

    for chain_id in chains_to_merge:
        chain = structure.select(f"chain {chain_id}")
        if chain is not None:
            unique_resnums = sorted(set(chain.getResnums()))
            resnum_mapping = {
                old: current_resnum + i for i, old in enumerate(unique_resnums)
            }
            new_resnums = [resnum_mapping[old] for old in chain.getResnums()]
            chain.setResnums(new_resnums)
            chain.setChids([target_chain] * len(chain))
            chain_atoms.append(chain)
            current_resnum += len(unique_resnums)

    if not chain_atoms:
        raise ValueError(
            f"None of the specified chains {chains_to_merge} found in structure"
        )

    combined_chains = chain_atoms[0]
    for chain in chain_atoms[1:]:
        combined_chains = combined_chains + chain

    # Select chains that were NOT merged (from original structure)
    other_chains_selection = f"not chain {' '.join(chains_to_merge)}"
    other_chains = structure.select(other_chains_selection)

    # Put combined chains FIRST, then any other chains after
    if other_chains is not None:
        final_structure = combined_chains + other_chains
    else:
        final_structure = combined_chains

    output_stream = StringIO()
    pr.writePDBStream(output_stream, final_structure)
    return output_stream.getvalue()


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes={f"/{GERMINAL_VOLUME_NAME}": GERMINAL_MODEL_VOLUME},
)
def mber_design_vhh(
    target_pdb_content: bytes,
    target_id: str,
    target_name: str,
    masked_binder_seq: str = DEFAULT_VHH_MASKED,
    chains: str = "A",
    target_hotspot_residues: str | None = None,
    include_surrounding_context: bool = False,
) -> list[tuple[Path, bytes]]:
    """Design a VHH binder against a target using mBER.

    Args:
        target_pdb_content: PDB file content as bytes
        target_id: Target protein ID (e.g., PDB ID)
        target_name: Name of the target protein
        masked_binder_seq: VHH sequence with * marking positions to design
        chains: Chains to process (e.g., "A" or "A,B")
        target_hotspot_residues: Specific hotspot residues (e.g., "A110,A120")

    Returns:
        List of (filename, content) tuples with all output files
    """
    import sys
    from tempfile import TemporaryDirectory

    print(f"Starting mBER VHH design for {target_name} (ID: {target_id})")
    print(f"Chains to process: {chains}")
    print(f"Masked binder sequence: {masked_binder_seq}")
    if target_hotspot_residues:
        print(f"Hotspot residues: {target_hotspot_residues}")
    sys.stdout.flush()

    # Build region specification from PDB structure
    region_str = build_region_string(target_pdb_content, chains)
    print(f"Target hotspots: {target_hotspot_residues}")
    print(f"Include surrounding context: {include_surrounding_context}")
    sys.stdout.flush()

    pdb_content_str = target_pdb_content.decode("utf-8")

    try:
        from mber_protocols.stable.VHH_binder_design.config import (
            ModelConfig,
            LossConfig,
            TrajectoryConfig,
            EnvironmentConfig,
            TemplateConfig,
            EvaluationConfig,
        )
        from mber_protocols.stable.VHH_binder_design.template import TemplateModule
        from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
        from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
        from mber_protocols.stable.VHH_binder_design.state import (
            DesignState,
            TemplateData,
        )

        print("Successfully imported mBER modules")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error importing mBER modules: {e}")
        raise

    with TemporaryDirectory() as tmpdir:
        # Write target PDB
        pdb_path = Path(tmpdir) / "target.pdb"
        pdb_path.write_bytes(target_pdb_content)
        print(f"Wrote target PDB to {pdb_path}")

        # Create output directory
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        # Initialize configurations
        print("Initializing configurations...")
        template_config = TemplateConfig()
        template_config.include_surrounding_context = include_surrounding_context
        model_config = ModelConfig()
        loss_config = LossConfig()
        trajectory_config = TrajectoryConfig()
        # Point to AlphaFold params on germinal-models volume
        environment_config = EnvironmentConfig(
            af_params_dir="/germinal-models/alphafold_params"
        )
        evaluation_config = EvaluationConfig()

        # Create design state with PDB content and region specification
        design_state = DesignState(
            template_data=TemplateData(
                target_id=target_id,
                target_name=target_name,
                masked_binder_seq=masked_binder_seq,
                region=region_str,  # region specification like "A:1-10,B:1-170"
                target_hotspot_residues=target_hotspot_residues,
                template_pdb=pdb_content_str,  # Pre-populate with provided PDB content
                target_source="input",  # Mark as user-provided
                include_surrounding_context=include_surrounding_context,
            )
        )

        # Run template module
        print(f"\n{'=' * 60}")
        print(f"TEMPLATE MODULE - {target_name}")
        print(f"{'=' * 60}")
        sys.stdout.flush()
        template_module = TemplateModule(
            template_config=template_config,
            environment_config=environment_config,
        )
        template_module.setup(design_state)
        design_state = template_module.run(design_state)
        template_module.teardown(design_state)
        print(f"Template module completed")

        # Run trajectory module
        print(f"\n{'=' * 60}")
        print(f"TRAJECTORY MODULE - {target_name}")
        print(f"{'=' * 60}")
        sys.stdout.flush()
        trajectory_module = TrajectoryModule(
            model_config=model_config,
            loss_config=loss_config,
            trajectory_config=trajectory_config,
            environment_config=environment_config,
        )
        trajectory_module.setup(design_state)
        design_state = trajectory_module.run(design_state)
        trajectory_module.teardown(design_state)
        print(f"Trajectory module completed")

        # Run evaluation module
        print(f"\n{'=' * 60}")
        print(f"EVALUATION MODULE - {target_name}")
        print(f"{'=' * 60}")
        sys.stdout.flush()
        evaluation_module = EvaluationModule(
            model_config=model_config,
            evaluation_config=evaluation_config,
            loss_config=loss_config,
            environment_config=environment_config,
        )
        evaluation_module.setup(design_state)
        design_state = evaluation_module.run(design_state)
        evaluation_module.teardown(design_state)
        print(f"Evaluation module completed")

        # Save results
        print(f"\nSaving results to {output_dir}...")
        design_state.to_dir(str(output_dir))

        # Collect all output files
        outputs = []
        for out_file in output_dir.rglob("*"):
            if out_file.is_file():
                outputs.append(
                    (out_file.relative_to(output_dir), out_file.read_bytes())
                )

        print(f"\n{'=' * 60}")
        print(f"Design completed for {target_name}!")
        print(f"Generated {len(outputs)} output files")
        print(f"{'=' * 60}")
        sys.stdout.flush()

        return outputs


@app.local_entrypoint()
def main(
    target_pdb: str,
    target_name: str,
    target_id: str | None = None,
    masked_binder_seq: str = DEFAULT_VHH_MASKED,
    chains: str = "A",
    target_hotspot_residues: str | None = None,
    include_surrounding_context: bool = True,
    chain_offsets: str | None = None,
    output_dir: str = "./out/mber",
    run_name: str | None = None,
):
    """Design a VHH binder against a target PDB using mBER."""
    from datetime import datetime

    # Read target PDB
    target_pdb_path = Path(target_pdb)
    if not target_pdb_path.exists():
        raise FileNotFoundError(f"Target PDB file not found: {target_pdb}")

    target_pdb_content = target_pdb_path.read_bytes()

    # Apply chain renumbering if specified
    if chain_offsets:
        # Format: "B:200,C:400" - renumber chain B starting at 200, chain C at 400
        for offset_spec in chain_offsets.split(","):
            chain_id, offset_str = offset_spec.strip().split(":")
            offset = int(offset_str)
            print(f"Renumbering chain {chain_id} with offset {offset}")
            target_pdb_content = renumber_pdb_chain(target_pdb_content, chain_id, offset)

    # Use filename as target_id if not provided
    if target_id is None:
        target_id = target_pdb_path.stem

    # Other tools use X, this tool uses * apparently
    masked_binder_seq = masked_binder_seq.replace("X", "*")

    print(f"Designing VHH binder for {target_name} (ID: {target_id})")
    print(f"Chains to process: {chains}")
    print(f"Masked binder sequence: {masked_binder_seq}")
    if target_hotspot_residues:
        print(f"Target hotspot residues: {target_hotspot_residues}")

    # Run design
    outputs = mber_design_vhh.remote(
        target_pdb_content=target_pdb_content,
        target_id=target_id,
        target_name=target_name,
        masked_binder_seq=masked_binder_seq,
        chains=chains,
        target_hotspot_residues=target_hotspot_residues,
        include_surrounding_context=include_surrounding_context,
    )

    # Create output directory
    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(output_dir) / (run_name or f"{target_name}_{today}")
    out_dir_full.mkdir(parents=True, exist_ok=True)

    # Write output files
    for out_file, content in outputs:
        output_path = out_dir_full / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)

    print(f"\nDesign completed! Output saved to: {out_dir_full}")
    print(f"Generated {len(outputs)} files")

    # List key output files
    key_files = ["final_designs.fasta", "evaluation_results.csv", "best_design.pdb"]
    print("\nKey output files:")
    for key_file in key_files:
        if (out_dir_full / key_file).exists():
            print(f"  - {key_file}")
