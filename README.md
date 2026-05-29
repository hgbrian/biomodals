# biomodals
Bioinformatics tools running on modal.

## install and set up modal
```bash
pip install modal
python3 -m modal setup
```

Or alternatively, use uv, e.g.:
```bash
uv run --with modal modal run modal_minimap2.py
```

## Apps

Sorted alphabetically.

- [AF2Rank](#af2rank) — structure ranking via AF2 prediction
- [AFDesign](#afdesign) — peptide/binder design via AlphaFold2
- [AlphaFold-Multimer](#alphafold-multimer) — multimer structure prediction
- [ANARCI](#anarci) — antibody sequence annotation
- [BindCraft](#bindcraft) — protein binder design
- [Boltz](#boltz) — AF3-like open structure prediction
- [BoltzGen](#boltzgen) — generative structure model
- [Chai-1](#chai-1) — AF3-like open structure prediction
- [DiffDock](#diffdock) — small molecule docking
- [ESM2](#esm2-masked-position-prediction) — masked amino acid prediction
- [FASPR](#faspr-side-chain-packing) — side-chain packing
- [Germinal](#germinal) — binder design
- [IgGM](#iggm) — antibody design
- [LigandMPNN](#ligandmpnn) — protein sequence design conditioned on ligands
- [mBER](#mber-vhh-nanobody-design) — VHH nanobody design
- [minimap2](#minimap2-short-reads-example) — short-read alignment
- [nextflow](#nextflow) — nextflow hello-world image
- [pdb2png](#pdb2png) — PDB → PNG rendering via pymol
- [Protenix](#protenix) — AF3 reproduction
- [RSO](#rso-binder-design) — Rejection Sampling Optimization binder design
- [SASA](#sasa-solvent-accessible-surface-area) — solvent-accessible surface area
- [tmol](#tmol-rosetta-energy-scoring) — GPU Rosetta energy scoring
- [USalign](#usalign-structural-alignment) — TM-score / RMSD structural alignment

## AF2Rank

```bash
wget https://files.rcsb.org/download/1YWI.pdb
uv run --with modal modal run modal_af2rank.py --input-pdb 1YWI.pdb --run-name 1YWI
```

## AFDesign

Create a cyclic peptide against a pdb file (using `pdb-redo` data by default)

```bash
uv run --with modal modal run modal_afdesign.py --pdb 4MZK --target-chain A
```

Set the first and last amino acid of the (cyclic) peptide to cysteine.
Here using a small number of iterations for speed reasons...
Use `--soft-iters 30` `--hard-iters 6` or more for better results.

```bash
uv run --with modal modal run modal_afdesign.py --pdb 1A00 --target-chain A --soft-iters 2 --hard-iters 2 --binder-len 6 --set-fixed-aas C....C
```

Create a linear peptide against a local PDB file that has been manually edited.
This is unfortunately sometimes necessary when e.g. a chain is too long or there are too many chains.

```bash
uv run --with modal modal run modal_afdesign.py --pdb in/afdesign/1igy_cropped.fixed.pdb --target-chain B
```

## AlphaFold-Multimer
A very basic implementation.
```bash
wget https://www.rcsb.org/fasta/entry/3NIT -O 3NIT.faa
uv run --with modal modal run modal_alphafold.py --input-fasta 3NIT.faa
```

## ANARCI
A tool for annotating antibody sequences https://github.com/oxpig/ANARCI

```bash
printf '>test_anarci\nDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLESGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT\n' > test_anarci.faa
uv run --with modal modal run modal_anarci.py --input-fasta test_anarci.faa
```

## BindCraft

Basic PDL1 binder (example from https://github.com/martinpacesa/BindCraft)
```bash
wget https://raw.githubusercontent.com/martinpacesa/BindCraft/refs/heads/main/example/PDL1.pdb
GPU=A100 uv run --with modal modal run modal_bindcraft.py --input-pdb PDL1.pdb --number-of-final-designs 1
```

## Boltz
[Boltz](https://github.com/jwohlwend/boltz), an open source AlphaFold 3-like model.
```bash
printf 'sequences:\n    - protein:\n        id: A\n        sequence: TDKLIFGKGTRVTVEP\n' > test_boltz.yaml
uv run --with modal modal run modal_boltz.py --input-yaml test_boltz.yaml --params-str "--seed 42"
```

## BoltzGen
[BoltzGen](https://github.com/HannesStark/boltzgen), generative model for biomolecular structures.
```bash
wget https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13.cif
wget https://raw.githubusercontent.com/HannesStark/boltzgen/refs/heads/main/example/vanilla_protein/1g13prot.yaml
uv run --with modal modal run modal_boltzgen.py --input-yaml 1g13prot.yaml --num-designs 1
```

## Chai-1
[Chai-1](https://github.com/chaidiscovery/chai-lab), another open source AlphaFold 3-like model.
```bash
printf '>protein|name=insulin\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS\n>RNA|name=rna\nACUGACUGGAAGUCCCCCGUAGUACCCGACG\n>ligand|name=caffeine\nN[C@@H](Cc1ccc(O)cc1)C(=O)O\n' > test_chai1.faa
uv run --with modal modal run modal_chai1.py --input-faa test_chai1.faa
```

## DiffDock

WARNING: DiffDock's image build is very slow (downloads ~3GB of ESM2 + DiffDock models).

Dock a .mol2 file against a local pdb file.
DiffDock may require an 80GB A100 to run for larger proteins.

```bash
wget https://files.rcsb.org/download/1IGY.pdb
wget https://gist.github.com/hgbrian/393ec799893cbf518f3084847c17cb2d/raw/1IGY_example.mol2
uv run --with modal modal run modal_diffdock.py --pdb-file 1IGY.pdb --mol2-file 1IGY_example.mol2
```

## ESM2 (masked-position prediction)

Predict the amino acid at a masked position in a sequence.

```bash
printf '>1\nMA<mask>GMT\n' > test_esm2.faa
uv run --with modal modal run modal_esm2_predict_masked.py --input-faa test_esm2.faa
```

## FASPR (side-chain packing)

[FASPR](https://github.com/tommyhuangthu/FASPR) — fast and accurate side-chain
packing. Repacks side chains of a PDB (requires complete main-chain atoms) and
can introduce mutations via a sequence file.

```bash
wget https://files.rcsb.org/download/1CRN.pdb
uv run --with modal modal run modal_faspr.py --input-pdb 1CRN.pdb
```

## Germinal
Germinal took some serious hacking to get working.
It seems to work ok but buyer beware.
I recommend using BoltzGen instead.
Unlike some other apps here, it creates a Volume to store params instead of storing them in the image.

```bash
# Get the PD-L1 structure from RCSB PDB
wget https://files.rcsb.org/download/5O45.pdb

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
uv run --with modal --with PyYAML modal run modal_germinal.py --target-yaml target_example.yaml --max-trajectories 1 --max-passing-designs 1
```

## IgGM
[IgGM](https://github.com/THUNLP-MT/IgGM), antibody design model.
```bash
wget https://files.rcsb.org/download/5O45.pdb
printf '>H\nEVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS\n>L\nDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT\n>A\n' > test_iggm.faa
uv run --with modal modal run modal_iggm.py --input-fasta test_iggm.faa --antigen 5O45.pdb --epitope 19,20,21
```

## LigandMPNN

```bash
wget https://files.rcsb.org/download/1IVO.pdb
uv run --with modal modal run modal_ligandmpnn.py --input-pdb 1IVO.pdb --extract-chains AC --params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt"  --chains_to_design "C" --save_stats 1'
```

## mBER (VHH nanobody design)

Design VHH nanobody binders against a target protein using [mBER](https://github.com/manifoldbio/mber-open).

```bash
wget https://files.rcsb.org/download/7STF.pdb
uv run --with modal modal run modal_mber.py --target-pdb 7STF.pdb --target-name PDL1
```

With custom masked sequence (* = positions to design):
```bash
uv run --with modal modal run modal_mber.py --target-pdb target.pdb --target-name MyTarget \
    --masked-binder-seq "EVQLVESGGGLVQPGGSLRLSCAASG*********WFRQAPGKEREF***********NADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYC************WGQGTLVTVSS"
```

## minimap2 (short reads example)

Runs `minimap2 -ax sr <fasta> <reads>`

Just a simple example of running a binary on a powerful box.

```bash
wget https://gist.githubusercontent.com/hgbrian/56787d9b3ce2e68f698ac94d537340d8/raw/mito.fasta
wget https://gist.githubusercontent.com/hgbrian/802d8094bb4fed435bbb93a8c9092ee2/raw/mito_reads.fastq
uv run --with modal modal run modal_minimap2.py --input-ref-fasta mito.fasta --input-reads-fastq mito_reads.fastq
```

## nextflow
Minimal hello world app, with conda and nextflow installed (not trivial!)
```bash
uv run --with modal modal run modal_nextflow_example.py
```

## pdb2png

A simple pymol-based script to convert PDBs to PNGs for easy output viewing.

```bash
wget https://files.rcsb.org/download/1YWI.pdb
uv run --with modal modal run modal_pdb2png.py --input-pdb 1YWI.pdb --protein-zoom 0.8 --protein-color 240,200,190
```

## Protenix

[Protenix](https://github.com/bytedance/Protenix), an open-source PyTorch reproduction of AlphaFold 3.
```bash
printf '>protein|A\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS\n' > test_protenix.faa
uv run --with modal modal run modal_protenix.py --input-faa test_protenix.faa --seeds 42 --no-use-msa
```

## RSO (binder design)

Design binders using [RSO](https://github.com/jykim/rso) (Rejection Sampling Optimization).
```bash
wget https://files.rcsb.org/download/5O45.pdb
grep "^ATOM.*\ A\ " 5O45.pdb > 5O45_chainA.pdb
uv run --with modal modal run modal_rso.py --input-pdb 5O45_chainA.pdb --num-designs 1 --traj-iters 10 --binder-len 30
```

## SASA (solvent-accessible surface area)

[dr_sasa](https://github.com/nioroso-x3/dr_sasa_n) — annotates a PDB (or CIF,
auto-converted via openbabel) with per-atom SASA in the B-factor column.

```bash
wget https://files.rcsb.org/download/1CRN.pdb
uv run --with modal modal run modal_sasa.py --input-pdb 1CRN.pdb
# pymol out/sasa/<run>/1CRN.asa.pdb
```

## tmol (Rosetta energy scoring)

[tmol](https://github.com/uw-ipd/tmol) — GPU-accelerated Rosetta beta_nov2016
energy scoring. Runs an L-BFGS Cartesian relax and (optionally) a FASPR repack
before scoring. Pass `--input-dir` to score every PDB in a directory in parallel.

Note: tmol requires a clean PDB (no chain breaks, no missing residues, no
extra hydrogens). PDBs straight from the RCSB often need preprocessing.

```bash
modal deploy modal_faspr.py     # only needed if using --faspr (default)
wget https://files.rcsb.org/download/1CRN.pdb
uv run --with modal modal run modal_tmol.py --input-pdb 1CRN.pdb
```

## USalign (structural alignment)

[US-align](https://zhanggroup.org/US-align/) — universal structural alignment
(successor to TM-align). Reports TM-score and RMSD between two or more
structures (PDB or CIF, proteins/RNA/DNA, monomer or complex).

```bash
wget https://files.rcsb.org/download/5O45.pdb
grep "^ATOM.*\ A\ " 5O45.pdb > 5O45_chainA.pdb
uv run --with modal modal run modal_usalign.py --pdb 5O45.pdb --vs-pdbs 5O45_chainA.pdb
```

Score a single chain inside aligned complexes (two-step alignment):
```bash
uv run --with modal modal run modal_usalign.py --pdb 5O45.pdb --vs-pdbs 5O45_chainA.pdb --chain A
```

## Testing

Build all images (no GPU, but slow):
```bash
uv run --with modal --with pytest pytest tests/test_build_images.py -v
uv run --with modal --with pytest pytest tests/test_build_images.py -v -k alphafold  # single app
```

Run all apps with minimal inputs (uses GPU, costs money):
```bash
uv run --with modal --with pytest pytest tests/test_quick_runs.py -v
uv run --with modal --with pytest pytest tests/test_quick_runs.py -v -k sasa  # single app
```

## Other modal repos

- [RNA Seq Pipeline](https://github.com/tdsone/modal-rna-seq-pipeline) by @tdsone
