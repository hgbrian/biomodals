# biomodals
Bioinformatics tools running on modal.

## install and set up modal
```bash
pip install modal
python3 -m modal setup
```

Or alternatively, use uvx, e.g.:
```bash
uvx modal run modal_minimap.py
```

## AF2Rank

```bash
wget https://files.rcsb.org/download/1YWI.pdb
uvx modal run modal_af2rank.py --input-pdb 1YWI.pdb --run-name 1YWI
```

## minimap2 (short reads example)

Runs `minimap2 -ax sr <fasta> <reads>`

Just a simple example of running a binary on a powerful box.

```bash
wget https://gist.githubusercontent.com/hgbrian/56787d9b3ce2e68f698ac94d537340d8/raw/mito.fasta
wget https://gist.githubusercontent.com/hgbrian/802d8094bb4fed435bbb93a8c9092ee2/raw/mito_reads.fastq
uvx modal run modal_minimap2.py --input-ref-fasta mito.fasta --input-reads-fastq mito_reads.fastq
```

## AlphaFold-Multimer
A very basic implementation.
```bash
wget https://www.rcsb.org/fasta/entry/3NIT -O 3NIT.faa
uvx modal run modal_alphafold.py --input-fasta 3NIT.faa
```

## AFDesign

Create a cyclic peptide against a pdb file (using `pdb-redo` data by default)

```bash
uvx modal run modal_afdesign.py --pdb 4MZK --target-chain A
```

Set the first and last amino acid of the (cyclic) peptide to cysteine.
Here using a small number of iterations for speed reasons...
Use `--soft-iters 30` `--hard-iters 6` or more for better results.

```bash
uvx modal run modal_afdesign.py --pdb 1A00 --target-chain A --soft-iters 2 --hard-iters 2 --binder-len 6 --set-fixed-aas C....C
```

Create a linear peptide against a local PDB file that has been manually edited.
This is unfortunately sometimes necessary when e.g. a chain is too long or there are too many chains.

```bash
uvx modal run modal_afdesign.py --pdb in/afdesign/1igy_cropped.fixed.pdb --target-chain B
```

## DiffDock

Dock a .mol2 file against a local pdb file.
DiffDock may require an 80GB A100 to run for larger proteins.

```bash
wget https://files.rcsb.org/download/1IGY.pdb
wget https://gist.github.com/hgbrian/393ec799893cbf518f3084847c17cb2d/raw/1IGY_example.mol2
uvx modal run modal_diffdock.py --pdb 1IGY.pdb --mol2 in/diffdock/1IGY_example.mol2
```

## ESM2

Predict the amino acid at a masked position in a sequence.

```bash
echo ">1\nMA<mask>GMT" > test_esm2.faa
uvx modal run modal_esm2_predict_masked.py --input-faa test_esm2.faa
```

## pdb2png

A simple pymol-based script to convert PDBs to PNGs for easy output viewing.

```bash
wget https://files.rcsb.org/download/1YWI.pdb
uvx modal run modal_pdb2png.py --input-pdb 1YWI.pdb --protein-zoom 0.8 --protein-color 240,200,190
```

## ANARCI
A tool for annotating antibody sequences https://github.com/oxpig/ANARCI

```bash
echo ">test_anarci\nDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLESGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT" > test_anarci.faa
uvx modal run modal_anarci.py --input-fasta test_anarci.faa
```

## MD_protein_ligand
Basic MD
```bash
wget https://files.rcsb.org/download/5O45.pdb
grep "^ATOM.*\ A\ " 5O45.pdb > 5O45_chainA.pdb
uvx modal run modal_md_protein_ligand.py --pdb-id 5O45_chainA.pdb
```

Basic MD protein + ligand
```bash
uvx modal run modal_md_protein_ligand.py --pdb-id 4O75 --ligand-id 2RC --ligand-chain A
```

## BindCraft

Basic PDL1 binder (example from https://github.com/martinpacesa/BindCraft)
```bash
wget https://raw.githubusercontent.com/martinpacesa/BindCraft/refs/heads/main/example/PDL1.pdb
GPU=A100 uvx modal run modal_bindcraft.py --input-pdb PDL1.pdb --number-of-final-designs 1
```

# nextflow
Minimal hello world app, with conda and nextflow installed (not trivial!)
```bash
uvx modal run modal_nextflow_example.py
```

# Boltz
[Boltz](https://github.com/jwohlwend/boltz), an open source AlphaFold 3-like model.
```bash
echo "sequences:\n    - protein:\n        id: A\n        sequence: TDKLIFGKGTRVTVEP" > test_boltz.yaml
uvx modal==1.2.1 run modal_boltz.py --input-yaml test_boltz.yaml
```

# Chai-1
[Chai-1](https://github.com/chaidiscovery/chai-lab), another open source AlphaFold 3-like model.
```bash
echo ">protein|name=insulin\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS\n>RNA|name=rna\nACUGACUGGAAGUCCCCCGUAGUACCCGACG\n>ligand|name=caffeine\nN[C@@H](Cc1ccc(O)cc1)C(=O)O" > test_chai1.faa
uvx modal run modal_chai1.py --input-faa test_chai1.faa
```

## LigandMPNN

```bash
wget https://files.rcsb.org/download/1IVO.pdb
uvx modal run modal_ligandmpnn.py --input-pdb 1IVO.pdb --extract-chains AC --params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt"  --chains_to_design "C" --save_stats 1'
```

## RFDiffusion
Not working currently due to library incompatibilities!
I am not sure this model is useful anymore.
I would use one of the many newer options (especially BindCraft and BoltzGen).

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
uvx --with PyYAML modal run modal_germinal.py --target-yaml target_example.yaml --max-trajectories 1 --max-passing-designs 1
```

## Other modal repos

- [RNA Seq Pipeline](https://github.com/tdsone/modal-rna-seq-pipeline) by @tdsone
