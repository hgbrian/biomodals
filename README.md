# biomodals
Bioinformatics tools running on modal.

## install and set up modal
```
pip install modal
python3 -m modal setup
```

Or alternatively, use uv:
```
uv run --with modal modal run modal_minimap.py
```

## AF2Rank

```
wget https://files.rcsb.org/download/1YWI.pdb
modal run modal_af2rank.py --input-pdb 1YWI.pdb --run-name 1YWI
```

## OmegaFold

Runs `omegafold --model 2 <fasta>`

This can run out of memory, and may need an 80GB A100.

```
wget https://www.rcsb.org/fasta/entry/3NIT -O 3NIT.faa
modal run modal_omegafold.py --input-fasta 3NIT.faa
```

## minimap2 (short reads example)

Runs `minimap2 -ax sr <fasta> <reads>`

Just a simple example of running a binary on a powerful box.

```
wget https://gist.githubusercontent.com/hgbrian/56787d9b3ce2e68f698ac94d537340d8/raw/mito.fasta
wget https://gist.githubusercontent.com/hgbrian/802d8094bb4fed435bbb93a8c9092ee2/raw/mito_reads.fastq
modal run modal_minimap2.py --input-ref-fasta mito.fasta --input-reads-fastq mito_reads.fastq
```

## AlphaFold-Multimer
A very basic implementation.
```
wget https://www.rcsb.org/fasta/entry/3NIT -O 3NIT.faa
modal run modal_alphafold.py --input-fasta 3NIT.faa
```

## AFDesign

Create a cyclic peptide against a pdb file (using `pdb-redo` data by default)

```
modal run modal_afdesign.py --pdb 4MZK --target-chain A
```

Set the first and last amino acid of the (cyclic) peptide to cysteine.
Here using a small number of iterations for speed reasons...
Use `--soft-iters 30` `--hard-iters 6` or more for better results.

```
modal run modal_afdesign.py --pdb 1A00 --target-chain A --soft-iters 2 --hard-iters 2 --binder-len 6 --set-fixed-aas C....C
```

Create a linear peptide against a local PDB file that has been manually edited.
This is unfortunately sometimes necessary when e.g. a chain is too long or there are too many chains.

```
modal run modal_afdesign.py --pdb in/afdesign/1igy_cropped.fixed.pdb --target-chain B
```

## DiffDock

Dock a .mol2 file against a local pdb file.
DiffDock may require an 80GB A100 to run for larger proteins.

```
wget https://files.rcsb.org/download/1IGY.pdb
wget https://gist.github.com/hgbrian/393ec799893cbf518f3084847c17cb2d/raw/1IGY_example.mol2
modal run modal_diffdock.py --pdb 1IGY.pdb --mol2 in/diffdock/1IGY_example.mol2
```

## ESM2

Predict the amino acid at a masked position in a sequence.

```
echo ">1\nMA<mask>GMT" > test_esm2.faa
modal run modal_esm2_predict_masked.py --input-faa test_esm2.faa
```

## pdb2png

A simple pymol-based script to convert PDBs to PNGs for easy output viewing.

```
wget https://files.rcsb.org/download/1YWI.pdb
modal run modal_pdb2png.py --input-pdb 1YWI.pdb --protein-zoom 0.8 --protein-color 240,200,190
```

## ANARCI
A tool for annotating antibody sequences https://github.com/oxpig/ANARCI

```
echo ">test_anarci\nDIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLESGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIKRT" > test_anarci.faa
modal run modal_anarci.py --input-fasta test_anarci.faa
```

## MD_protein_ligand
Basic MD
```
modal run modal_MD_protein_ligand.py --pdb-id in/md_protein_ligand/1A1O_reordered.pdb
```

Basic MD protein + ligand
```
modal run modal_md_protein_ligand.py --pdb-id 4O75 --ligand-id 2RC --ligand-chain A
```

## BindCraft

Basic PDL1 binder (example from https://github.com/martinpacesa/BindCraft)
```
wget https://raw.githubusercontent.com/martinpacesa/BindCraft/refs/heads/main/example/PDL1.pdb
GPU=A100 modal run modal_bindcraft.py --input-pdb PDL1.pdb --number-of-final-designs 1
```

# nextflow
Minimal hello world app, with conda and nextflow installed (not trivial!)
```
modal run modal_nextflow_example.py
```

# Boltz-1
[Boltz-1](https://github.com/jwohlwend/boltz), an open source AlphaFold 3-like model.
```
echo "sequences:\n    - protein:\n        id: A\n        sequence: TDKLIFGKGTRVTVEP" > test_boltz.yaml
modal run modal_boltz.py --input-yaml test_boltz.yaml
```

# Chai-1
[Chai-1](https://github.com/chaidiscovery/chai-lab), another open source AlphaFold 3-like model.
```
echo ">protein|name=insulin\nMAWTPLLLLLLSHCTGSLSQPVLTQPTSLSASPGASARFTCTLRSGINVGTYRIYWYQQKPGSLPRYLLRYKSDSDKQGSGVPSRFSGSKDASTNAGLLLISGLQSEDEADYYCAIWYSSTS\n>RNA|name=rna\nACUGACUGGAAGUCCCCCGUAGUACCCGACG\n>ligand|name=caffeine\nN[C@@H](Cc1ccc(O)cc1)C(=O)O" > test_chai1.faa
modal run modal_chai1.py --input-faa test_chai1.faa
```

## LigandMPNN

```
wget https://files.rcsb.org/download/1IVO.pdb
modal run modal_ligandmpnn.py --input-pdb 1IVO.pdb --extract-chains AC --params-str '--seed 1 --checkpoint_protein_mpnn "/LigandMPNN/model_params/proteinmpnn_v_48_020.pt"  --chains_to_design "C" --save_stats 1'
```

## RFDiffusion
Not working currently due to library incompatibilities!

## Other modal repos

- [RNA Seq Pipeline](https://github.com/tdsone/modal-rna-seq-pipeline) by @tdsone
