# biomodals
Bioinformatics tools running on modal.

## install and set up modal
```
pip install modal
python3 -m modal setup
```

## OmegaFold

Runs `omegafold --model 2 <fasta>`

This can run out of memory, and may need an 80GB A100. 

```
modal run modal_omegafold.py --input-fasta in/omegafold/insulin.faa
```

## minimap2 (short reads example)

Runs `minimap2 -ax sr <fasta> <reads>`

Just a simple example of running a binary on a powerful box.

```
modal run modal_minimap2.py --input-fasta in/minimap2/mito.fasta --input-reads in/minimap2/reads.fastq
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
modal run modal_diffdock.py --pdb in/diffdock/1igy.pdb --mol2 in/diffdock/1igy.mol2
```

## pdb2png

A simple pymol-based script to convert PDBs to PNGs for easy output viewing.

```
modal run modal_pdb2png.py --input-pdb in/pdb2png/1igy.pdb --protein-zoom 0.8 --protein-color 240,200,190
```

## ANARCI
A tool for annotating antibody sequences https://github.com/oxpig/ANARCI

```
modal run modal_anarci.py --input-fasta in/anarci/antibody.faa
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

