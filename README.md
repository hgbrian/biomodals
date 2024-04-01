# biomodals
Bioinformatics tools running on modal.

## install and set up modal
```
pip install modal
python3 -m modal setup
```


## OmegaFold

Runs `omegafold --model 2 <fasta>`

This can run out of memory too. 

```
modal run modal_omegafold.py --input-fasta modal_in/omegafold/insulin.faa
```
## minimap2 (short reads example)

Runs `minimap2 -ax sr <fasta> <reads>`

Just a simple example of running any script on a large box.

```
modal run modal_minimap2.py --input-fasta modal_in/minimap2/mito.fasta --input-reads modal_in/minimap2/reads.fastq
```

## AFDesign

Cyclic peptide against pdb (using pdb-redo)
```
modal run modal_afdesign.py --pdb 4MZK --target-chain A
```

Set the first and last amino acid of the (cyclic) peptide to cysteine.
Here using a small number of iterations for speed reasons...
Use `--soft-iters 30` `--hard-iters 6` or more.
```
modal run modal_afdesign.py --pdb 1A00 --target-chain A --soft-iters 2 --hard-iters 2 --binder-len 6 --set-fixed-aas C....C
```

Linear peptide against a local PDB file that has been manually edited.
This is unfortunately sometimes necessary when e.g. a chain is too long!
```
modal run modal_afdesign.py --pdb in/afdesign/1igy_cropped.fixed.pdb --target-chain B
```

## DiffDock

DiffDock seems to require an 80GB A100 to run without running out of memory.

```
modal run modal_diffdock.py --pdb in/diffdock/1igy.pdb --mol2 in/diffdock/1igy.mol2
```

## TODO
- unify file input / output in some sensible way
- replace modal_in directory to in for all examples