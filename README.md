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

```
modal run modal_afdesign.py --pdb 4MZK --target-chain A
```

## DiffDock

DiffDock seems to require an 80GB A100 to run without running out of memory.

```
modal run modal_diffdock.py --pdb in/diffdock/1igy.pdb --mol2 in/diffdock/1igy.mol2
```

## TODO
- unify file input / output in some sensible way
- replace modal_in directory to in for all examples