# biomodals
Bioinformatics tools running on modal.

## install and set up modal
```
pip install modal
python3 -m modal setup
```

## OmegaFold
```
modal run modal_omegafold.py --input-fasta modal_in/omegafold/insulin.faa
```
## minimap2 (short reads example)
```
modal run modal_minimap2.py --input-fasta modal_in/minimap2/mito.fasta --input-reads modal_in/minimap2/reads.fastq
```
