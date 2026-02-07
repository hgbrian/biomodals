"""Unit tests for _fasta_to_yaml from modal_boltz.py."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
import yaml

from modal_boltz import _fasta_to_yaml


def _write_fasta(content: str) -> str:
    """Write FASTA content to a temp file and return path."""
    f = NamedTemporaryFile(mode="w", suffix=".faa", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


class TestFastaToYaml:
    def test_simple_protein(self):
        path = _write_fasta(">test\nTDKLIFGKGTRVTVEP")
        result = yaml.safe_load(_fasta_to_yaml(path))
        assert len(result["sequences"]) == 1
        entry = result["sequences"][0]
        assert "protein" in entry
        assert entry["protein"]["sequence"] == "TDKLIFGKGTRVTVEP"
        assert entry["protein"]["id"] == "A"

    def test_multi_chain(self):
        fasta = ">chain1\nTDKLIFGK\n>chain2\nMKFLILLF"
        path = _write_fasta(fasta)
        result = yaml.safe_load(_fasta_to_yaml(path))
        assert len(result["sequences"]) == 2
        assert result["sequences"][0]["protein"]["id"] == "A"
        assert result["sequences"][1]["protein"]["id"] == "B"

    def test_dna_entity_type(self):
        fasta = ">A|dna|empty\nACGTACGT"
        path = _write_fasta(fasta)
        result = yaml.safe_load(_fasta_to_yaml(path))
        assert "dna" in result["sequences"][0]
        assert result["sequences"][0]["dna"]["sequence"] == "ACGTACGT"

    def test_rna_entity_type(self):
        fasta = ">A|rna|empty\nACGUACGU"
        path = _write_fasta(fasta)
        result = yaml.safe_load(_fasta_to_yaml(path))
        assert "rna" in result["sequences"][0]

    def test_chain_id_from_header(self):
        fasta = ">B|protein|empty\nMKFLILLF"
        path = _write_fasta(fasta)
        result = yaml.safe_load(_fasta_to_yaml(path))
        assert result["sequences"][0]["protein"]["id"] == "B"
