"""Unit tests for FASTA validation logic from modal_alphafold.py."""

import pytest


def validate_fasta(fasta_str: str):
    """Replicates the validation logic from modal_alphafold.py:182-185."""
    header = fasta_str.splitlines()[0]
    fasta_seq = "".join(seq.strip() for seq in fasta_str.splitlines()[1:])
    if header[0] != ">" or any(aa not in "ACDEFGHIKLMNPQRSTVWY:" for aa in fasta_seq):
        raise AssertionError(f"invalid fasta:\n{fasta_str}")
    return header, fasta_seq


class TestFastaValidation:
    def test_valid_single_sequence(self):
        header, seq = validate_fasta(">test\nMKFLILLFNILCLF")
        assert header == ">test"
        assert seq == "MKFLILLFNILCLF"

    def test_valid_complex_with_colon(self):
        header, seq = validate_fasta(">complex\nMKFLILLF:NILEACLF")
        assert ":" in seq

    def test_valid_multiline_sequence(self):
        header, seq = validate_fasta(">multi\nMKFLI\nLLFNI\nLCLF")
        assert seq == "MKFLILLFNILCLF"

    def test_invalid_characters(self):
        with pytest.raises(AssertionError, match="invalid fasta"):
            validate_fasta(">bad\nMKFLI123LLF")

    def test_missing_header(self):
        with pytest.raises(AssertionError, match="invalid fasta"):
            validate_fasta("MKFLILLFNILCLF")

    def test_lowercase_rejected(self):
        with pytest.raises(AssertionError, match="invalid fasta"):
            validate_fasta(">lower\nmkflillfnilclf")

    def test_empty_sequence_is_valid(self):
        """Empty sequence after header is technically valid (no invalid chars)."""
        header, seq = validate_fasta(">empty\n")
        assert seq == ""
