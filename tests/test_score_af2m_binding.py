"""Unit tests for score_af2m_binding from modal_alphafold.py."""

import numpy as np
import pytest

# Import the function directly — it's defined at module scope
from modal_alphafold import score_af2m_binding


class TestScoreAf2mBinding:
    def test_single_binder(self):
        """Single binder with known arrays."""
        target_len = 5
        binder_len = 3
        total = target_len + binder_len

        af2m_dict = {
            "plddt": [80.0] * target_len + [90.0] * binder_len,
            "pae": np.full((total, total), 5.0).tolist(),
        }
        result = score_af2m_binding(af2m_dict, target_len, [binder_len])

        assert result["plddt_target"] == pytest.approx(80.0)
        assert result["plddt_binder"][0] == pytest.approx(90.0)
        assert "ipae" in result
        assert 0 in result["ipae"]

    def test_multiple_binders(self):
        """Two binders should produce scores for both."""
        target_len = 4
        binders_len = [3, 2]
        total = target_len + sum(binders_len)

        af2m_dict = {
            "plddt": [70.0] * target_len + [85.0] * 3 + [95.0] * 2,
            "pae": np.full((total, total), 8.0).tolist(),
        }
        result = score_af2m_binding(af2m_dict, target_len, binders_len)

        assert 0 in result["plddt_binder"] and 1 in result["plddt_binder"]
        assert result["plddt_binder"][0] == pytest.approx(85.0)
        assert result["plddt_binder"][1] == pytest.approx(95.0)
        assert 0 in result["ipae"] and 1 in result["ipae"]

    def test_perfect_prediction(self):
        """Perfect confidence should yield low PAE scores."""
        target_len = 4
        binder_len = 3
        total = target_len + binder_len

        af2m_dict = {
            "plddt": [100.0] * total,
            "pae": np.zeros((total, total)).tolist(),
        }
        result = score_af2m_binding(af2m_dict, target_len, [binder_len])

        assert result["pae_target"] == pytest.approx(0.0)
        assert result["pae_binder"][0] == pytest.approx(0.0)
        assert result["ipae"][0] == pytest.approx(0.0)

    def test_bad_prediction(self):
        """Bad prediction should yield high PAE scores."""
        target_len = 4
        binder_len = 3
        total = target_len + binder_len

        af2m_dict = {
            "plddt": [30.0] * total,
            "pae": np.full((total, total), 25.0).tolist(),
        }
        result = score_af2m_binding(af2m_dict, target_len, [binder_len])

        assert result["ipae"][0] == pytest.approx(25.0)
        assert result["plddt_target"] == pytest.approx(30.0)

    def test_length_mismatch_assertion(self):
        """Mismatched lengths should raise AssertionError."""
        af2m_dict = {
            "plddt": [80.0] * 5,
            "pae": np.full((5, 5), 5.0).tolist(),
        }
        with pytest.raises(AssertionError):
            score_af2m_binding(af2m_dict, target_len=3, binders_len=[3])
