"""Unit tests for calc_ipsae from modal_af2rank.py.

calc_ipsae is defined inside `with image.imports():` so we extract and test
a copy of the function here.
"""

import numpy as np
import pytest


def calc_ipsae(pae_matrix: np.ndarray, chain_ids: np.ndarray, pae_cutoff: float = 10.0) -> float:
    """Copy of calc_ipsae from modal_af2rank.py for testing."""
    def ptm_func(x, d0):
        return 1.0 / (1 + (x / d0) ** 2.0)

    def calc_d0(L):
        L = float(max(27, L))
        return max(1.0, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)

    unique_chains = np.unique(chain_ids)
    if len(unique_chains) < 2:
        return 0.0

    ipsae_values = []

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            chain1_mask = chain_ids == chain1
            chain2_mask = chain_ids == chain2

            unique_res_chain1 = set()
            unique_res_chain2 = set()

            for i in np.where(chain1_mask)[0]:
                valid = chain2_mask & (pae_matrix[i] < pae_cutoff)
                if valid.any():
                    unique_res_chain1.add(i)
                    for j in np.where(valid)[0]:
                        unique_res_chain2.add(j)

            n0dom = len(unique_res_chain1) + len(unique_res_chain2)
            if n0dom == 0:
                continue

            d0dom = calc_d0(n0dom)

            ipsae_byres = []
            for i in np.where(chain1_mask)[0]:
                valid = chain2_mask & (pae_matrix[i] < pae_cutoff)
                if valid.any():
                    ptm_vals = ptm_func(pae_matrix[i, valid], d0dom)
                    ipsae_byres.append(ptm_vals.mean())

            if ipsae_byres:
                ipsae_values.append(max(ipsae_byres))

    return max(ipsae_values) if ipsae_values else 0.0


class TestCalcIpsae:
    def test_single_chain_returns_zero(self):
        """Single chain should return 0.0 (no interface)."""
        pae = np.full((5, 5), 3.0)
        chains = np.array([0, 0, 0, 0, 0])
        assert calc_ipsae(pae, chains) == 0.0

    def test_two_chains_good_pae(self):
        """Two chains with low PAE at interface should give high score."""
        n = 10
        pae = np.full((n, n), 2.0)
        chains = np.array([0] * 5 + [1] * 5)
        score = calc_ipsae(pae, chains)
        assert score > 0.1, f"Expected positive ipSAE for good interface, got {score}"

    def test_two_chains_bad_pae(self):
        """Two chains with high PAE at interface should give low score."""
        n = 10
        pae = np.full((n, n), 2.0)
        # Make interchain PAE very high
        pae[:5, 5:] = 25.0
        pae[5:, :5] = 25.0
        chains = np.array([0] * 5 + [1] * 5)
        score = calc_ipsae(pae, chains)
        assert score == 0.0, f"Expected 0.0 for bad interface, got {score}"

    def test_all_pae_above_cutoff(self):
        """If all interchain PAE is above cutoff, score should be 0.0."""
        n = 10
        pae = np.full((n, n), 15.0)  # all above default cutoff of 10.0
        chains = np.array([0] * 5 + [1] * 5)
        assert calc_ipsae(pae, chains) == 0.0

    def test_custom_cutoff(self):
        """Custom cutoff should change which contacts are considered."""
        n = 10
        pae = np.full((n, n), 8.0)
        chains = np.array([0] * 5 + [1] * 5)

        # With default cutoff (10.0), PAE=8 is below cutoff → score > 0
        score_default = calc_ipsae(pae, chains, pae_cutoff=10.0)
        assert score_default > 0.0

        # With strict cutoff (5.0), PAE=8 is above cutoff → score = 0
        score_strict = calc_ipsae(pae, chains, pae_cutoff=5.0)
        assert score_strict == 0.0
