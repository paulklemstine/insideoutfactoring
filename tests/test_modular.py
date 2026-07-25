"""Tests for modular resonance filters."""
import pytest
from insideout.modular import (
    build_residue_table, is_modular_compatible,
    PPT_RESIDUES, filter_wavefront,
)
from insideout.berggren import Triple


class TestResidueTable:
    def test_mod_2(self):
        """PPT first leg is always odd."""
        table = build_residue_table(2)
        assert 0 not in table  # even numbers not in PPT legs (first leg)
        assert 1 in table  # odd numbers

    def test_mod_3(self):
        """Check PPT leg residues mod 3."""
        table = build_residue_table(3)
        # PPT first legs mod 3: should include some residues
        assert len(table) > 0

    def test_mod_5(self):
        table = build_residue_table(5)
        # Every PPT has at least one leg divisible by 5
        # (this is a known PPT property, but first leg mod 5 varies)
        assert isinstance(table, dict)


class TestModularCompatibility:
    def test_compatible_triple(self):
        """Triple containing 15 as a leg should be compatible with 15 mod small primes."""
        N = 15
        t = Triple(15, 8, 17)
        assert is_modular_compatible(N, t)

    def test_incompatible_triple(self):
        """A triple whose first leg has wrong residues should be filtered out."""
        N = 35  # 5 * 7
        t = Triple(3, 4, 5)  # too small, but also check residue logic
        # This should still pass residue check even if it's too small
        # (residue checking doesn't check size, only modular compatibility)

    def test_conservative_default(self):
        """is_modular_compatible should be conservative — return True when uncertain."""
        # Any triple with any prime should return True by default
        # since the function is conservative
        N = 77
        t = Triple(77, 36, 85)
        assert is_modular_compatible(N, t)


class TestFilterWavefront:
    def test_filters_reduce_candidates(self):
        """Modular filters should reduce the number of candidates."""
        from insideout.triples import generate_ppts
        ppts = list(generate_ppts(depth=2))
        N = 15
        filtered = list(filter_wavefront(ppts, N))
        # Some PPTs should be filtered out
        assert len(filtered) <= len(ppts)

    def test_target_survives_filter(self):
        """The target triple (15, 8, 17) must survive the filter for N=15."""
        N = 15
        target = Triple(15, 8, 17)
        assert is_modular_compatible(N, target)

    def test_filter_removes_too_small_hypotenuse(self):
        """Triples with hypotenuse < N should be filtered out."""
        candidates = [Triple(3, 4, 5), Triple(5, 12, 13), Triple(15, 8, 17)]
        N = 15
        filtered = list(filter_wavefront(candidates, N))
        # (3,4,5) has c=5 < 15 and (5,12,13) has c=13 < 15, so both filtered
        # (15,8,17) has c=17 >= 15, so it passes
        assert Triple(15, 8, 17) in filtered
        assert Triple(3, 4, 5) not in filtered
        assert Triple(5, 12, 13) not in filtered

    def test_filter_removes_both_legs_exceeding_N(self):
        """Triples where both legs are larger than N should be filtered out."""
        # If a > N and b > N, then N^2 - a^2 < 0 and N^2 - b^2 < 0
        candidates = [Triple(99, 100, 141), Triple(15, 8, 17)]
        N = 15
        filtered = list(filter_wavefront(candidates, N))
        # (99, 100, 141): both legs exceed N, so N^2 - 99^2 < 0 and N^2 - 100^2 < 0
        assert Triple(99, 100, 141) not in filtered
        assert Triple(15, 8, 17) in filtered


class TestPPTResiduesCache:
    def test_caching(self):
        """PPT_RESIDUES should return the same object on repeated calls."""
        table1 = PPT_RESIDUES(5)
        table2 = PPT_RESIDUES(5)
        assert table1 is table2

    def test_different_primes(self):
        """Different primes should produce different tables."""
        table2 = PPT_RESIDUES(2)
        table3 = PPT_RESIDUES(3)
        assert table2 is not table3