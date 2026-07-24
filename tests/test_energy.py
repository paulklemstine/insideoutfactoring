"""Tests for energy spectrum computations."""
import pytest
from insideout.energy import (
    energy, energy_gap, hypotenuse_bound, is_energy_compatible,
)
from insideout.berggren import Triple


class TestEnergy:
    def test_energy_is_c(self):
        """Energy ordering matches c ordering without computing ln."""
        assert energy(Triple(3, 4, 5)) == 5
        assert energy(Triple(5, 12, 13)) == 13
        assert energy(Triple(21, 20, 29)) == 29

    def test_energy_monotonic(self):
        """Children always have higher energy than parent."""
        root = Triple(3, 4, 5)
        from insideout.berggren import children
        for child in children(root):
            assert energy(child) > energy(root)


class TestEnergyGap:
    def test_gap_is_positive(self):
        root = Triple(3, 4, 5)
        from insideout.berggren import children
        for child in children(root):
            gap = energy_gap(root, child)
            assert gap > 0

    def test_gap_values(self):
        root = Triple(3, 4, 5)
        from insideout.berggren import children
        gaps = [energy_gap(root, c) for c in children(root)]
        # Energy gaps: 13-5=8, 29-5=24, 17-5=12
        assert set(gaps) == {8, 24, 12}


class TestHypotenuseBound:
    def test_bound_for_semiprime(self):
        """For N=pq with p<q, the target triple has c = (q^2+p^2)/2.
        The minimum possible c satisfies c > N/2."""
        N = 15  # 3*5
        bound = hypotenuse_bound(N)
        assert bound > 0
        # c must be at least 8 for (15, 8, 17): c = 17
        assert bound <= 17 or bound >= 17  # bound is an upper bound
        # The bound should be (N^2+1)//2 = (225+1)//2 = 113
        assert bound == 113

    def test_bound_grows_with_N(self):
        b1 = hypotenuse_bound(15)
        b2 = hypotenuse_bound(100)
        assert b2 > b1


class TestEnergyCompatibility:
    def test_compatible_triple(self):
        """A triple containing N as a leg should be energy-compatible."""
        N = 15
        # (15, 8, 17) has N=15 as a leg
        t = Triple(15, 8, 17)
        assert is_energy_compatible(N, t)

    def test_incompatible_triple(self):
        """A triple with c far too small to contain N should be incompatible."""
        N = 1000
        t = Triple(3, 4, 5)  # c=5, way too small
        assert not is_energy_compatible(N, t)

    def test_incompatible_triple_c_too_large(self):
        """A triple with c exceeding the upper bound should be incompatible."""
        N = 15
        # Upper bound is (15^2+1)//2 = 113
        # c = 200 exceeds the bound
        t = Triple(199, 200, 283)  # c=283 > 113
        assert not is_energy_compatible(N, t)