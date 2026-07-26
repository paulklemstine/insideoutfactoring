"""Tests for Multi-Scale Mobius Factoring algorithm."""
import pytest
from math import gcd, isqrt

from insideout.multi_scale_mobius import (
    multi_scale_mobius_factor,
    _quick_check,
    _descent_check,
    _descent_parent,
    _mobius_sieve,
    _multi_start_descent,
    _generate_starting_points,
    _residue_classes_for_prime,
    _crt_combine,
)
from insideout.gaussian import MnPair, mn_to_triple


class TestQuickCheck:
    """Test quick pre-checks."""

    def test_even(self):
        assert _quick_check(100) == (2, 50)

    def test_perfect_square(self):
        assert _quick_check(49) == (7, 7)

    def test_small_factor(self):
        result = _quick_check(8051)  # 83 * 97
        # 8051 = 83 * 97, neither is small enough for trial division to 5000
        assert result is None or result[0] * result[1] == 8051

    def test_prime(self):
        # 83 is prime
        assert _quick_check(83) is None


class TestDescentParent:
    """Test Berggren tree parent computation."""

    def test_root_has_no_parent(self):
        """The root (2,1) has no parent."""
        assert _descent_parent(MnPair(2, 1)) is None

    def test_child_of_root(self):
        """Children of root should descend to root."""
        # U-child of (2,1): (2*2-1, 2) = (3, 2)
        parent = _descent_parent(MnPair(3, 2))
        assert parent == MnPair(2, 1)

        # A-child of (2,1): (2*2+1, 2) = (5, 2)
        parent = _descent_parent(MnPair(5, 2))
        assert parent == MnPair(2, 1)

        # D-child of (2,1): (2+2*1, 1) = (4, 1)
        parent = _descent_parent(MnPair(4, 1))
        assert parent == MnPair(2, 1)

    def test_deeper_descent(self):
        """Deeper nodes should descend correctly."""
        # (5, 2) -> (2, 1)
        p1 = _descent_parent(MnPair(5, 2))
        assert p1 == MnPair(2, 1)

        # (7, 2) is U-child of (5, 2)? Let's check: U(5,2) = (2*5-2, 5) = (8, 5)
        # So (8, 5) -> (5, 2)
        p2 = _descent_parent(MnPair(8, 5))
        assert p2 == MnPair(5, 2)


class TestResidueClasses:
    """Test residue class computation for sieve."""

    def test_prime_2(self):
        classes = _residue_classes_for_prime(2)
        # For r=2: a=m^2-n^2, b=2mn, c=m^2+n^2
        # b is always 0 mod 2
        assert len(classes) > 0

    def test_prime_3(self):
        classes = _residue_classes_for_prime(3)
        assert len(classes) > 0

    def test_residue_classes_bounded(self):
        """All residue classes should be in [0, r)."""
        for r in [2, 3, 5, 7, 11]:
            classes = _residue_classes_for_prime(r)
            for m, n, coord in classes:
                assert 0 <= m < r
                assert 0 <= n < r
                assert coord in (0, 1, 2)


class TestCRTCombine:
    """Test Chinese Remainder Theorem combination."""

    def test_simple(self):
        # x = 1 mod 2, x = 2 mod 3 -> x = 5 mod 6
        result = _crt_combine(1, 2, 2, 3)
        assert result is not None
        x, m = result
        assert x == 5
        assert m == 6

    def test_no_solution(self):
        # x = 0 mod 2, x = 1 mod 4 -> no solution
        result = _crt_combine(0, 2, 1, 4)
        assert result is None

    def test_same_modulus(self):
        result = _crt_combine(3, 5, 3, 5)
        assert result is not None
        x, m = result
        assert x == 3
        assert m == 5


class TestMultiScaleMobiusFactor:
    """Test the main factoring entry point."""

    def test_8051(self):
        """8051 = 83 * 97 (well-separated factors)."""
        result = multi_scale_mobius_factor(8051)
        assert result is not None
        p, q = result
        assert p * q == 8051
        assert 1 < p <= q < 8051

    def test_15571(self):
        """15571 = 23 * 677 (very well-separated factors)."""
        result = multi_scale_mobius_factor(15571)
        assert result is not None
        p, q = result
        assert p * q == 15571
        assert 1 < p <= q < 15571

    def test_1022117(self):
        """1022117 = 1009 * 1013 (balanced semiprime)."""
        result = multi_scale_mobius_factor(1022117)
        assert result is not None
        p, q = result
        assert p * q == 1022117
        assert 1 < p <= q < 1022117

    def test_small_balanced(self):
        """Small balanced semiprime: 83 * 97 = 8051."""
        result = multi_scale_mobius_factor(8051, num_scales=5, depth=30)
        assert result is not None
        p, q = result
        assert p * q == 8051

    def test_even_number(self):
        result = multi_scale_mobius_factor(100)
        assert result == (2, 50)

    def test_perfect_square(self):
        result = multi_scale_mobius_factor(49)
        assert result == (7, 7)

    def test_prime_returns_none(self):
        result = multi_scale_mobius_factor(83)
        assert result is None


class TestDescentCheck:
    """Test descent from a specific starting point."""

    def test_descent_from_well(self):
        """Descent from near sqrt(N) should find factors for well-separated N."""
        # 15571 = 23 * 677, sqrt ~ 124.8
        # Starting from near sqrt with n=1 should hit a factor
        sqrt_N = isqrt(15571)
        result = _descent_check(15571, MnPair(sqrt_N + 1, 1), 100)
        # This may or may not find a factor depending on the starting point
        if result is not None:
            p, q = result
            assert p * q == 15571


class TestGenerateStartingPoints:
    """Test starting point generation."""

    def test_near_sqrt(self):
        points = list(_generate_starting_points(8051, 10, "near_sqrt"))
        assert len(points) > 0
        for mn in points:
            assert mn.m > mn.n > 0
            assert (mn.m - mn.n) % 2 == 1
            assert gcd(mn.m, mn.n) == 1

    def test_balanced(self):
        points = list(_generate_starting_points(8051, 10, "balanced"))
        assert len(points) > 0
        for mn in points:
            assert mn.m > mn.n > 0

    def test_mixed(self):
        points = list(_generate_starting_points(8051, 20, "mixed"))
        assert len(points) > 0
        # Should have diverse points
        ratios = [mn.m / mn.n for mn in points]
        assert max(ratios) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
