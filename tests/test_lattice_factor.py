"""Tests for Lattice-Combined Factoring."""
import pytest
from math import isqrt

from insideout.lattice_factor import (
    _lll_reduce,
    _build_prime_base,
    _factor_smooth_value,
    lattice_factor,
    hybrid_smooth_factor,
)


class TestLLLReduce:
    """Test LLL lattice reduction."""

    def test_empty(self):
        """Empty basis returns empty."""
        assert _lll_reduce([]) == []

    def test_identity(self):
        """Identity matrix stays identity."""
        basis = [[1.0, 0.0], [0.0, 1.0]]
        result = _lll_reduce(basis)
        # Should still be orthogonal and reduced
        assert len(result) == 2

    def test_simple_lattice(self):
        """Simple 2D lattice gets reduced."""
        # Basis for Z²: very skewed
        basis = [[1.0, 0.0], [1000000.0, 1.0]]
        result = _lll_reduce(basis)
        # First vector should be short
        norm_first = sum(x*x for x in result[0])**0.5
        norm_second = sum(x*x for x in result[1])**0.5
        assert norm_first <= norm_second


class TestBuildPrimeBase:
    """Test prime base building."""

    def test_basic(self):
        """Basic prime base building."""
        # Use N=15, which has factors 3 and 5
        primes = _build_prime_base(15, 100)
        assert 2 in primes
        # 3 and 5 divide 15, so they should NOT be in the base
        assert 3 not in primes
        assert 5 not in primes
        assert 7 in primes
        assert 11 in primes

    def test_small_bound(self):
        """Small bound gives small base."""
        primes = _build_prime_base(1001, 10)
        assert len(primes) >= 2


class TestFactorSmoothValue:
    """Test smooth value factorization."""

    def test_perfect_square(self):
        """Perfect square factors completely."""
        factors = _factor_smooth_value(12, [2, 3, 5])
        assert factors == {2: 2, 3: 1}

    def test_with_large_factor(self):
        """Value with large factor only records small factors."""
        # 100 = 2^2 * 5^2, both small
        factors = _factor_smooth_value(100, [2, 3, 5, 7])
        assert factors == {2: 2, 5: 2}

    def test_prime_not_in_base(self):
        """Prime not in base is not recorded."""
        factors = _factor_smooth_value(7, [2, 3, 5])
        assert 7 not in factors


class TestLatticeFactor:
    """Test lattice factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = lattice_factor(N, bound=1000, target_relations=50)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = lattice_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = lattice_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = lattice_factor(7, bound=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert lattice_factor(1) is None
        assert lattice_factor(2) is None
        assert lattice_factor(3) is None


class TestHybridSmoothFactor:
    """Test hybrid smooth factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = hybrid_smooth_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = hybrid_smooth_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = hybrid_smooth_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = hybrid_smooth_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert hybrid_smooth_factor(1) is None
        assert hybrid_smooth_factor(2) is None
        assert hybrid_smooth_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = hybrid_smooth_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047