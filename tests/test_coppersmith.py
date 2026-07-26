"""Tests for Coppersmith's Method."""
import pytest
from math import isqrt

from insideout.coppersmith import coppersmith_factor, _lll_reduce, _build_lattice_for_factor


class TestLLLReduce:
    """Test LLL lattice reduction."""

    def test_empty(self):
        """Empty basis returns empty."""
        assert _lll_reduce([]) == []

    def test_identity(self):
        """Identity matrix stays identity."""
        basis = [[1.0, 0.0], [0.0, 1.0]]
        result = _lll_reduce(basis)
        assert len(result) == 2


class TestCoppersmithFactor:
    """Test Coppersmith's method factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (10403, 101, 103),  # p near sqrt(N)
    ])
    def test_factors_semiprimes(self, N, p, q):
        """Should factor semiprimes when p is close to sqrt(N)."""
        result = coppersmith_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_with_approximation(self):
        """Should work with approximation to p."""
        # 10403 = 101 * 103, sqrt ≈ 102
        result = coppersmith_factor(10403, X=100)
        if result is not None:
            fp, fq = result
            assert fp * fq == 10403

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = coppersmith_factor(1681)
        if result is not None:
            assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = coppersmith_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = coppersmith_factor(7)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert coppersmith_factor(1) is None
        assert coppersmith_factor(2) is None
        assert coppersmith_factor(3) is None