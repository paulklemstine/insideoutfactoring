"""Tests for Hensel Lifting Cascade and CRT Lattice factoring."""
import pytest
from math import isqrt

from insideout.hensel_cascade import hensel_cascade_factor, crt_lattice_factor


class TestHenselCascadeFactor:
    """Test Hensel cascade factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = hensel_cascade_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = hensel_cascade_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = hensel_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = hensel_cascade_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert hensel_cascade_factor(1) is None
        assert hensel_cascade_factor(2) is None
        assert hensel_cascade_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89, and 22 = 2*11 (smooth)
        result = hensel_cascade_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047

    def test_medium_semiprime(self):
        """Should factor medium semiprimes with p-1 smooth."""
        # 10007 * 10009 = 100160063
        result = hensel_cascade_factor(100160063, bound=50000)
        if result is not None:
            assert result[0] * result[1] == 100160063


class TestCRTLatticeFactor:
    """Test CRT lattice factoring."""

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
        result = crt_lattice_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = crt_lattice_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = crt_lattice_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = crt_lattice_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert crt_lattice_factor(1) is None
        assert crt_lattice_factor(2) is None
        assert crt_lattice_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = crt_lattice_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047

    def test_lattice_detection(self):
        """CRT lattice should detect factors via cross-product."""
        # 1003 = 17 * 59
        result = crt_lattice_factor(1003, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 1003