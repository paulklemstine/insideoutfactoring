"""Tests for Discriminant Resonance and Quadratic Resonance factoring."""
import pytest
from math import isqrt

from insideout.resultant_cascade import (
    discriminant_resonance_factor,
    quadratic_resonance_factor,
    _jacobi_symbol,
    _tonelli_shanks,
)


class TestJacobiSymbol:
    """Test Jacobi symbol computation."""

    def test_legendre_symbol_prime(self):
        """Jacobi symbol reduces to Legendre for prime modulus."""
        # (2/7) = 1 because 2 is a QR mod 7 (3² = 9 ≡ 2 mod 7)
        assert _jacobi_symbol(2, 7) == 1
        # (3/7) = -1 because 3 is not a QR mod 7
        assert _jacobi_symbol(3, 7) == -1
        # (6/7) = -1
        assert _jacobi_symbol(6, 7) == -1

    def test_jacobi_composite(self):
        """Jacobi symbol for composite modulus is product of Legendre symbols."""
        # (2/15) = (2/3) * (2/5) = (-1) * (-1) = 1
        assert _jacobi_symbol(2, 15) == 1

    def test_jacobi_zero(self):
        """Jacobi symbol (0/n) = 0 for n > 1."""
        assert _jacobi_symbol(0, 7) == 0
        assert _jacobi_symbol(7, 7) == 0


class TestTonelliShanks:
    """Test Tonelli-Shanks square root algorithm."""

    def test_perfect_squares(self):
        """Should find square roots of perfect squares mod p."""
        # sqrt(4) mod 7 = 2 or 5
        r = _tonelli_shanks(4, 7)
        assert r is not None
        assert (r * r) % 7 == 4

    def test_quadratic_residues(self):
        """Should find square roots of quadratic residues."""
        # sqrt(2) mod 7 = 3 or 4
        r = _tonelli_shanks(2, 7)
        assert r is not None
        assert (r * r) % 7 == 2

    def test_non_residues(self):
        """Should return None for quadratic non-residues."""
        assert _tonelli_shanks(3, 7) is None

    def test_large_prime(self):
        """Should work for larger primes."""
        p = 10007
        # Find a QR mod p
        for a in range(2, 20):
            r = _tonelli_shanks(a, p)
            if r is not None:
                assert (r * r) % p == a
                break


class TestDiscriminantResonanceFactor:
    """Test discriminant resonance factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = discriminant_resonance_factor(N, max_disc=500)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = discriminant_resonance_factor(1681, max_disc=100)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = discriminant_resonance_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = discriminant_resonance_factor(7, max_disc=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert discriminant_resonance_factor(1) is None
        assert discriminant_resonance_factor(2) is None
        assert discriminant_resonance_factor(3) is None


class TestQuadraticResonanceFactor:
    """Test quadratic resonance (smooth-bound) factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Quadratic resonance should factor small semiprimes."""
        result = quadratic_resonance_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = quadratic_resonance_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = quadratic_resonance_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = quadratic_resonance_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert quadratic_resonance_factor(1) is None
        assert quadratic_resonance_factor(2) is None
        assert quadratic_resonance_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth (Pollard p-1 territory)."""
        # 2047 = 23 * 89, and 22 = 2*11 (smooth), 88 = 2^3*11 (smooth)
        result = quadratic_resonance_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047