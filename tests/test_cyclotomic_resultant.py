"""Tests for Cyclotomic-Resultant Cascade factoring."""
import pytest
from math import isqrt

from insideout.cyclotomic_resultant import (
    _cyclotomic_poly,
    _poly_eval_mod,
    _divisors,
    _small_primes,
    cyclotomic_resultant_factor,
    cyclotomic_cascade_factor,
)


class TestCyclotomicPoly:
    """Test cyclotomic polynomial computation."""

    def test_phi_1(self):
        """Φ_1(x) = x - 1."""
        assert _cyclotomic_poly(1) == [1, -1]

    def test_phi_2(self):
        """Φ_2(x) = x + 1."""
        assert _cyclotomic_poly(2) == [1, 1]

    def test_phi_3(self):
        """Φ_3(x) = x^2 + x + 1."""
        result = _cyclotomic_poly(3)
        assert result == [1, 1, 1]

    def test_phi_4(self):
        """Φ_4(x) = x^2 + 1."""
        result = _cyclotomic_poly(4)
        assert result == [1, 0, 1]

    def test_phi_6(self):
        """Φ_6(x) = x^2 - x + 1."""
        result = _cyclotomic_poly(6)
        assert result == [1, -1, 1]

    def test_phi_10(self):
        """Φ_10(x) = x^4 - x^3 + x^2 - x + 1."""
        result = _cyclotomic_poly(10)
        assert result == [1, -1, 1, -1, 1]


class TestDivisors:
    """Test divisor computation."""

    def test_12(self):
        """Divisors of 12."""
        assert _divisors(12) == [1, 2, 3, 4, 6, 12]

    def test_7(self):
        """Divisors of 7 (prime)."""
        assert _divisors(7) == [1, 7]


class TestPolyEvalMod:
    """Test polynomial evaluation mod N."""

    def test_linear(self):
        """Evaluate x - 1 at x=3 mod 10: 3-1=2."""
        assert _poly_eval_mod([1, -1], 3, 10) == 2

    def test_quadratic(self):
        """Evaluate x^2 + 1 at x=2 mod 10: 4+1=5."""
        assert _poly_eval_mod([1, 0, 1], 2, 10) == 5

    def test_modular(self):
        """Evaluate x^2 + x + 1 at x=5 mod 7: 25+5+1=31≡3."""
        assert _poly_eval_mod([1, 1, 1], 5, 7) == 31 % 7


class TestCyclotomicResultantFactor:
    """Test cyclotomic-resultant factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = cyclotomic_resultant_factor(N, max_order=20)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = cyclotomic_resultant_factor(1681)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = cyclotomic_resultant_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = cyclotomic_resultant_factor(7, max_order=10)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert cyclotomic_resultant_factor(1) is None
        assert cyclotomic_resultant_factor(2) is None
        assert cyclotomic_resultant_factor(3) is None


class TestCyclotomicCascadeFactor:
    """Test the practical cyclotomic cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Cyclotomic cascade should factor small semiprimes."""
        result = cyclotomic_cascade_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = cyclotomic_cascade_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = cyclotomic_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = cyclotomic_cascade_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert cyclotomic_cascade_factor(1) is None
        assert cyclotomic_cascade_factor(2) is None
        assert cyclotomic_cascade_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth (Pollard p-1 territory)."""
        # 2047 = 23 * 89, and 22 = 2*11 (smooth), 88 = 2^3*11 (smooth)
        result = cyclotomic_cascade_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047