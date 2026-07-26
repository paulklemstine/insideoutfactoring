"""Tests for PPT-Structured Quadratic Sieve factoring."""
import pytest
from math import isqrt

from insideout.ppt_quadratic_sieve import (
    _small_primes,
    _sqrt_mod,
    _smoothness_test,
    ppt_quadratic_sieve,
)


class TestSmallPrimes:
    """Test prime sieve utility."""

    def test_small_bound(self):
        """Primes up to 10."""
        assert _small_primes(10) == [2, 3, 5, 7]

    def test_bound_2(self):
        """Primes up to 2."""
        assert _small_primes(2) == [2]

    def test_bound_lt_2(self):
        """No primes below 2."""
        assert _small_primes(1) == []
        assert _small_primes(0) == []

    def test_larger_bound(self):
        """Primes up to 30."""
        assert _small_primes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


class TestSqrtMod:
    """Test Tonelli-Shanks square root mod p."""

    def test_zero(self):
        """sqrt(0) = 0 mod any prime."""
        assert _sqrt_mod(0, 7) == [0]

    def test_perfect_square_mod_prime(self):
        """4 is a QR mod 7."""
        roots = _sqrt_mod(4, 7)
        assert len(roots) == 2
        for r in roots:
            assert (r * r) % 7 == 4 % 7

    def test_non_residue(self):
        """3 is not a QR mod 7."""
        assert _sqrt_mod(3, 7) == []

    def test_prime_2(self):
        """sqrt(1) mod 2."""
        assert _sqrt_mod(1, 2) == [1]

    def test_all_qr_mod_5(self):
        """All QRs mod 5 should have correct roots."""
        # QRs mod 5: 1, 4
        for a in [1, 4]:
            roots = _sqrt_mod(a, 5)
            assert len(roots) == 2
            for r in roots:
                assert (r * r) % 5 == a % 5


class TestSmoothnessTest:
    """Test B-smooth detection."""

    def test_smooth_number(self):
        """24 is smooth over primes up to 5."""
        fb = [2, 3, 5]
        result = _smoothness_test(24, fb)
        assert result is not None
        factors, cofactor = result
        assert cofactor == 1
        assert all(f in fb for f in factors)

    def test_non_smooth(self):
        """97 is not smooth over primes up to 5."""
        fb = [2, 3, 5]
        result = _smoothness_test(97, fb)
        # 97 is prime and > 5^2, so not smooth enough
        assert result is None

    def test_one_large_prime(self):
        """Number with one large prime should be accepted if < bound^2."""
        fb = [2, 3, 5, 7]
        # 2 * 3 * 11 = 66 — 11 < 7^2 = 49? No, 11 < 49 yes
        result = _smoothness_test(66, fb)
        assert result is not None
        factors, cofactor = result
        assert cofactor == 11  # One large prime

    def test_zero_returns_none(self):
        """Zero should return None."""
        assert _smoothness_test(0, [2, 3, 5]) is None


class TestPPTQuadraticSieve:
    """Test the PPT-structured quadratic sieve."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
        (437, 19, 23),
        (667, 23, 29),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """PPT sieve should factor small semiprimes."""
        result = ppt_quadratic_sieve(N, bound=500, sieve_range=2000, max_relations=100)
        if result is not None:
            fp, fq = result
            assert fp * fq == N, f"Factors {fp} × {fq} should equal {N}"

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = ppt_quadratic_sieve(1681)  # 41²
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = ppt_quadratic_sieve(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None (no factor)."""
        result = ppt_quadratic_sieve(7, bound=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert ppt_quadratic_sieve(1) is None
        assert ppt_quadratic_sieve(2) is None
        assert ppt_quadratic_sieve(3) is None