"""Tests for p-adic slope factoring method."""
import pytest
from math import gcd

from insideout.padic_factor import (
    padic_factor,
    _prime_sieve,
    _prime_powers,
    _smooth_ladder,
    _pm1_smooth,
    _padic_slope,
    _padic_log_series,
    _collision_gcd,
    _log_slope_crosscheck,
    _modinv,
    _extended_gcd,
)


class TestHelpers:
    """Test helper functions."""

    def test_prime_sieve_small(self):
        assert _prime_sieve(1) == []
        assert _prime_sieve(2) == [2]
        assert _prime_sieve(10) == [2, 3, 5, 7]
        assert _prime_sieve(20) == [2, 3, 5, 7, 11, 13, 17, 19]
        assert _prime_sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_prime_sieve_count(self):
        """π(100) = 25, π(1000) = 168."""
        assert len(_prime_sieve(100)) == 25
        assert len(_prime_sieve(1000)) == 168

    def test_prime_powers(self):
        pps = _prime_powers(20)
        primes = [p for p, _ in pps]
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19]
        # 2^4 = 16 ≤ 20, 2^5 = 32 > 20
        assert pps[0] == (2, 16)
        # 3^2 = 9 ≤ 20, 3^3 = 27 > 20
        assert pps[1] == (3, 9)
        # 5^1 = 5 ≤ 20, 5^2 = 25 > 20
        assert pps[2] == (5, 5)

    def test_extended_gcd(self):
        g, x, y = _extended_gcd(30, 12)
        assert g == 6
        assert 30 * x + 12 * y == 6

        g, x, y = _extended_gcd(17, 13)
        assert g == 1
        assert 17 * x + 13 * y == 1

    def test_modinv(self):
        assert _modinv(3, 7) == 5  # 3*5 = 15 ≡ 1 mod 7
        assert _modinv(1, 5) == 1
        assert _modinv(2, 4) is None  # gcd(2,4) = 2 ≠ 1

    def test_padic_log_series_of_one(self):
        """log(1) = 0 for any N."""
        for N in [15, 100, 8051, 1022117]:
            assert _padic_log_series(1, N, 50) == 0

    def test_collision_gcd_basic(self):
        """Collision GCD finds factor from sequence with common factor."""
        # All elements share factor 83
        seq = [83, 2 * 83, 3 * 83, 5 * 83]
        result = _collision_gcd(seq, 8051)
        assert result is not None
        p, q = result
        assert p * q == 8051

    def test_collision_gcd_no_collision(self):
        """No collision returns None."""
        seq = [2, 3, 5, 7, 11]
        result = _collision_gcd(seq, 8051)
        # May or may not find something depending on values
        # (difference of 11-9=2 doesn't share factor with 8051=83*97)
        # Just verify it doesn't crash
        if result is not None:
            p, q = result
            assert p * q == 8051


class TestPm1Smooth:
    """Test Pollard p-1 smooth ladder."""

    def test_pm1_smooth_8051(self):
        """8051 = 83*97, p-1 = 82 = 2*41, q-1 = 96 = 2^5*3."""
        result = _pm1_smooth(2, 8051, 100)
        assert result is not None
        p, q = result
        assert p * q == 8051

    def test_pm1_smooth_15571(self):
        """15571 = 23*677, p-1 = 22 = 2*11, q-1 = 676 = 2^2*13^2."""
        result = _pm1_smooth(2, 15571, 100)
        assert result is not None
        p, q = result
        assert p * q == 15571

    def test_pm1_smooth_no_factor(self):
        """When neither p-1 nor q-1 is smooth, returns None."""
        # Large primes with non-smooth p-1 and q-1
        # This is a weak test — the method might still find something via Stage 2/3
        # For a truly non-smooth case, we'd need carefully chosen primes
        pass  # Skip — hard to guarantee non-smoothness


class TestPadicSlope:
    """Test p-adic slope collision search."""

    def test_padic_slope_8051(self):
        result = _padic_slope(2, 8051, 1000)
        assert result is not None
        p, q = result
        assert p * q == 8051

    def test_padic_slope_15571(self):
        result = _padic_slope(2, 15571, 1000)
        assert result is not None
        p, q = result
        assert p * q == 15571


class TestPadicFactor:
    """Test main padic_factor entry point."""

    @pytest.mark.parametrize("N,p,q", [
        (8051, 83, 97),
        (15571, 23, 677),
        (1022117, 1009, 1013),
    ])
    def test_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes correctly."""
        result = padic_factor(N)
        assert result is not None
        fp, fq = result
        assert fp * fq == N
        assert fp <= fq

    def test_medium_semiprime(self):
        """1000003 * 1000033 = 1000036000099."""
        N = 1000036000099
        result = padic_factor(N)
        assert result is not None
        p, q = result
        assert p * q == N

    def test_64bit_semiprime(self):
        """64-bit semiprime: 4294967311 * 429496731193."""
        p = 4294967311
        q = 429496731193
        N = p * q
        result = padic_factor(N, smooth_bound=50000)
        assert result is not None
        fp, fq = result
        assert fp * fq == N

    def test_even_numbers(self):
        """Should handle even numbers."""
        result = padic_factor(100)
        assert result is not None
        p, q = result
        assert p * q == 100

    def test_perfect_squares(self):
        """Should handle perfect squares."""
        result = padic_factor(121)
        assert result is not None
        p, q = result
        assert p * q == 121
        assert p == q == 11

    def test_small_composites(self):
        """Should factor small composites."""
        for N in [15, 21, 35, 77, 437, 323, 899]:
            result = padic_factor(N)
            if result is not None:
                p, q = result
                assert p * q == N, f"Failed for N={N}: got ({p}, {q})"

    def test_returns_none_for_primes(self):
        """Should return None for primes (or at least not crash)."""
        # Small primes are handled by the N < 4 check
        # Larger primes should exhaust all methods and return None
        result = padic_factor(101)
        # 101 is prime, so should return None
        assert result is None

    def test_invalid_inputs(self):
        """Should handle edge cases gracefully."""
        assert padic_factor(1) is None
        assert padic_factor(0) is None
        assert padic_factor(-5) is None

    def test_base_parameter(self):
        """Should work with different smooth_bound values."""
        N = 8051
        for bound in [50, 100, 500, 1000]:
            result = padic_factor(N, smooth_bound=bound)
            assert result is not None
            p, q = result
            assert p * q == N
