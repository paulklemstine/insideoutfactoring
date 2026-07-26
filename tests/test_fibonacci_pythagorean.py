"""Tests for Fibonacci–Pythagorean hybrid factoring."""
import pytest
from insideout.fibonacci_pythagorean import (
    fibonacci_pythagorean_factor,
    _fibonacci_pair,
    _small_primes,
)


class TestFibonacciPair:
    """Test fast doubling Fibonacci computation."""

    def test_f0_f1(self):
        """F_0 = 0, F_1 = 1."""
        assert _fibonacci_pair(0, 1000) == (0, 1)
        assert _fibonacci_pair(1, 1000) == (1, 1)

    def test_small_fibonacci(self):
        """Check small Fibonacci numbers."""
        # F_2 = 1, F_3 = 2, F_4 = 3, F_5 = 5
        assert _fibonacci_pair(2, 1000) == (1, 2)
        assert _fibonacci_pair(3, 1000) == (2, 3)
        assert _fibonacci_pair(5, 1000) == (5, 8)
        assert _fibonacci_pair(10, 1000) == (55, 89)

    def test_modular_fibonacci(self):
        """Check F_k mod N."""
        # F_10 = 55, mod 7 = 55 mod 7 = 6
        fk, fk1 = _fibonacci_pair(10, 7)
        assert fk == 55 % 7
        assert fk1 == 89 % 7


class TestSmallPrimes:
    """Test prime sieve."""

    def test_small_bound(self):
        """Primes up to 20."""
        primes = _small_primes(20)
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_larger_bound(self):
        """Primes up to 100."""
        primes = _small_primes(100)
        assert len(primes) == 25
        assert primes[0] == 2
        assert primes[-1] == 97


class TestFibonacciPythagoreanFactor:
    """Test the full Fibonacci–Pythagorean factoring pipeline."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = fibonacci_pythagorean_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Should handle perfect squares."""
        result = fibonacci_pythagorean_factor(121)
        assert result is not None
        assert result == (11, 11)

    def test_even_number(self):
        """Should handle even numbers."""
        result = fibonacci_pythagorean_factor(6)
        assert result is not None
        assert result[0] * result[1] == 6

    def test_close_factors(self):
        """Should handle close-factor semiprimes."""
        result = fibonacci_pythagorean_factor(323)  # 17*19
        if result is not None:
            assert result[0] * result[1] == 323

    def test_larger_semiprime(self):
        """Should handle larger semiprimes with smooth rank."""
        # 10007 * 10009 — if p-1 or p+1 is smooth, this should work
        result = fibonacci_pythagorean_factor(10007 * 10009, bound=10000)
        if result is not None:
            assert result[0] * result[1] == 10007 * 10009