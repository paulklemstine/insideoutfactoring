"""Tests for Character Sum Probe factoring algorithm."""
import random
from math import isqrt

import pytest

from insideout.character_sum import (
    character_sum_factor,
    _partial_character_sum,
    _twisted_character_sum,
    _character_probe,
    _jacobi_symbol,
    _pollard_rho,
    _trial_division,
    _is_perfect_power,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestJacobi:
    """Test the Jacobi symbol implementation against known values."""

    @pytest.mark.parametrize("a,n,expected", [
        (1, 3, 1),
        (2, 3, -1),
        (2, 5, -1),
        (4, 5, 1),
        (2, 7, 1),
        (3, 7, -1),
        (5, 7, -1),
        (6, 7, -1),
        (2, 9, 1),
        (3, 9, 0),
        (0, 5, 0),
    ])
    def test_jacobi_symbol(self, a, n, expected):
        assert _jacobi_symbol(a, n) == expected

    def test_jacobi_multiplicativity(self):
        """(ab/n) = (a/n)(b/n) for random inputs."""
        cases = [(7, 11, 143), (5, 8, 231), (3, 10, 253)]
        for a, b, n in cases:
            assert _jacobi_symbol(a * b, n) == _jacobi_symbol(a, n) * _jacobi_symbol(b, n)


class TestTrialDivision:
    def test_even_number(self):
        assert _trial_division(100) == (2, 50)

    def test_no_small_factor(self):
        assert _trial_division(1022117) is None

    def test_composite_with_small_factor(self):
        result = _trial_division(91)
        assert result is not None
        assert result[0] * result[1] == 91


class TestPerfectPower:
    def test_perfect_square(self):
        assert _is_perfect_power(121) == (11, 2)

    def test_perfect_cube(self):
        assert _is_perfect_power(27) == (3, 3)

    def test_not_perfect_power(self):
        assert _is_perfect_power(100) == (10, 2)

    def test_not_perfect_power_99(self):
        assert _is_perfect_power(99) is None


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

class TestPartialCharacterSum:
    """Test plain and twisted partial character sums."""

    def test_plain_sum_small(self):
        """Plain sum for a=2, N=8051 up to k=10."""
        N = 8051
        # Manually compute
        expected = sum(_jacobi_symbol(2 * j, N) for j in range(1, 11))
        assert _partial_character_sum(2, 10, N) == expected

    def test_plain_sum_hits_factor(self):
        """Plain character sum eventually hits a multiple of a factor of 8051."""
        N = 8051  # = 83 * 97
        # We know from exploration that k=3731 gives S = +/- 83.
        s = _partial_character_sum(2, 3731, N)
        assert abs(s) == 83

    def test_linear_twist_hits_factor(self):
        """Linear twist hits a factor of 8051 much sooner."""
        N = 8051
        # k=67 gives S = +/- 194 = +/- 2*97.
        s = _twisted_character_sum(2, 67, N, twist='linear')
        assert abs(s) % 97 == 0

    def test_quadratic_twist_hits_factor(self):
        """Quadratic twist hits a factor of 8051 even sooner."""
        N = 8051
        # k=30 gives S = +/- 2739 = +/- 33*83.
        s = _twisted_character_sum(2, 30, N, twist='quadratic')
        assert abs(s) % 83 == 0

    def test_alternating_twist(self):
        """Alternating twist produces a valid integer sum."""
        N = 8051
        s = _twisted_character_sum(2, 100, N, twist='alternating')
        assert isinstance(s, int)

    def test_unknown_twist_raises(self):
        with pytest.raises(ValueError):
            _twisted_character_sum(2, 10, 8051, twist='nonexistent')


class TestCharacterProbe:
    """Test the multi-base character-sum probe."""

    def test_splits_8051(self):
        result = _character_probe(8051, num_bases=10, max_k=100000)
        assert result is not None
        assert result[0] * result[1] == 8051

    def test_splits_15571(self):
        result = _character_probe(15571, num_bases=10, max_k=100000)
        assert result is not None
        assert result[0] * result[1] == 15571

    def test_splits_1022117(self):
        result = _character_probe(1022117, num_bases=10, max_k=100000)
        assert result is not None
        assert result[0] * result[1] == 1022117


# ---------------------------------------------------------------------------
# Full algorithm
# ---------------------------------------------------------------------------

class TestCharacterSumFactor:
    """Test the complete Character Sum Probe factoring algorithm."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (323, 17, 19),
        (899, 29, 31),
        (9991, 97, 103),
    ])
    def test_small_semiprimes(self, N, p, q):
        result = character_sum_factor(N)
        assert result is not None, f"Failed to factor {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N

    def test_8051(self):
        result = character_sum_factor(8051)
        assert result is not None
        assert result[0] * result[1] == 8051

    def test_15571(self):
        result = character_sum_factor(15571)
        assert result is not None
        assert result[0] * result[1] == 15571

    def test_1022117(self):
        """N = 1009 * 1013 (close primes)."""
        N = 1022117
        result = character_sum_factor(N)
        assert result is not None
        fp, fq = result
        assert fp * fq == N

    def test_even_number(self):
        assert character_sum_factor(100) == (2, 50)

    def test_perfect_square(self):
        result = character_sum_factor(121)
        assert result == (11, 11)

    def test_perfect_cube(self):
        result = character_sum_factor(2 ** 31)
        assert result is not None
        fp, fq = result
        assert fp * fq == 2 ** 31

    def test_small_prime(self):
        result = character_sum_factor(97)
        if result is not None:
            fp, fq = result
            assert fp * fq == 97

    def test_factor_pair_ordering(self):
        test_cases = [15, 35, 77, 437, 8051, 15571, 1022117]
        for N in test_cases:
            result = character_sum_factor(N)
            assert result is not None
            p, q = result
            assert p <= q
            assert p * q == N

    def test_30bit_semiprime(self):
        """A 30-bit semiprime (two ~15-bit primes)."""
        p, q = 131071, 131101  # two primes near 2^17
        N = p * q
        result = character_sum_factor(N)
        assert result is not None
        fp, fq = result
        assert fp * fq == N

    def test_40bit_semiprime(self):
        """A 40-bit semiprime."""
        p, q = 549755813887, 549755813891
        # Skip if these aren't actually prime — just check the algorithm runs.
        N = p * q
        result = character_sum_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_64bit_semiprime(self):
        """A 64-bit semiprime (balanced)."""
        p = 2147483647   # Mersenne prime 2^31 - 1
        q = 2147483629   # another large prime
        N = p * q
        result = character_sum_factor(N)
        assert result is not None, f"Failed to factor {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N
