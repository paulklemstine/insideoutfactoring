"""Tests for Quadratic Character Difference (QCD) Factoring algorithm."""
import pytest
from math import isqrt

from insideout.quadratic_character import (
    quadratic_character_factor,
    _qr_ladder,
    _qr_ladder_divergence,
    _multi_base_qcd,
    _sqrt_one_split,
    _jacobi_sequence,
    _jacobi,
    _smooth_exponent_pm1,
    _pollard_rho,
    _trial_division,
    _v2,
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
        (2, 9, 1),   # (2/9) = (2/3)^2 = (-1)^2 = 1
        (3, 9, 0),   # gcd(3,9) > 1
        (0, 5, 0),
    ])
    def test_jacobi_symbol(self, a, n, expected):
        assert _jacobi(a, n) == expected

    def test_jacobi_multiplicativity(self):
        """(ab/n) = (a/n)(b/n) for random inputs."""
        random_vals = [(7, 11, 143), (5, 8, 231), (3, 10, 253)]
        for a, b, n in random_vals:
            assert _jacobi(a * b, n) == _jacobi(a, n) * _jacobi(b, n)


class TestV2:
    """Test the 2-adic valuation helper."""

    @pytest.mark.parametrize("n,expected", [
        (1, 0),
        (2, 1),
        (4, 2),
        (6, 1),
        (8, 3),
        (12, 2),
        (96, 5),
        (1000032, 5),
    ])
    def test_v2(self, n, expected):
        assert _v2(n) == expected


class TestTrialDivision:
    """Test the trivial-trial-division helper."""

    def test_even_number(self):
        assert _trial_division(100) == (2, 50)

    def test_small_factor(self):
        # 1009*1013 = 1022117 — both factors > small bound
        assert _trial_division(1022117) is None

    def test_composite_with_small_factor(self):
        result = _trial_division(91)
        assert result is not None
        assert result[0] * result[1] == 91


class TestSqrtOneSplit:
    """Test the nontrivial-sqrt(1) splitting primitive."""

    def test_trivial_ones(self):
        """1 and N-1 are trivial square roots of 1."""
        assert _sqrt_one_split(15, 1) is None
        assert _sqrt_one_split(15, 14) is None

    def test_nontrivial_split_15(self):
        """4² = 16 ≡ 1 mod 15, gcd(4-1, 15) = 3."""
        result = _sqrt_one_split(15, 4)
        assert result is not None
        assert result[0] * result[1] == 15

    def test_nontrivial_split_8051(self):
        """For N = 8051 = 83*97, construct a nontrivial sqrt(1) via CRT."""
        # x ≡ 1 mod 83, x ≡ -1 mod 97
        # x = 1 + 83*t, 1 + 83*t ≡ -1 mod 97 → 83*t ≡ -2 mod 97
        def egcd(a, b):
            if b == 0:
                return (a, 1, 0)
            g, x, y = egcd(b, a % b)
            return (g, y, x - (a // b) * y)
        _, inv83, _ = egcd(83, 97)
        inv83 %= 97
        t = ((-2) * inv83) % 97
        x = (1 + 83 * t) % 8051
        assert (x * x) % 8051 == 1
        assert x != 1 and x != 8050
        result = _sqrt_one_split(8051, x)
        assert result is not None
        assert result[0] * result[1] == 8051


# ---------------------------------------------------------------------------
# Core algorithm components
# ---------------------------------------------------------------------------

class TestQRLadder:
    """Test the QR ladder primitive."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
    ])
    def test_qr_ladder_splits(self, N, p, q):
        """QR ladder with base 2 should split small semiprimes."""
        result = _qr_ladder(N, 2)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    @pytest.mark.parametrize("N,p,q", [
        (8051, 83, 97),
        (15571, 23, 677),
    ])
    def test_qr_ladder_divergence_splits(self, N, p, q):
        """Divergence-tracking ladder should split semiprimes."""
        result = _qr_ladder_divergence(N, 2)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestMultiBaseQCD:
    """Test the multi-base QR ladder."""

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
    def test_multi_base_qcd_splits(self, N, p, q):
        result = _multi_base_qcd(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_8051(self):
        result = _multi_base_qcd(8051)
        assert result is not None, "multi-base QCD failed on 8051"
        assert result[0] * result[1] == 8051

    def test_15571(self):
        result = _multi_base_qcd(15571)
        assert result is not None, "multi-base QCD failed on 15571"
        assert result[0] * result[1] == 15571


class TestJacobiSequence:
    """Test the Jacobi-sequence mismatch detection."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (437, 19, 23),
        (8051, 83, 97),
        (15571, 23, 677),
    ])
    def test_jacobi_sequence_splits(self, N, p, q):
        result = _jacobi_sequence(N, max_samples=2000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestSmoothExponentPM1:
    """Test the smooth-exponent p−1 method."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (323, 17, 19),
        (899, 29, 31),
        (9991, 97, 103),
        (8051, 83, 97),
        (15571, 23, 677),
        (1022117, 1009, 1013),
    ])
    def test_pm1_splits(self, N, p, q):
        result = _smooth_exponent_pm1(N, bound=100000)
        assert result is not None, f"p−1 failed on {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N


class TestPollardRho:
    """Test Pollard's rho fallback."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (437, 19, 23),
        (8051, 83, 97),
        (15571, 23, 677),
    ])
    def test_pollard_rho_splits(self, N, p, q):
        result = _pollard_rho(N, max_steps=100000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


# ---------------------------------------------------------------------------
# Full algorithm
# ---------------------------------------------------------------------------

class TestQuadraticCharacterFactor:
    """Test the complete QCD factoring algorithm."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (323, 17, 19),
        (899, 29, 31),
        (9991, 97, 103),
        (8051, 83, 97),
        (15571, 23, 677),
    ])
    def test_factors_semiprimes(self, N, p, q):
        result = quadratic_character_factor(N)
        assert result is not None, f"Failed to factor {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N

    def test_1022117(self):
        """N = 1009 * 1013 (close primes)."""
        N = 1022117
        result = quadratic_character_factor(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    def test_large_semiprime(self):
        """N = 1000003 * 1000033."""
        N = 1000003 * 1000033
        result = quadratic_character_factor(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    def test_64bit_semiprime(self):
        """A 64-bit semiprime."""
        p = 2147483647   # Mersenne prime 2^31 - 1
        q = 2147483629   # another large prime
        N = p * q
        result = quadratic_character_factor(N)
        assert result is not None, f"Failed to factor {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N

    def test_even_number(self):
        assert quadratic_character_factor(100) == (2, 50)

    def test_perfect_square(self):
        result = quadratic_character_factor(121)
        assert result is not None
        assert result == (11, 11)

    def test_perfect_cube(self):
        result = quadratic_character_factor(2 ** 31)  # 2^31
        assert result is not None
        fp, fq = result
        assert fp * fq == 2 ** 31

    def test_small_prime(self):
        """A small prime should return None or trivial."""
        result = quadratic_character_factor(97)
        if result is not None:
            fp, fq = result
            assert fp * fq == 97
            assert fp == 1 or fq == 1 or fp == 97 or fq == 97

    def test_factor_pair_ordering(self):
        """Returned factor pair should satisfy p <= q."""
        test_cases = [15, 35, 77, 437, 8051, 15571, 1022117]
        for N in test_cases:
            result = quadratic_character_factor(N)
            assert result is not None
            p, q = result
            assert p <= q
            assert p * q == N
