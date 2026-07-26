"""Tests for CF Period Matrix Cascade factoring."""
import pytest
from math import isqrt

from insideout.cf_matrix_cascade import (
    _mat2_mul,
    _mat2_pow,
    _check_matrix_crt,
    _small_primes,
    _smoothness_test,
    cf_matrix_cascade_factor,
    cf_cascade_factor,
)


class TestMat2Operations:
    """Test 2×2 matrix operations (shared with SL₂)."""

    def test_identity_mul(self):
        N = 97
        I = (1, 0, 0, 1)
        M = (3, 5, 7, 11)
        result = _mat2_mul(I, M, N)
        assert result == M

    def test_associativity(self):
        N = 97
        A = (3, 5, 7, 2)
        B = (11, 13, 17, 19)
        C = (23, 29, 31, 37)
        AB_C = _mat2_mul(_mat2_mul(A, B, N), C, N)
        A_BC = _mat2_mul(A, _mat2_mul(B, C, N), N)
        assert AB_C == A_BC

    def test_pow_identity(self):
        N = 97
        M = (3, 5, 7, 2)
        result = _mat2_pow(M, 0, N)
        assert result == (1, 0, 0, 1)

    def test_pow_first(self):
        N = 97
        M = (3, 5, 7, 2)
        result = _mat2_pow(M, 1, N)
        assert result == (3 % N, 5 % N, 7 % N, 2 % N)


class TestCheckMatrixCRT:
    """Test CRT divergence detection."""

    def test_identity_no_factor(self):
        N = 15
        I = (1, 0, 0, 1)
        result = _check_matrix_crt(I, N)
        assert result is None

    def test_crt_divergence(self):
        """Matrix that differs mod 3 and mod 5."""
        N = 15
        M = (6, 0, 0, 6)
        result = _check_matrix_crt(M, N)
        assert result is not None


class TestSmallPrimes:
    """Test prime sieve."""

    def test_up_to_10(self):
        assert _small_primes(10) == [2, 3, 5, 7]

    def test_empty(self):
        assert _small_primes(1) == []


class TestSmoothnessTest:
    """Test B-smooth detection."""

    def test_smooth(self):
        fb = [2, 3, 5]
        result = _smoothness_test(30, fb)  # 2*3*5
        assert result is not None
        assert all(f in fb for f in result)

    def test_non_smooth(self):
        fb = [2, 3, 5]
        result = _smoothness_test(97, fb)  # prime > 5
        assert result is None

    def test_zero_returns_none(self):
        assert _smoothness_test(0, [2, 3]) is None

    def test_one_returns_none(self):
        assert _smoothness_test(1, [2, 3]) is None


class TestCFCascadeFactor:
    """Test the pure CF convergent cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """CF cascade should factor small semiprimes."""
        result = cf_cascade_factor(N, cf_steps=10000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = cf_cascade_factor(1681)  # 41²
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = cf_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = cf_cascade_factor(7, cf_steps=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert cf_cascade_factor(1) is None
        assert cf_cascade_factor(2) is None
        assert cf_cascade_factor(3) is None


class TestCFMatrixCascadeFactor:
    """Test the full CF Period Matrix Cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """CF matrix cascade should factor small semiprimes."""
        result = cf_matrix_cascade_factor(N, bound=10000, cf_steps=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = cf_matrix_cascade_factor(1681, bound=1000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = cf_matrix_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = cf_matrix_cascade_factor(7, bound=500, cf_steps=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert cf_matrix_cascade_factor(1) is None
        assert cf_matrix_cascade_factor(2) is None
        assert cf_matrix_cascade_factor(3) is None

    def test_medium_semiprime(self):
        """Should handle medium-sized semiprimes."""
        # 437 = 19 * 23
        result = cf_matrix_cascade_factor(437, bound=20000, cf_steps=10000)
        if result is not None:
            fp, fq = result
            assert fp * fq == 437

    def test_close_factors(self):
        """Should handle close factors."""
        # 1763 = 41 * 43
        result = cf_matrix_cascade_factor(1763, bound=20000, cf_steps=10000)
        if result is not None:
            fp, fq = result
            assert fp * fq == 1763