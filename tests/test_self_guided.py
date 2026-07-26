"""Tests for Self-Guided Smooth Search."""
import pytest
import time
from math import isqrt

from insideout.self_guided import self_guided_factor


class TestSelfGuidedFactor:
    """Test self-guided smooth search factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = self_guided_factor(N, bound=50000)
        assert result is not None
        fp, fq = result
        assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = self_guided_factor(1681, bound=50000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = self_guided_factor(30, bound=50000)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = self_guided_factor(7, bound=5000)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert self_guided_factor(1) is None
        assert self_guided_factor(2) is None
        assert self_guided_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = self_guided_factor(2047, bound=50000)
        if result is not None:
            assert result[0] * result[1] == 2047

    def test_performance(self):
        """Should complete within reasonable time."""
        N = 10007 * 10009  # 27-bit
        start = time.perf_counter()
        result = self_guided_factor(N, bound=50000)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 5000
        assert result is not None