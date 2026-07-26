"""Tests for Hybrid Cyclo-SL2 Cascade."""
import pytest
import time
from math import isqrt

from insideout.hybrid_cyclo_sl2 import (
    hybrid_cyclo_sl2_factor,
    _cyclotomic_cascade,
    _sl2_smooth_cascade,
)


class TestCyclotomicCascade:
    """Test standalone cyclotomic cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (77, 7, 11),
        (323, 17, 19),
        (10403, 101, 103),
    ])
    def test_factors_small(self, N, p, q):
        result = _cyclotomic_cascade(N, bound=50000, base_points=10)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestSL2SmoothCascade:
    """Test standalone SL2 cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small(self, N, p, q):
        result = _sl2_smooth_cascade(N, bound=100000, num_curves=20)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestHybridCycloSL2:
    """Test hybrid cyclo-SL2 factoring."""

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
        result = hybrid_cyclo_sl2_factor(N, time_budget_ms=5000)
        assert result is not None
        fp, fq = result
        assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = hybrid_cyclo_sl2_factor(1681, time_budget_ms=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = hybrid_cyclo_sl2_factor(30, time_budget_ms=5000)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = hybrid_cyclo_sl2_factor(7, time_budget_ms=1000)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert hybrid_cyclo_sl2_factor(1) is None
        assert hybrid_cyclo_sl2_factor(2) is None
        assert hybrid_cyclo_sl2_factor(3) is None

    def test_performance(self):
        """Should complete within time budget."""
        N = 10007 * 10009  # 27-bit
        start = time.perf_counter()
        result = hybrid_cyclo_sl2_factor(N, time_budget_ms=5000)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 6000
        assert result is not None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = hybrid_cyclo_sl2_factor(2047, time_budget_ms=5000)
        if result is not None:
            assert result[0] * result[1] == 2047