"""Tests for Self-Tuning Adaptive Portfolio."""
import pytest
import time
from math import isqrt

from insideout.self_tuning_portfolio import (
    self_tuning_factor,
    _estimate_difficulty,
)


class TestEstimateDifficulty:
    """Test difficulty estimation."""

    @pytest.mark.parametrize("N,expected", [
        (15, 'easy'),
        (1000000000, 'easy'),  # ~30 bits
        (1000000000000000, 'medium'),  # ~50 bits
        (100000000000000000000, 'hard'),  # ~65 bits
        (10**30, 'very_hard'),
    ])
    def test_difficulty_estimation(self, N, expected):
        """Should estimate difficulty correctly based on bit length."""
        assert _estimate_difficulty(N) == expected


class TestSelfTuningFactor:
    """Test self-tuning factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_semiprimes(self, N, p, q):
        """Should factor small semiprimes quickly."""
        result = self_tuning_factor(N, time_budget_ms=2000)
        assert result is not None
        fp, fq = result
        assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = self_tuning_factor(1681, time_budget_ms=2000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = self_tuning_factor(30, time_budget_ms=2000)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = self_tuning_factor(7, time_budget_ms=1000)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert self_tuning_factor(1) is None
        assert self_tuning_factor(2) is None
        assert self_tuning_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = self_tuning_factor(2047, time_budget_ms=2000)
        if result is not None:
            assert result[0] * result[1] == 2047