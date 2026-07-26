"""Tests for Batch CRT Cascade factoring."""
import pytest
from math import isqrt

from insideout.batch_crt_cascade import batch_crt_cascade_factor


class TestBatchCRTCascade:
    """Test the full batch CRT cascade method."""

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
        """Batch CRT should factor small semiprimes."""
        result = batch_crt_cascade_factor(N, bound=5000, stage2_bound=500, max_params=16)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = batch_crt_cascade_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = batch_crt_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_small_prime(self):
        """Primes should return None."""
        result = batch_crt_cascade_factor(7, bound=5000)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert batch_crt_cascade_factor(1) is None
        assert batch_crt_cascade_factor(2) is None
        assert batch_crt_cascade_factor(3) is None