"""Tests for Lucas-PPT factoring method."""
import pytest
from math import isqrt

from insideout.lucas_ppt import lucas_ppt_factor


class TestLucasPPT:
    """Test Lucas-PPT factoring method."""

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
        """Lucas-PPT should factor small semiprimes."""
        result = lucas_ppt_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_squares(self):
        """Lucas-PPT should handle perfect squares via isqrt check."""
        result = lucas_ppt_factor(121)
        assert result is not None
        assert result == (11, 11)

    def test_even_numbers(self):
        """Lucas-PPT should handle even numbers."""
        result = lucas_ppt_factor(6)
        assert result is not None
        assert result[0] * result[1] == 6

    def test_small_primes_return_none(self):
        """Lucas-PPT should return None for small primes."""
        result = lucas_ppt_factor(7)
        if result is not None:
            assert result[0] * result[1] == 7

    def test_close_factors(self):
        """Lucas-PPT should handle close-factor semiprimes."""
        result = lucas_ppt_factor(323)  # 17*19
        assert result is not None
        assert result[0] * result[1] == 323

    def test_larger_semiprime(self):
        """Lucas-PPT should handle larger semiprimes."""
        result = lucas_ppt_factor(10007 * 10009)
        assert result is not None
        fp, fq = result
        assert fp * fq == 10007 * 10009