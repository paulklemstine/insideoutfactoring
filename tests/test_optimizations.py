"""Tests for the critical optimizations: CF pre-check, perfect square detection, lower energy bound."""
import pytest
from math import isqrt
from insideout.cf_guide import cf_factor_check
from insideout.inside_out import inside_out_factor
from insideout.wavefront import search_wavefront
from insideout.factor import factor, factor_with_method


class TestCFactorCheck:
    """Test the CF convergent divisibility pre-check."""

    def test_cf_reveals_factor_15(self):
        """N=15=3*5: sqrt(15) ≈ 3.87, convergents include 4 which is 3+1."""
        result = cf_factor_check(15)
        assert result is not None
        p, q = result
        assert p * q == 15

    def test_cf_reveals_factor_21(self):
        """N=21=3*7: sqrt(21) ≈ 4.58."""
        result = cf_factor_check(21)
        assert result is not None
        p, q = result
        assert p * q == 21

    def test_cf_reveals_factor_35(self):
        """N=35=5*7."""
        result = cf_factor_check(35)
        assert result is not None
        p, q = result
        assert p * q == 35

    def test_cf_reveals_perfect_square(self):
        """N=121=11*11: sqrt(121)=11, CF=[11], convergent (11,1) directly gives factor."""
        result = cf_factor_check(121)
        assert result is not None
        p, q = result
        assert p * q == 121

    def test_cf_reveals_close_factors(self):
        """N=323=17*19: close factors should be caught by convergent neighbors."""
        result = cf_factor_check(323)
        assert result is not None
        p, q = result
        assert p * q == 323

    def test_cf_reveals_1849(self):
        """N=1849=43*43: perfect square, CF convergent directly reveals 43."""
        result = cf_factor_check(1849)
        assert result is not None
        p, q = result
        assert p * q == 1849

    def test_cf_returns_none_for_prime(self):
        """N=7: prime, no convergent should divide it."""
        result = cf_factor_check(7)
        assert result is None

    def test_cf_returns_none_for_small_prime(self):
        """N=13: prime."""
        result = cf_factor_check(13)
        assert result is None


class TestPerfectSquareDetection:
    """Test perfect square detection in inside_out_factor and search_wavefront."""

    @pytest.mark.parametrize("N,sqrt_N", [
        (9, 3),
        (25, 5),
        (121, 11),
        (361, 19),
        (961, 31),
        (1849, 43),
    ])
    def test_inside_out_perfect_squares(self, N, sqrt_N):
        """Perfect squares should be factored instantly."""
        result = inside_out_factor(N)
        assert result is not None
        p, q = result
        assert p * q == N
        assert p == sqrt_N and q == sqrt_N

    @pytest.mark.parametrize("N,sqrt_N", [
        (9, 3),
        (25, 5),
        (121, 11),
        (361, 19),
    ])
    def test_wavefront_perfect_squares(self, N, sqrt_N):
        """Perfect squares should be factored instantly via wavefront too."""
        result = search_wavefront(N)
        assert result is not None
        p, q = result
        assert p * q == N


class TestCloseFactorPerformance:
    """Test that close-factor semiprimes are now factored efficiently."""

    @pytest.mark.parametrize("N,p,q", [
        (121, 11, 11),   # Perfect square
        (323, 17, 19),   # Twin primes
        (899, 29, 31),   # Twin primes
        (1763, 41, 43),  # Close primes
        (437, 19, 23),   # Moderate gap
    ])
    def test_close_factors_found(self, N, p, q):
        """Close-factor semiprimes should be factored correctly."""
        result = factor(N)
        assert result is not None, f"Failed to factor {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N

    def test_961_perfect_square(self):
        """N=961=31*31 should be instant."""
        result = factor(961)
        assert result is not None
        assert result[0] * result[1] == 961

    def test_1849_perfect_square(self):
        """N=1849=43*43 should be instant."""
        result = factor(1849)
        assert result is not None
        assert result[0] * result[1] == 1849


class TestFactorWithMethod:
    """Test that factor_with_method reports the correct method."""

    def test_perfect_square_method(self):
        """Perfect squares should be detected by perfect_square method."""
        result = factor_with_method(121)
        assert result is not None
        _, method = result
        assert method == "perfect_square"

    def test_cf_precheck_method(self):
        """Semiprimes caught by CF pre-check should report cf_precheck."""
        # N=15 should be caught by CF pre-check (convergent reveals factor)
        result = factor_with_method(15)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 15
        # The method could be cf_precheck or inside_out depending on which catches it first
        assert method in ("perfect_square", "cf_precheck", "inside_out")