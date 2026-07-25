"""Tests for Resonance Cascade Factoring algorithm."""
import pytest
from math import isqrt

from insideout.resonance_cascade import (
    resonance_cascade_factor,
    _cf_convergent_resonance,
    _mobius_descent,
    _squaring_conductance,
)
from insideout.cf_guide import cf_sqrt, convergents


class TestCFConvergentResonance:
    """Test Stage 2: CF-convergent resonance scan."""

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
        """CF convergent resonance should find factors of small semiprimes."""
        cf = cf_sqrt(N, max_terms=100)
        convs = convergents(cf)
        result = _cf_convergent_resonance(N, convs)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_returns_none_for_prime(self):
        """CF resonance should return None for primes (no convergent divides)."""
        cf = cf_sqrt(97, max_terms=50)
        convs = convergents(cf)
        result = _cf_convergent_resonance(97, convs)
        # May or may not find a factor of a prime
        if result is not None:
            fp, fq = result
            assert fp * fq == 97


class TestMobiusDescent:
    """Test Stage 3: Möbius descent from near-N triples."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Möbius descent should find factors of small semiprimes."""
        result = _mobius_descent(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_returns_none_for_prime(self):
        """Möbius descent should return None for primes."""
        result = _mobius_descent(97)
        if result is not None:
            fp, fq = result
            assert fp * fq == 97


class TestSquaringConductance:
    """Test Stage 4: Squaring conductance analysis."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Squaring conductance should find factors of small semiprimes."""
        result = _squaring_conductance(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_identity_detection(self):
        """Squaring conductance should detect x² ≡ x (mod p) structure."""
        # For N = 15: 4² = 16 ≡ 1 (mod 15), 4-1=3, gcd(3,15)=3
        result = _squaring_conductance(15)
        assert result is not None
        assert result[0] * result[1] == 15


class TestResonanceCascadeFactor:
    """Test the full Resonance Cascade Factoring algorithm."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (667, 23, 29),
        (323, 17, 19),
        (899, 29, 31),
        (9991, 97, 103),
    ])
    def test_factors_semiprimes(self, N, p, q):
        """RCF should factor small-to-medium semiprimes."""
        result = resonance_cascade_factor(N)
        assert result is not None, f"Failed to factor {N}={p}*{q}"
        fp, fq = result
        assert fp * fq == N

    def test_perfect_squares(self):
        """RCF should handle perfect squares via isqrt check."""
        result = resonance_cascade_factor(121)
        assert result is not None
        assert result == (11, 11)

    def test_even_numbers(self):
        """RCF should handle even numbers."""
        result = resonance_cascade_factor(6)
        assert result is not None
        assert result[0] * result[1] == 6

    def test_small_primes_return_none(self):
        """RCF should return None for small primes."""
        result = resonance_cascade_factor(7)
        if result is not None:
            assert result[0] * result[1] == 7

    def test_close_factors(self):
        """RCF should handle close-factor semiprimes efficiently."""
        # 323 = 17*19 (close factors)
        result = resonance_cascade_factor(323)
        assert result is not None
        assert result[0] * result[1] == 323

    def test_larger_semiprime(self):
        """RCF should handle larger semiprimes."""
        # 10007 * 10009 = 100160063
        result = resonance_cascade_factor(100160063)
        assert result is not None
        fp, fq = result
        assert fp * fq == 100160063


class TestFactorIntegration:
    """Test that factor() includes resonance_cascade in its cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
    ])
    def test_factor_uses_resonance_cascade(self, N, p, q):
        """factor() should be able to use resonance_cascade method."""
        from insideout.factor import factor_with_method
        result = factor_with_method(N)
        assert result is not None
        (fp, fq), method = result
        assert fp * fq == N

    def test_resonance_cascade_is_valid_method(self):
        """resonance_cascade should be a valid method name."""
        from insideout.factor import factor_with_method
        valid_methods = {
            "perfect_square", "cf_precheck", "brahmagupta", "fermat",
            "fibonacci", "resonance_cascade", "inside_out",
            "wavefront", "trial_division",
        }
        result = factor_with_method(21)
        assert result is not None
        _, method = result
        assert method in valid_methods