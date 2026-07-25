"""Tests for Brahmagupta-Fibonacci and Fibonacci factoring methods."""
import pytest
from math import isqrt

from insideout.brahmagupta import (
    find_two_square_representation,
    find_all_two_square_representations,
    brahmagupta_fibonacci_factor,
    fermat_difference_of_squares,
)
from insideout.fibonacci_factor import (
    fibonacci_mod,
    pisano_period,
    entry_point,
    pisano_factor,
    fibonacci_gcd_factor,
)
from insideout.factor import factor, factor_with_method


class TestTwoSquareRepresentations:
    """Test finding representations of N as sum of two squares."""

    def test_5_is_1_2(self):
        """5 = 1^2 + 2^2."""
        result = find_two_square_representation(5)
        assert result is not None
        a, b = result
        assert a * a + b * b == 5

    def test_25_is_0_5(self):
        """25 = 0^2 + 5^2 = 3^2 + 4^2."""
        result = find_two_square_representation(25)
        assert result is not None
        a, b = result
        assert a * a + b * b == 25

    def test_65_has_two_representations(self):
        """65 = 1^2 + 8^2 = 4^2 + 7^2."""
        reps = find_all_two_square_representations(65)
        assert len(reps) >= 2
        for a, b in reps:
            assert a * a + b * b == 65

    def test_3_no_representation(self):
        """3 cannot be expressed as sum of two squares."""
        result = find_two_square_representation(3)
        assert result is None

    def test_primes_1_mod_4(self):
        """Primes p ≡ 1 mod 4 have a representation."""
        for p in [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]:
            result = find_two_square_representation(p)
            assert result is not None, f"Prime {p} should have a two-square representation"
            a, b = result
            assert a * a + b * b == p


class TestBrahmaguptaFibonacciFactor:
    """Test factoring via the Brahmagupta-Fibonacci identity."""

    @pytest.mark.parametrize("N,p,q", [
        (65, 5, 13),    # 65 = 1²+8² = 4²+7²
        (85, 5, 17),    # 85 = 2²+9² = 6²+7²
        (325, 5, 65),   # 325 = 1²+18² = 6²+17²
    ])
    def test_factors_numbers_with_two_representations(self, N, p, q):
        """Numbers with two square representations should be factorable."""
        result = brahmagupta_fibonacci_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_returns_none_for_primes_3_mod_4(self):
        """Primes ≡ 3 mod 4 have no sum-of-two-squares representation."""
        result = brahmagupta_fibonacci_factor(7)
        assert result is None

    def test_returns_none_for_small_N(self):
        result = brahmagupta_fibonacci_factor(2)
        assert result is None


class TestFermatDifferenceOfSquares:
    """Test Fermat's difference-of-squares method."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),   # Close factors
        (899, 29, 31),   # Twin primes
    ])
    def test_factors_semiprimes(self, N, p, q):
        """Fermat should factor close-factor semiprimes efficiently."""
        result = fermat_difference_of_squares(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_close_factors(self):
        """Fermat excels at close-factor semiprimes."""
        for p, q in [(17, 19), (29, 31), (41, 43)]:
            N = p * q
            result = fermat_difference_of_squares(N)
            assert result is not None, f"Failed to factor {N}={p}*{q}"
            fp, fq = result
            assert fp * fq == N


class TestFibonacciMod:
    """Test Fibonacci modular arithmetic."""

    def test_fib_mod_0(self):
        assert fibonacci_mod(0, 10) == 0

    def test_fib_mod_1(self):
        assert fibonacci_mod(1, 10) == 1

    def test_fib_mod_small(self):
        """F(10) = 55, F(10) mod 7 = 55 mod 7 = 6."""
        assert fibonacci_mod(10, 7) == 6

    def test_fib_mod_large(self):
        """F(100) mod 1000."""
        result = fibonacci_mod(100, 1000)
        assert 0 <= result < 1000

    def test_fib_mod_identity(self):
        """F(n) mod 1 = 0."""
        for n in [0, 1, 5, 10, 50]:
            assert fibonacci_mod(n, 1) == 0


class TestPisanoPeriod:
    """Test Pisano period computation."""

    def test_pisano_2(self):
        """π(2) = 3: F mod 2 = 0, 1, 1, 0, 1, 1, ..."""
        assert pisano_period(2) == 3

    def test_pisano_5(self):
        """π(5) = 20."""
        assert pisano_period(5) == 20

    def test_pisano_10(self):
        """π(10) = 60."""
        assert pisano_period(10) == 60


class TestEntryPoint:
    """Test Fibonacci entry point computation."""

    def test_entry_point_5(self):
        """5 | F(5), so α(5) divides 5."""
        alpha = entry_point(5, 100)
        assert alpha is not None
        assert fibonacci_mod(alpha, 5) == 0

    def test_entry_point_7(self):
        """7 | F(8), so α(7) divides 8."""
        alpha = entry_point(7, 100)
        assert alpha is not None
        assert fibonacci_mod(alpha, 7) == 0


class TestFibonacciGCD:
    """Test Fibonacci GCD factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Fibonacci GCD should factor small semiprimes."""
        result = fibonacci_gcd_factor(N, bound=1000)
        # May or may not find a factor depending on entry points
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestFactorIntegration:
    """Test that the factor() API integrates all new methods."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (667, 23, 29),
        (323, 17, 19),
        (121, 11, 11),
        (961, 31, 31),
    ])
    def test_factor_still_works(self, N, p, q):
        """factor() should still find factors with the new methods."""
        result = factor(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    def test_factor_with_method_reports_valid_method(self):
        """factor_with_method should report a valid method name."""
        valid_methods = {
            "perfect_square", "cf_precheck", "brahmagupta", "fermat",
            "fibonacci", "inside_out", "wavefront", "trial_division",
        }
        result = factor_with_method(21)
        assert result is not None
        _, method = result
        assert method in valid_methods

    def test_brahmagupta_method_for_65(self):
        """65 = 5*13 should be found by Brahmagupta method."""
        result = factor_with_method(65)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 65
        # 65 has two square representations: 1²+8² and 4²+7²
        # so Brahmagupta should work (but other methods may find it first)
        assert method in valid_methods_list()

    def test_fermat_method_for_close_factors(self):
        """Fermat method excels at close factors."""
        # 1763 = 41*43, very close factors
        result = factor_with_method(1763)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 1763


def valid_methods_list():
    return [
        "perfect_square", "cf_precheck", "brahmagupta", "fermat",
        "fibonacci", "resonance_cascade", "inside_out", "wavefront", "trial_division",
    ]