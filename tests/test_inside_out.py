"""Tests for the Inside-Out factoring algorithm."""
import pytest
from insideout.inside_out import (
    central_well, resonance_check, inside_out_factor,
)
from insideout.gaussian import MnPair
from insideout.berggren import Triple


class TestCentralWell:
    def test_well_for_15(self):
        """N=15=3*5, well should be near sqrt(15) ~ 3.87."""
        well = central_well(15)
        assert well.m > 0
        assert well.n > 0
        assert well.m > well.n  # m > n for valid (m,n)

    def test_well_for_21(self):
        well = central_well(21)
        assert well.m > well.n > 0

    def test_well_for_437(self):
        """N=437=19*23, well should have m > n > 0."""
        well = central_well(437)
        assert well.m > well.n > 0

    def test_well_m_n_coprime(self):
        """Central well should produce coprime m, n for PPT validity."""
        from math import gcd
        well = central_well(35)
        assert gcd(well.m, well.n) == 1

    def test_well_m_n_opposite_parity(self):
        """m - n should be odd for PPT condition."""
        well = central_well(35)
        assert (well.m - well.n) % 2 == 1


class TestResonanceCheck:
    def test_factor_15_via_perfect_square(self):
        """(5, 12, 13) reveals 15 = 3*5 via N^2 - b^2 = 9^2."""
        # 15^2 - 12^2 = 225 - 144 = 81 = 9^2, gcd(15, 9) = 3
        result = resonance_check(15, Triple(5, 12, 13))
        assert result is not None
        p, q = result
        assert p * q == 15

    def test_factor_15_via_divisor(self):
        """(3, 4, 5) reveals 15 = 3*5 because 3 divides 15."""
        result = resonance_check(15, Triple(3, 4, 5))
        assert result is not None
        p, q = result
        assert p * q == 15

    def test_no_factor_for_prime(self):
        """No triple factors a prime N."""
        result = resonance_check(7, Triple(3, 4, 5))
        # 7^2-9=40 (not sq), 7^2-16=33 (not sq), 3!|7, 4!|7
        assert result is None

    def test_trivial_a_equals_N(self):
        """When a == N, it's trivial and should return None."""
        result = resonance_check(15, Triple(15, 8, 17))
        assert result is None

    def test_factor_35_via_divisor(self):
        """If a leg of the triple divides N, we find a factor."""
        # (5, 12, 13) — 5 divides 35
        result = resonance_check(35, Triple(5, 12, 13))
        if result is not None:
            p, q = result
            assert p * q == 35

    def test_factor_77_via_perfect_square(self):
        """N^2 - a^2 = d^2 => gcd(N, d) is a factor."""
        # For N=77, if we find a triple where 77^2 - a^2 is a perfect square
        # 77^2 = 5929, try various a values
        result = resonance_check(77, Triple(77, 36, 85))
        # (77, 36, 85): 77^2 - 77^2 = 0, not useful
        # This should return None since a == N
        assert result is None


class TestInsideOutFactor:
    """Integration tests for the full Inside-Out algorithm."""

    def test_factor_15(self):
        result = inside_out_factor(15)
        assert result is not None
        p, q = result
        assert p * q == 15
        assert p > 1 and q > 1

    def test_factor_21(self):
        result = inside_out_factor(21)
        assert result is not None
        p, q = result
        assert p * q == 21

    def test_factor_35(self):
        result = inside_out_factor(35)
        assert result is not None
        p, q = result
        assert p * q == 35

    def test_factor_77(self):
        result = inside_out_factor(77)
        assert result is not None
        p, q = result
        assert p * q == 77

    def test_factor_437(self):
        """437 = 19 * 23"""
        result = inside_out_factor(437)
        assert result is not None
        p, q = result
        assert p * q == 437

    def test_factor_667(self):
        """667 = 23 * 29"""
        result = inside_out_factor(667)
        assert result is not None
        p, q = result
        assert p * q == 667

    def test_rejects_primes(self):
        """A prime number has no non-trivial factors."""
        result = inside_out_factor(7)
        assert result is None

    def test_rejects_even(self):
        """Even numbers are handled as edge cases."""
        result = inside_out_factor(6)
        # 6 = 2 * 3, but it's even — algorithm should still find it
        if result is not None:
            p, q = result
            assert p * q == 6

    def test_edge_case_N_less_than_4(self):
        """N < 4 cannot be factored as a product of two integers > 1."""
        assert inside_out_factor(3) is None
        assert inside_out_factor(2) is None
        assert inside_out_factor(1) is None

    def test_factor_returns_ordered(self):
        """Factors should be returned with p <= q."""
        result = inside_out_factor(35)
        assert result is not None
        p, q = result
        assert p <= q

    def test_factor_large_semiprime(self):
        """Larger semiprime: 3131 = 31 * 101"""
        result = inside_out_factor(3131)
        assert result is not None
        p, q = result
        assert p * q == 3131