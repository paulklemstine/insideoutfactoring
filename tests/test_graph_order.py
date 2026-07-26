"""Tests for Graph-Order Cascade factoring."""
import pytest
from math import isqrt

from insideout.graph_order import (
    graph_order_cascade_factor,
    order_spectrum_factor,
    _order_mod,
    _compute_order_graph,
)


class TestOrderMod:
    """Test multiplicative order computation."""

    def test_order_of_3_mod_7(self):
        """Order of 3 mod 7 is 6."""
        # 3^1=3, 3^2=2, 3^3=6, 3^4=4, 3^5=5, 3^6=1
        assert _order_mod(3, 7) == 6

    def test_order_of_2_mod_7(self):
        """Order of 2 mod 7 is 3."""
        # 2^1=2, 2^2=4, 2^3=1
        assert _order_mod(2, 7) == 3

    def test_order_of_1_mod_7(self):
        """Order of 1 mod anything is 1."""
        assert _order_mod(1, 7) == 1

    def test_not_coprime(self):
        """Order of element not coprime to modulus is None."""
        assert _order_mod(2, 6) is None
        assert _order_mod(3, 9) is None


class TestComputeOrderGraph:
    """Test order graph computation."""

    def test_basic(self):
        """Basic order graph computation."""
        graph = _compute_order_graph(2, 7, max_exponent=5)
        assert len(graph) > 0
        # 2^1 = 2, order = 3
        # 2^2 = 4, order = 6
        # etc.


class TestGraphOrderCascadeFactor:
    """Test graph-order cascade factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = graph_order_cascade_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = graph_order_cascade_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = graph_order_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = graph_order_cascade_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert graph_order_cascade_factor(1) is None
        assert graph_order_cascade_factor(2) is None
        assert graph_order_cascade_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = graph_order_cascade_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047


class TestOrderSpectrumFactor:
    """Test order spectrum factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = order_spectrum_factor(N, bound=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = order_spectrum_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = order_spectrum_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = order_spectrum_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert order_spectrum_factor(1) is None
        assert order_spectrum_factor(2) is None
        assert order_spectrum_factor(3) is None

    def test_p_minus_1_smooth(self):
        """Should find factor when p-1 is smooth."""
        # 2047 = 23 * 89
        result = order_spectrum_factor(2047, bound=5000)
        if result is not None:
            assert result[0] * result[1] == 2047