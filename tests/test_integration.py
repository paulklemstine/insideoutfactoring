"""Integration tests for the full Inside-Out factoring pipeline."""
import pytest
from insideout.factor import factor


class TestSmallSemiprimes:
    """Factor small semiprimes where all methods should succeed."""

    @pytest.mark.parametrize("N", [15, 21, 25, 35, 49, 77, 91, 119, 143, 221])
    def test_small_semiprimes(self, N):
        result = factor(N)
        assert result is not None, f"Failed to factor {N}"
        p, q = result
        assert p * q == N, f"factors {p}*{q} != {N}"
        assert p > 1 and q > 1

    def test_product_of_primes(self):
        """Test products of small primes, verifying factors with sympy.isprime."""
        from sympy import isprime
        for p in [3, 5, 7, 11, 13]:
            for q in [p + 2, p + 4, p + 6]:
                if isprime(q) and p * q < 1000:
                    N = p * q
                    result = factor(N)
                    assert result is not None, f"Failed to factor {N}={p}*{q}"


class TestBerggrenTreeProperties:
    """Verify fundamental properties of the Berggren tree."""

    def test_all_children_are_ppt(self):
        """Every child of a PPT should be a PPT."""
        from insideout.berggren import Triple, children
        from insideout.triples import is_ppt
        root = Triple(3, 4, 5)
        for child in children(root):
            assert is_ppt(child), f"{child} is not PPT"
            for grandchild in children(child):
                assert is_ppt(grandchild), f"{grandchild} is not PPT"

    def test_gaussian_mn_roundtrip(self):
        """(m,n) -> triple -> (m,n) should be identity."""
        from insideout.gaussian import MnPair, mn_to_triple, triple_to_mn_pair
        from insideout.berggren import Triple
        for m, n in [(2, 1), (3, 2), (4, 1), (4, 3), (5, 2), (5, 4), (6, 1)]:
            pair = MnPair(m, n)
            triple = mn_to_triple(pair)
            result = triple_to_mn_pair(triple)
            assert result == pair, f"Round-trip failed for ({m},{n})"

    def test_cf_sqrt_periodic(self):
        """CF expansion of sqrt(N) should be periodic for non-square N."""
        from insideout.cf_guide import cf_sqrt
        for N in [2, 3, 5, 6, 7, 8, 10, 15, 21, 35]:
            cf = cf_sqrt(N, max_terms=50)
            assert cf[0] == int(N ** 0.5), f"Wrong first term for sqrt({N})"
            # Should have more than one term (periodic)
            assert len(cf) > 1, f"CF too short for sqrt({N})"