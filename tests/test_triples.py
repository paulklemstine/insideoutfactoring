"""Tests for PPT generation, validation, and (m,n) parametrization."""
import pytest
from itertools import islice
from insideout.triples import (
    is_ppt, is_valid_triple, normalize_triple,
    generate_ppts, triple_to_mn, mn_to_triple, scale_triple,
)
from insideout.berggren import Triple


class TestPPTValidation:
    def test_root_is_ppt(self):
        assert is_ppt(Triple(3, 4, 5))

    def test_non_primitive(self):
        # (6, 8, 10) is a Pythagorean triple but not primitive (gcd=2)
        assert not is_ppt(Triple(6, 8, 10))

    def test_not_pythagorean(self):
        assert not is_ppt(Triple(1, 2, 3))

    def test_children_are_ppts(self):
        root = Triple(3, 4, 5)
        from insideout.berggren import children
        for child in children(root):
            assert is_ppt(child), f"{child} is not a PPT"

    def test_opposite_parity(self):
        """In a PPT, one leg is odd and the other is even."""
        root = Triple(3, 4, 5)
        from insideout.berggren import children
        for child in children(root):
            assert (child.a + child.b) % 2 == 1  # odd + even


class TestMnParametrization:
    def test_roundtrip_root(self):
        # (3, 4, 5) comes from m=2, n=1
        result = triple_to_mn(Triple(3, 4, 5))
        assert result == (2, 1)

    def test_roundtrip_5_12_13(self):
        result = triple_to_mn(Triple(5, 12, 13))
        assert result == (3, 2)

    def test_mn_to_triple(self):
        assert mn_to_triple(2, 1) == Triple(3, 4, 5)
        assert mn_to_triple(3, 2) == Triple(5, 12, 13)

    def test_roundtrip_various(self):
        for m, n in [(2, 1), (3, 2), (4, 1), (4, 3), (5, 2)]:
            triple = mn_to_triple(m, n)
            result = triple_to_mn(triple)
            assert result == (m, n), f"Failed for ({m},{n}): {triple} -> {result}"

    def test_coprime_requirement(self):
        """gcd(m,n) must be 1 for a primitive triple."""
        # m=4, n=2: gcd=2, gives non-primitive triple
        t = mn_to_triple(4, 2)
        assert not is_ppt(t)


class TestPPTGeneration:
    def test_generate_depth_0(self):
        ppts = list(generate_ppts(depth=0))
        assert Triple(3, 4, 5) in ppts

    def test_generate_depth_1(self):
        ppts = list(generate_ppts(depth=1))
        assert Triple(3, 4, 5) in ppts
        assert Triple(5, 12, 13) in ppts
        assert Triple(21, 20, 29) in ppts
        assert Triple(15, 8, 17) in ppts

    def test_generate_all_ppts(self):
        """Every generated triple must be a valid PPT."""
        for t in islice(generate_ppts(depth=3), 50):
            assert is_ppt(t), f"{t} is not a PPT"


class TestScaleTriple:
    def test_double_root(self):
        result = scale_triple(Triple(3, 4, 5), 2)
        assert result == Triple(6, 8, 10)

    def test_triple_root(self):
        result = scale_triple(Triple(3, 4, 5), 3)
        assert result == Triple(9, 12, 15)