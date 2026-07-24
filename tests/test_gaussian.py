"""Tests for Gaussian integer (m,n) parametrization."""
import pytest
from insideout.gaussian import (
    MnPair, U_MN, A_MN, D_MN,
    U_MN_INV, A_MN_INV, D_MN_INV,
    apply_mn_matrix, mn_children, mn_parent,
    mn_to_triple, triple_to_mn_pair,
)
from insideout.berggren import Triple


class TestMnMatrixDefinitions:
    """Verify 2x2 Berggren transforms in (m,n) space."""

    def test_U_mn_values(self):
        # U: (m,n) -> (2m-n, m)
        assert U_MN == ((2, -1), (1, 0))

    def test_A_mn_values(self):
        # A: (m,n) -> (2m+n, m)
        assert A_MN == ((2, 1), (1, 0))

    def test_D_mn_values(self):
        # D: (m,n) -> (m+2n, n)
        assert D_MN == ((1, 2), (0, 1))


class TestMnMatrixInverses:
    """Verify 2x2 inverse matrices."""

    def test_U_mn_inverse(self):
        # U_inv: (m,n) -> (n, -m+2n) = [[0,1],[-1,2]]
        assert U_MN_INV == ((0, 1), (-1, 2))

    def test_A_mn_inverse(self):
        # A_inv: [[0,1],[1,-2]]
        assert A_MN_INV == ((0, 1), (1, -2))

    def test_D_mn_inverse(self):
        # D_inv: [[1,-2],[0,1]]
        assert D_MN_INV == ((1, -2), (0, 1))


class TestMnTransforms:
    """Verify (m,n) transforms produce correct PPTs."""

    def test_U_on_root(self):
        # U: (2,1) -> (3,2) -> PPT (5,12,13)
        root = MnPair(2, 1)
        result = apply_mn_matrix(U_MN, root)
        assert result == MnPair(3, 2)
        triple = mn_to_triple(result)
        assert triple == Triple(5, 12, 13)

    def test_A_on_root(self):
        # A: (2,1) -> (5,2) -> PPT (21,20,29)
        root = MnPair(2, 1)
        result = apply_mn_matrix(A_MN, root)
        assert result == MnPair(5, 2)
        triple = mn_to_triple(result)
        assert triple == Triple(21, 20, 29)

    def test_D_on_root(self):
        # D: (2,1) -> (4,1) -> PPT (15,8,17)
        root = MnPair(2, 1)
        result = apply_mn_matrix(D_MN, root)
        assert result == MnPair(4, 1)
        triple = mn_to_triple(result)
        assert triple == Triple(15, 8, 17)

    def test_children_of_root(self):
        root = MnPair(2, 1)
        kids = mn_children(root)
        assert set(kids) == {MnPair(3, 2), MnPair(5, 2), MnPair(4, 1)}

    def test_parent_roundtrip(self):
        child = MnPair(3, 2)
        p = mn_parent(child, U_MN)
        assert p == MnPair(2, 1)


class TestTripleConversion:
    def test_triple_to_mn_roundtrip(self):
        for m, n in [(2, 1), (3, 2), (4, 1), (5, 2)]:
            pair = MnPair(m, n)
            triple = mn_to_triple(pair)
            result = triple_to_mn_pair(triple)
            assert result == pair, f"Failed for ({m},{n})"