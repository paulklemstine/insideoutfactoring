"""Tests for Berggren matrices and tree traversal primitives."""
import pytest
from insideout.berggren import (
    U, A, D, U_INV, A_INV, D_INV,
    apply_matrix, children, parent,
    Matrix3x3, Triple,
)


class TestMatrixDefinitions:
    """Verify Berggren matrices are correctly defined."""

    def test_U_values(self):
        expected = Matrix3x3(1, -2, 2, 2, -1, 2, 2, -2, 3)
        assert U == expected

    def test_A_values(self):
        expected = Matrix3x3(1, 2, 2, 2, 1, 2, 2, 2, 3)
        assert A == expected

    def test_D_values(self):
        expected = Matrix3x3(-1, 2, 2, -2, 1, 2, -2, 2, 3)
        assert D == expected


class TestMatrixInverses:
    """Verify U*U_inv = A*A_inv = D*D_inv = I."""

    def test_U_inverse(self):
        identity = apply_matrix(U, U_INV, as_mat=True)
        expected = Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1)
        assert identity == expected

    def test_A_inverse(self):
        identity = apply_matrix(A, A_INV, as_mat=True)
        expected = Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1)
        assert identity == expected

    def test_D_inverse(self):
        identity = apply_matrix(D, D_INV, as_mat=True)
        expected = Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1)
        assert identity == expected

    def test_inverse_values(self):
        """Verify exact integer values of inverses (computed via adjugate)."""
        assert U_INV == Matrix3x3(1, 2, -2, -2, -1, 2, -2, -2, 3)
        assert A_INV == Matrix3x3(1, 2, -2, 2, 1, -2, -2, -2, 3)
        assert D_INV == Matrix3x3(-1, -2, 2, 2, 1, -2, -2, -2, 3)


class TestTreeTraversal:
    """Verify Berggren matrices generate valid PPTs from root."""

    def test_children_of_root(self):
        root = Triple(3, 4, 5)
        kids = children(root)
        assert len(kids) == 3
        # U*(3,4,5) = (5,12,13)
        # A*(3,4,5) = (21,20,29)
        # D*(3,4,5) = (15,8,17)
        expected = {Triple(5, 12, 13), Triple(21, 20, 29), Triple(15, 8, 17)}
        assert set(kids) == expected

    def test_U_child_of_root(self):
        result = apply_matrix(U, Triple(3, 4, 5))
        assert result == Triple(5, 12, 13)

    def test_A_child_of_root(self):
        result = apply_matrix(A, Triple(3, 4, 5))
        assert result == Triple(21, 20, 29)

    def test_D_child_of_root(self):
        result = apply_matrix(D, Triple(3, 4, 5))
        assert result == Triple(15, 8, 17)

    def test_parent_roundtrip(self):
        """Applying inverse to a child returns the parent."""
        child = Triple(5, 12, 13)
        p = parent(child, U)
        assert p == Triple(3, 4, 5)

    def test_parent_returns_none_for_root(self):
        """Inverse of root gives negative values (not a valid PPT)."""
        result = parent(Triple(3, 4, 5), U)
        # (3,4,5) has no parent — inverse gives negative entries
        assert result is None or any(x <= 0 for x in result)