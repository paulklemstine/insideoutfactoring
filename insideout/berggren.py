"""Berggren's matrix transformations for Pythagorean triples.

Provides the three unimodular matrices U, A, D that generate all
primitive Pythagorean triples (PPTs) from the root (3, 4, 5),
along with their inverses for Inside-Out traversal.

References:
    Berggren, B. (1934). "Pytagoreiska trianglar".
"""
from __future__ import annotations

from typing import NamedTuple


class Matrix3x3(NamedTuple):
    """A 3x3 integer matrix stored row-major."""
    r0c0: int; r0c1: int; r0c2: int
    r1c0: int; r1c1: int; r1c2: int
    r2c0: int; r2c1: int; r2c2: int

    def row(self, i: int) -> tuple[int, int, int]:
        """Return row i as a tuple."""
        if i == 0: return (self.r0c0, self.r0c1, self.r0c2)
        if i == 1: return (self.r1c0, self.r1c1, self.r1c2)
        return (self.r2c0, self.r2c1, self.r2c2)


class Triple(NamedTuple):
    """A Pythagorean triple (a, b, c) where a**2 + b**2 = c**2."""
    a: int
    b: int
    c: int


# Berggren's three matrices (det = +/-1, unimodular)
U = Matrix3x3(1, -2, 2, 2, -1, 2, 2, -2, 3)
A = Matrix3x3(1,  2, 2, 2,  1, 2, 2,  2, 3)
D = Matrix3x3(-1, 2, 2, -2, 1, 2, -2, 2, 3)

# Verified inverses (computed via adjugate / cofactor method)
U_INV = Matrix3x3(1, 2, -2, -2, -1, 2, -2, -2, 3)
A_INV = Matrix3x3(1, 2, -2, 2, 1, -2, -2, -2, 3)
D_INV = Matrix3x3(-1, -2, 2, 2, 1, -2, -2, -2, 3)

ALL_MATRICES = (U, A, D)
ALL_INVERSES = (U_INV, A_INV, D_INV)


def apply_matrix(M: Matrix3x3, v: Triple | Matrix3x3, as_mat: bool = False) -> Triple | Matrix3x3:
    """Apply matrix M to triple v, or multiply two matrices if as_mat=True.

    For a 3x3 matrix M and a triple v=(a,b,c), computes M*v as a column vector.
    For two matrices M1 and M2, computes the matrix product M1*M2.
    """
    if as_mat:
        N = v  # v is actually a Matrix3x3 here
        rows = []
        for i in range(3):
            m_row = M.row(i)
            row_result = []
            for j in range(3):
                n_col = (N.r0c0 if j == 0 else N.r0c1 if j == 1 else N.r0c2,
                          N.r1c0 if j == 0 else N.r1c1 if j == 1 else N.r1c2,
                          N.r2c0 if j == 0 else N.r2c1 if j == 1 else N.r2c2)
                val = (m_row[0] * n_col[0] +
                        m_row[1] * n_col[1] +
                        m_row[2] * n_col[2])
                row_result.append(val)
            rows.extend(row_result)
        return Matrix3x3(*rows)

    # Matrix-vector product: M * (a, b, c)^T
    a, b, c = v
    r0 = M.row(0)
    r1 = M.row(1)
    r2 = M.row(2)
    new_a = r0[0] * a + r0[1] * b + r0[2] * c
    new_b = r1[0] * a + r1[1] * b + r1[2] * c
    new_c = r2[0] * a + r2[1] * b + r2[2] * c
    return Triple(new_a, new_b, new_c)


def children(triple: Triple) -> list[Triple]:
    """Generate the three children of a PPT via Berggren matrices."""
    return [apply_matrix(M, triple) for M in ALL_MATRICES]


def parent(triple: Triple, child_matrix: Matrix3x3) -> Triple | None:
    """Compute the parent of a PPT by applying the inverse of child_matrix.

    Returns None if the result has non-positive entries (not a valid PPT).
    """
    if child_matrix == U:
        inv = U_INV
    elif child_matrix == A:
        inv = A_INV
    elif child_matrix == D:
        inv = D_INV
    else:
        raise ValueError(f"Unknown matrix: {child_matrix}")

    result = apply_matrix(inv, triple)
    if result.a > 0 and result.b > 0 and result.c > 0:
        return result
    return None