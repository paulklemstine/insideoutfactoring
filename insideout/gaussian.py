"""Gaussian integer (m,n) parametrization for Pythagorean triples.

Every PPT (a, b, c) with a = m**2 - n**2, b = 2mn, c = m**2 + n**2
corresponds to the Gaussian integer z = m + ni. The Berggren matrices
simplify to 2x2 integer transforms on (m, n), reducing computation from
3x3 to 2x2.

The (m,n) Berggren transforms are:
    U: (m,n) -> (2m-n, m)      matrix [[2,-1],[1, 0]]
    A: (m,n) -> (2m+n, m)      matrix [[2, 1],[1, 0]]
    D: (m,n) -> (m+2n, n)      matrix [[1, 2],[0, 1]]
"""
from __future__ import annotations

from typing import NamedTuple

from .berggren import Triple


class MnPair(NamedTuple):
    """A pair (m, n) representing the Gaussian integer m + ni."""
    m: int
    n: int


# 2x2 Berggren transforms in (m,n) space (row-major as tuple of row tuples)
U_MN = ((2, -1), (1, 0))
A_MN = ((2, 1), (1, 0))
D_MN = ((1, 2), (0, 1))

# Verified inverses (det(U_MN) = det(A_MN) = det(D_MN) = 1)
U_MN_INV = ((0, 1), (-1, 2))
A_MN_INV = ((0, 1), (1, -2))
D_MN_INV = ((1, -2), (0, 1))

ALL_MN_MATRICES = (U_MN, A_MN, D_MN)


def apply_mn_matrix(M: tuple[tuple[int, int], tuple[int, int]], pair: MnPair) -> MnPair:
    """Apply a 2x2 integer matrix to an (m,n) pair."""
    m, n = pair
    new_m = M[0][0] * m + M[0][1] * n
    new_n = M[1][0] * m + M[1][1] * n
    return MnPair(new_m, new_n)


def mn_children(pair: MnPair) -> list[MnPair]:
    """Generate the three children of (m,n) via Berggren transforms."""
    return [apply_mn_matrix(M, pair) for M in ALL_MN_MATRICES]


def mn_parent(pair: MnPair, child_matrix: tuple[tuple[int, int], tuple[int, int]]) -> MnPair | None:
    """Compute the parent by applying the inverse transform.

    Returns None if result has non-positive m or n, or m <= n.
    """
    inverse_map = {
        U_MN: U_MN_INV,
        A_MN: A_MN_INV,
        D_MN: D_MN_INV,
    }
    inv = inverse_map.get(child_matrix)
    if inv is None:
        raise ValueError(f"Unknown matrix: {child_matrix}")
    result = apply_mn_matrix(inv, pair)
    if result.m > result.n > 0:
        return result
    return None


def mn_to_triple(pair: MnPair) -> Triple:
    """Convert (m,n) to PPT (m**2-n**2, 2mn, m**2+n**2)."""
    m, n = pair
    a = m * m - n * n
    b = 2 * m * n
    c = m * m + n * n
    return Triple(a, b, c)


def triple_to_mn_pair(triple: Triple) -> MnPair | None:
    """Extract (m,n) from a PPT triple."""
    from .triples import triple_to_mn as _triple_to_mn
    result = _triple_to_mn(triple)
    if result is None:
        return None
    return MnPair(*result)