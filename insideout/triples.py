"""Primitive Pythagorean triple generation, validation, and (m,n) parametrization.

A primitive Pythagorean triple (PPT) is a triple (a, b, c) of positive
integers with a2 + b2 = c2, gcd(a, b) = 1, and a, b of opposite parity.
Every PPT can be written as (m2−n2, 2mn, m2+n2) for coprime m > n > 0
with m − n odd.
"""
from __future__ import annotations

from collections import deque
from math import gcd, isqrt
from typing import Iterator

from .berggren import Triple, children


def is_ppt(triple: Triple) -> bool:
    """Check whether a triple is a primitive Pythagorean triple."""
    a, b, c = triple
    if a <= 0 or b <= 0 or c <= 0:
        return False
    if a * a + b * b != c * c:
        return False
    if gcd(a, b) != 1:
        return False
    # Opposite parity: one of a, b is even, the other is odd
    if (a + b) % 2 == 0:
        return False
    return True


def is_valid_triple(triple: Triple) -> bool:
    """Check whether a triple satisfies a2 + b2 = c2 (may be non-primitive)."""
    a, b, c = triple
    return a > 0 and b > 0 and c > 0 and a * a + b * b == c * c


def normalize_triple(triple: Triple) -> Triple:
    """Sort legs so a < b (standard form)."""
    a, b, c = triple
    if a > b:
        return Triple(b, a, c)
    return triple


def mn_to_triple(m: int, n: int) -> Triple:
    """Convert (m, n) parametrization to a Pythagorean triple.

    Returns (m2−n2, 2mn, m2+n2). Does not enforce coprimality;
    call is_ppt() to check.
    """
    return Triple(m * m - n * n, 2 * m * n, m * m + n * n)


def triple_to_mn(triple: Triple) -> tuple[int, int] | None:
    """Extract (m, n) from a PPT.

    Returns None if the triple is not a valid PPT.
    """
    a, b, c = normalize_triple(triple)
    if not is_ppt(Triple(a, b, c)):
        return None
    # For a PPT in standard form (a < b): a = m2−n2, b = 2mn, c = m2+n2
    # So m2 = (c + a) / 2, n2 = (c − a) / 2
    m_sq = (c + a) // 2
    n_sq = (c - a) // 2
    m = isqrt(m_sq)
    n = isqrt(n_sq)
    if m * m == m_sq and n * n == n_sq and m > n > 0:
        return (m, n)
    # Try the other assignment: a = 2mn, b = m2−n2 (if b was the odd leg)
    m_sq = (c + b) // 2
    n_sq = (c - b) // 2
    m = isqrt(m_sq)
    n = isqrt(n_sq)
    if m * m == m_sq and n * n == n_sq and m > n > 0:
        return (m, n)
    return None


def scale_triple(triple: Triple, k: int) -> Triple:
    """Scale a PPT by integer factor k (produces a non-primitive triple)."""
    return Triple(triple.a * k, triple.b * k, triple.c * k)


def generate_ppts(depth: int = 0) -> Iterator[Triple]:
    """Generate PPTs using Berggren's tree up to the given depth.

    Yields triples in BFS order. depth=0 yields only the root (3,4,5).
    """
    root = Triple(3, 4, 5)
    yield root
    if depth == 0:
        return
    queue: deque[tuple[Triple, int]] = deque([(root, 0)])
    while queue:
        node, d = queue.popleft()
        if d >= depth:
            continue
        for child in children(node):
            yield child
            queue.append((child, d + 1))