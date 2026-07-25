"""Top-level factoring API.

Provides the main entry point for the Inside-Out factoring algorithm.
Tries multiple strategies in order: Inside-Out, wavefront search,
and trial division as a fallback.
"""
from __future__ import annotations

from math import isqrt

from .inside_out import inside_out_factor
from .wavefront import search_wavefront


def factor(N: int) -> tuple[int, int] | None:
    """Factor an integer N into two factors p and q where N = p*q.

    Uses the Inside-Out factoring algorithm with wavefront search
    and trial division as fallback.

    Returns (p, q) with p < q if N is composite, None if N is prime.
    """
    result = factor_with_method(N)
    if result is None:
        return None
    return result[0]


def factor_with_method(N: int) -> tuple[tuple[int, int], str] | None:
    """Factor N and return the factors along with the method used.

    Returns ((p, q), method_name) if N is composite, None if N is prime.
    Method name is one of: "inside_out", "wavefront", "trial_division".
    """
    if N < 4:
        return None

    # Handle even N
    if N % 2 == 0:
        if N == 2:
            return None
        return ((2, N // 2), "trial_division")

    # Strategy 1: Inside-Out (BFS from energy well)
    result = inside_out_factor(N, max_iterations=50000)
    if result is not None:
        p, q = result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "inside_out")

    # Strategy 2: Wavefront search
    result = search_wavefront(N, max_radius=500)
    if result is not None:
        p, q = result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "wavefront")

    # Strategy 3: Trial division fallback
    limit = isqrt(N) + 1
    for p in range(3, limit, 2):
        if N % p == 0:
            return ((p, N // p), "trial_division")

    return None