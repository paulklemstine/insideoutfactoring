"""Top-level factoring API.

Provides the main entry point for the Inside-Out factoring algorithm.
Tries multiple strategies in order: Inside-Out, wavefront search,
and trial division as a fallback.
"""
from __future__ import annotations

from math import isqrt

from .inside_out import inside_out_factor
from .wavefront import search_wavefront
from .cf_guide import cf_factor_check


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

    Method name is one of: "perfect_square", "cf_precheck", "steered",
    "inside_out", "wavefront", "trial_division".
    """
    if N < 4:
        return None

    # Handle even N
    if N % 2 == 0:
        if N == 2:
            return None
        return ((2, N // 2), "trial_division")

    # Perfect square detection
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return ((sqrt_N, sqrt_N), "perfect_square")

    # CF convergent divisibility pre-check
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        p, q = cf_result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "cf_precheck")

    # Strategy 1: Inside-Out (steered + BFS, combined)
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