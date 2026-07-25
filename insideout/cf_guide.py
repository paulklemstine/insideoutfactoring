"""Continued fraction steering for Inside-Out traversal.

The CF expansion of sqrt(N) produces convergents p_k/q_k that approximate
sqrt(N). Since tan(theta/2) = p/q at the target node, these convergents
predict the ideal branch (U, A, or D) at each level of the Pythagorean tree.

This module provides:
- Integer-only CF expansion of sqrt(N) (quadratic irrational algorithm)
- Convergent computation
- Branch prediction based on slope comparison
"""
from __future__ import annotations

from math import isqrt
from typing import NamedTuple

from .berggren import Triple, U, A, D, apply_matrix


def cf_sqrt(S: int, max_terms: int = 100) -> list[int]:
    """Compute the continued fraction expansion of sqrt(S).

    Uses the standard algorithm for quadratic irrationals (integer-only).
    Returns [a0, a1, a2, ...] where sqrt(S) = a0 + 1/(a1 + 1/(a2 + ...)).
    For perfect squares, returns [a0] with a0 = isqrt(S).

    The algorithm follows the recurrence:
        m_{n+1} = d_n * a_n - m_n
        d_{n+1} = (S - m_{n+1}^2) / d_n
        a_{n+1} = floor((a_0 + m_{n+1}) / d_{n+1})
    """
    if S < 0:
        raise ValueError(f"S must be non-negative, got {S}")

    a0 = isqrt(S)
    if a0 * a0 == S:
        return [a0]

    terms = [a0]
    m, d, a = 0, 1, a0

    for _ in range(max_terms):
        m = d * a - m
        d = (S - m * m) // d
        a = (a0 + m) // d
        terms.append(a)
        # Note: we do NOT break on d==1 (period completion).
        # The periodic part repeats, and max_terms controls how many
        # terms we generate. Breaking early would prevent computing
        # enough convergents for branch prediction.

    return terms


def convergents(cf: list[int]) -> list[tuple[int, int]]:
    """Compute convergents p_k/q_k from a continued fraction expansion.

    Uses the standard recurrence:
        p_{-1} = 1, p_0 = a_0
        q_{-1} = 0, q_0 = 1
        p_k = a_k * p_{k-1} + p_{k-2}
        q_k = a_k * q_{k-1} + q_{k-2}

    Returns list of (p_k, q_k) pairs where p_k/q_k -> sqrt(S).
    """
    if not cf:
        return []

    p_prev, p_curr = 1, cf[0]
    q_prev, q_curr = 0, 1
    result = [(p_curr, q_curr)]

    for a in cf[1:]:
        p_prev, p_curr = p_curr, a * p_curr + p_prev
        q_prev, q_curr = q_curr, a * q_curr + q_prev
        result.append((p_curr, q_curr))

    return result


def predict_branch(N: int, triple: tuple[int, int, int]) -> tuple[int, int, int]:
    """Predict which Berggren branch to follow from a given triple.

    Compares the slope of each child with the target slope sqrt(N).
    Returns (U_distance, A_distance, D_distance) — lower is better.

    The distance metric is |b^2 - N * a^2|, which measures how far
    the child triple's slope b/a is from sqrt(N). This uses integer-only
    arithmetic: we want b/a ~ sqrt(N), i.e., b^2 ~ N*a^2, so the
    distance is |b^2 - N*a^2|.
    """
    a, b, c = triple
    t = Triple(a, b, c)

    u_child = apply_matrix(U, t)
    a_child = apply_matrix(A, t)
    d_child = apply_matrix(D, t)

    def slope_distance(child: Triple) -> int:
        """Integer measure of how far child's slope is from sqrt(N).

        We want b/a ~ sqrt(N), i.e., b^2 ~ N*a^2.
        Distance = |b^2 - N * a^2| (smaller is better).
        """
        return abs(child.b * child.b - N * child.a * child.a)

    return (
        slope_distance(u_child),
        slope_distance(a_child),
        slope_distance(d_child),
    )


def cf_factor_check(N: int, max_terms: int = 100) -> tuple[int, int] | None:
    """Check if any CF convergent of sqrt(N) directly reveals a factor.

    For each convergent p_k/q_k of sqrt(N), check:
    1. p_k divides N
    2. q_k divides N
    3. (p_k - 1) divides N
    4. (p_k + 1) divides N
    5. (q_k - 1) divides N
    6. (q_k + 1) divides N

    This is extremely effective for close-factor semiprimes because
    the CF convergents of sqrt(N) rapidly approach p and q when p ≈ q.

    Returns (factor, N // factor) with factor < N // factor if found,
    None if no convergent reveals a factor.
    """
    if N < 4:
        return None

    cf = cf_sqrt(N, max_terms=max_terms)
    convs = convergents(cf)

    for pk, qk in convs:
        for candidate in (pk, qk, pk - 1, pk + 1, qk - 1, qk + 1):
            if 1 < candidate < N and N % candidate == 0:
                f = candidate
                return (min(f, N // f), max(f, N // f))

    return None


def cf_branch_sequence(N: int, max_depth: int = 50) -> list[tuple[str, int, int]]:
    """Compute the predicted branch sequence from CF convergents of sqrt(N).

    Returns list of (branch_label, convergent_p, convergent_q).
    branch_label is 'U', 'A', or 'D'.

    The branch is determined by comparing the convergent slope p/q
    against thresholds derived from the geometry of the Berggren tree:
    - slope < 1: 'D' (descending — triples with b < a)
    - slope > 2: 'U' (ascending — triples with much larger b)
    - otherwise: 'A' (balanced)
    """
    cf = cf_sqrt(N, max_terms=max_depth)
    convs = convergents(cf)

    result: list[tuple[str, int, int]] = []
    for p, q in convs:
        # Integer-only branch determination:
        # Compare p/q against thresholds using cross-multiplication
        # p/q < 1  =>  p < q   (D)
        # p/q > 2  =>  p > 2*q (U)
        # otherwise: A
        if p < q:
            result.append(('D', p, q))
        elif p > 2 * q:
            result.append(('U', p, q))
        else:
            result.append(('A', p, q))

    return result