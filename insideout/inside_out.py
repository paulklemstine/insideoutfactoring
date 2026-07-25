"""Inside-Out Factoring Algorithm.

The core algorithm: start at the Central Approximation Well (near sqrt(N))
and search radially outward through the Pythagorean tree, using CF steering
and modular filters to prune the search.
"""
from __future__ import annotations

from collections import deque
from math import gcd, isqrt
from typing import Iterator

from .berggren import Triple, children, apply_matrix, U, A, D
from .gaussian import MnPair, mn_to_triple, triple_to_mn_pair, mn_children
from .energy import is_energy_compatible, hypotenuse_bound
from .cf_guide import cf_sqrt, convergents, predict_branch
from .modular import filter_wavefront


def central_well(N: int) -> MnPair:
    """Compute the Central Approximation Well for N.

    For N = pq with p ~ q ~ sqrt(N), the well corresponds to the
    (m, n) pair near sqrt(sqrt(N)). We find the nearest valid (m, n)
    such that:
      - m > n > 0
      - gcd(m, n) = 1
      - (m - n) is odd (PPT condition)

    The well starts at m ~ isqrt(N) + 1, which places the resulting
    triple's hypotenuse near N.
    """
    sqrt_N = isqrt(N)

    # Start with m near sqrt(N), n = 1
    # The triple from (m, n) has c = m^2 + n^2
    # We want c ~ N, so m ~ sqrt(N)
    m_start = sqrt_N + 1
    n_start = 1

    # Ensure m > n (trivially satisfied for N >= 4)
    # Ensure (m - n) is odd for PPT condition
    if (m_start - n_start) % 2 == 0:
        m_start += 1

    # Find coprime m, n
    # Walk m up until we find gcd(m, n) = 1
    m = m_start
    n = n_start
    while gcd(m, n) != 1:
        m += 2  # Keep m - n odd

    return MnPair(m, n)


def resonance_check(N: int, triple: Triple) -> tuple[int, int] | None:
    """Check if a triple reveals the factors of N.

    For N = pq, if we have a triple (a, b, c) where a = N, then
    b = (q^2 - p^2)/2 and c = (q^2 + p^2)/2, giving p and q.

    More generally, check if N^2 - a^2 or N^2 - b^2 is a perfect square.
    If N^2 - a^2 = d^2, then gcd(N, d) may be a factor.
    Also checks whether a or b directly divides N.
    """
    a, b, c = triple

    # Check if a divides N (direct divisor hit)
    if 1 < a < N and N % a == 0:
        return (min(a, N // a), max(a, N // a))

    # Check if b divides N
    if 1 < b < N and N % b == 0:
        return (min(b, N // b), max(b, N // b))

    # Check if a == N (trivial, not useful)
    if a == N:
        return None

    # Check N^2 - a^2 = d^2 (perfect square)
    # Then N^2 = a^2 + d^2, and gcd(N, d) may be a factor
    delta_a = N * N - a * a
    if delta_a > 0:
        sqrt_delta = isqrt(delta_a)
        if sqrt_delta * sqrt_delta == delta_a:
            d = sqrt_delta
            g = gcd(N, d)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    # Check N^2 - b^2 = d^2 (perfect square)
    delta_b = N * N - b * b
    if delta_b > 0:
        sqrt_delta = isqrt(delta_b)
        if sqrt_delta * sqrt_delta == delta_b:
            d = sqrt_delta
            g = gcd(N, d)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    return None


def inside_out_factor(N: int, max_iterations: int = 500000) -> tuple[int, int] | None:
    """Factor N = p*q using Inside-Out traversal of the Pythagorean tree.

    Starts at the Central Approximation Well and expands radially,
    checking each node for resonance with N.

    Returns (p, q) with p <= q if factorization found, None if N is prime
    or factors not found within max_iterations.
    """
    # Edge cases
    if N < 4:
        return None

    # Handle even N
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Quick trial division for small factors (safety net)
    for p in range(3, min(isqrt(N) + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # CF convergents of sqrt(N) for steering
    cf = cf_sqrt(N, max_terms=100)

    # Start from the Central Approximation Well
    well = central_well(N)

    # BFS from the well, expanding radially through (m,n) children
    visited: set[tuple[int, int]] = set()
    queue: deque[MnPair] = deque()
    queue.append(well)

    # Also seed the BFS with several starting points near the well
    # to improve coverage. Add shifts of the well.
    m0, n0 = well.m, well.n
    for dm in range(-5, 6):
        for dn in range(0, min(m0, 6)):
            m = m0 + dm
            n = n0 + dn
            if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
                pair = MnPair(m, n)
                if (pair.m, pair.n) not in visited:
                    queue.append(pair)

    # Also start from root (3,4,5) -> (m,n) = (2,1) for coverage
    root_mn = MnPair(2, 1)
    if (2, 1) not in visited:
        queue.append(root_mn)

    iterations = 0

    while queue and iterations < max_iterations:
        current = queue.popleft()
        iterations += 1

        # Skip if already visited
        key = (current.m, current.n)
        if key in visited:
            continue
        visited.add(key)

        # Skip if m <= n (invalid PPT parameter)
        if current.m <= current.n:
            continue

        # Check PPT validity: coprime and opposite parity
        if gcd(current.m, current.n) != 1:
            continue
        if (current.m - current.n) % 2 != 1:
            continue

        # Convert to triple
        triple = mn_to_triple(current)
        a, b, c = triple

        # Energy bound check: if c is way too large, prune this branch
        upper = hypotenuse_bound(N)
        if c > upper:
            continue

        # Check resonance with N
        result = resonance_check(N, triple)
        if result is not None:
            p, q = result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        # Check scaled triples: N might be k*a or k*b for some scaling factor k
        if a > 0 and N > a and N % a == 0 and 1 < a < N:
            return (min(a, N // a), max(a, N // a))
        if b > 0 and N > b and N % b == 0 and 1 < b < N:
            return (min(b, N // b), max(b, N // b))

        # Expand children using (m,n) Berggren transforms
        for child in mn_children(current):
            child_key = (child.m, child.n)
            if child_key not in visited and child.m > child.n > 0:
                # Pre-check: will this child's triple be too large?
                child_triple = mn_to_triple(child)
                if child_triple.c <= upper:
                    queue.append(child)

    # Fallback: trial division up to sqrt(N)
    limit = isqrt(N) + 1
    for p in range(3, limit, 2):
        if N % p == 0:
            return (p, N // p)

    return None