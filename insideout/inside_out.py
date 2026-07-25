"""Inside-Out Factoring Algorithm.

The core algorithm: start at the Central Approximation Well (near sqrt(N))
and search radially outward through the Pythagorean tree, using CF steering
and modular filters to prune the search.

Two search strategies are available:
- BFS (breadth-first): Explores the tree level by level. Guaranteed to find
  the closest node but slow for close-factor semiprimes.
- Steered (best-first): Uses predict_branch distance as priority to guide
  the search toward the target. Much faster for close-factor cases.
"""
from __future__ import annotations

import heapq
from collections import deque
from math import gcd, isqrt
from typing import Iterator

from .berggren import Triple, children, apply_matrix, U, A, D
from .gaussian import MnPair, mn_to_triple, triple_to_mn_pair, mn_children
from .energy import is_energy_compatible, hypotenuse_bound
from .cf_guide import cf_sqrt, convergents, predict_branch, cf_factor_check
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


def cf_seeded_well_points(N: int) -> list[MnPair]:
    """Generate BFS seed points from CF convergents of sqrt(N).

    For close-factor semiprimes, CF convergents of sqrt(N) approximate
    the factors p and q. By generating (m,n) seed points from these
    convergents, we ensure the BFS starts near the target node even
    when p ~ q.

    Also includes the traditional well and its neighborhood for
    well-separated factor coverage.
    """
    seeds: list[MnPair] = []
    seen: set[tuple[int, int]] = set()

    def _add_seed(m: int, n: int) -> None:
        """Add a valid (m,n) seed point, skipping duplicates."""
        if (m, n) in seen:
            return
        if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
            seen.add((m, n))
            seeds.append(MnPair(m, n))

    # CF-convergent-derived seed points
    if N >= 4:
        cf = cf_sqrt(N, max_terms=50)
        convs = convergents(cf)

        for pk, qk in convs[:20]:
            # Try (pk, 1) as a seed — convergent numerator is near sqrt(N)
            _add_seed(pk, 1)
            # Try (pk, qk) if it satisfies PPT conditions
            _add_seed(pk, qk)
            # Try (qk, 1) — convergent denominator
            if qk > 1:
                _add_seed(qk, 1)
            # Try nearby values: pk ± 1 with small n values
            for delta in (-1, 1):
                m = pk + delta
                if m > 1:
                    _add_seed(m, 1)
                    for n_val in range(2, min(m, 8)):
                        _add_seed(m, n_val)

    # Traditional well and its neighborhood
    well = central_well(N)
    _add_seed(well.m, well.n)
    m0, n0 = well.m, well.n
    for dm in range(-5, 6):
        for dn in range(0, min(m0, 6)):
            _add_seed(m0 + dm, n0 + dn)

    # Root (3,4,5) -> (m,n) = (2,1) for full tree coverage
    _add_seed(2, 1)

    return seeds


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


def _steered_search(N: int, max_iterations: int = 50000) -> tuple[int, int] | None:
    """Factor N using CF-steered best-first search.

    Uses a priority queue (min-heap) ordered by predict_branch distance
    to guide the search toward nodes most likely to contain factors of N.
    This converts the BFS from O(N) iterations to O(log²N) directed search
    for semiprimes where CF convergents don't directly reveal factors.

    Returns (p, q) with p <= q if factorization found, None otherwise.
    """
    well = central_well(N)
    upper = hypotenuse_bound(N)

    # Priority queue: (distance, counter, MnPair)
    # counter is a tiebreaker to avoid comparing MnPair objects
    visited: set[tuple[int, int]] = set()
    heap: list[tuple[int, int, MnPair]] = []
    counter = 0

    # Seed with CF-convergent-derived points
    seed_points = cf_seeded_well_points(N)
    for seed in seed_points:
        if (seed.m, seed.n) not in visited:
            triple = mn_to_triple(seed)
            if triple.c <= upper:
                dist = min(predict_branch(N, (triple.a, triple.b, triple.c)))
                heapq.heappush(heap, (dist, counter, seed))
                counter += 1

    iterations = 0
    while heap and iterations < max_iterations:
        _, _, current = heapq.heappop(heap)
        iterations += 1

        key = (current.m, current.n)
        if key in visited:
            continue
        visited.add(key)

        # Skip invalid PPT parameters
        if current.m <= current.n:
            continue
        if gcd(current.m, current.n) != 1:
            continue
        if (current.m - current.n) % 2 != 1:
            continue

        # Convert to triple
        triple = mn_to_triple(current)
        a, b, c = triple

        # Energy bound check
        if c > upper:
            continue

        # Skip resonance check for triples below N (too small to contain N)
        if c >= N:
            result = resonance_check(N, triple)
            if result is not None:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return (min(p, q), max(p, q))

        # Check direct divisibility even for small triples
        if 1 < a < N and N % a == 0:
            return (min(a, N // a), max(a, N // a))
        if 1 < b < N and N % b == 0:
            return (min(b, N // b), max(b, N // b))

        # Expand children, prioritized by predict_branch distance
        for child in mn_children(current):
            child_key = (child.m, child.n)
            if child_key not in visited and child.m > child.n > 0:
                child_triple = mn_to_triple(child)
                if child_triple.c <= upper:
                    child_dist = min(
                        predict_branch(N, (child_triple.a, child_triple.b, child_triple.c))
                    )
                    heapq.heappush(heap, (child_dist, counter, child))
                    counter += 1

    return None


def inside_out_factor(N: int, max_iterations: int = 500000) -> tuple[int, int] | None:
    """Factor N = p*q using Inside-Out traversal of the Pythagorean tree.

    Uses a multi-strategy approach:
    1. Perfect square detection (O(1))
    2. CF convergent divisibility pre-check (O(log N))
    3. Quick trial division for small factors
    4. CF-steered best-first search (primary)
    5. BFS fallback (guaranteed coverage)
    6. Full trial division (safety net)

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

    # Perfect square detection: if N = p^2, then p is a factor
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # CF convergent divisibility pre-check: the convergents of sqrt(N)
    # often directly reveal factors, especially for close-factor semiprimes.
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        return cf_result

    # Quick trial division for small factors (safety net)
    for p in range(3, min(isqrt(N) + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Strategy 1: CF-steered best-first search (fast for close factors)
    steered_result = _steered_search(N, max_iterations=max_iterations)
    if steered_result is not None:
        return steered_result

    # Strategy 2: BFS from the well (guaranteed coverage for edge cases)
    bfs_result = _bfs_search(N, max_iterations=max_iterations)
    if bfs_result is not None:
        return bfs_result

    # Fallback: trial division up to sqrt(N)
    limit = isqrt(N) + 1
    for p in range(3, limit, 2):
        if N % p == 0:
            return (p, N // p)

    return None


def _bfs_search(N: int, max_iterations: int = 500000) -> tuple[int, int] | None:
    """Factor N using BFS from the well (original Inside-Out algorithm).

    This is the fallback search strategy. It expands the tree level by level
    from the well, checking each node for resonance with N.
    """
    well = central_well(N)
    upper = hypotenuse_bound(N)

    # Seed the BFS with CF-convergent-derived starting points
    seed_points = cf_seeded_well_points(N)

    visited: set[tuple[int, int]] = set()
    queue: deque[MnPair] = deque()

    for seed in seed_points:
        key = (seed.m, seed.n)
        if key not in visited:
            queue.append(seed)
            visited.add(key)

    # Also add root for coverage
    root_mn = MnPair(2, 1)
    if (2, 1) not in visited:
        queue.append(root_mn)
        visited.add((2, 1))

    iterations = 0

    while queue and iterations < max_iterations:
        current = queue.popleft()
        iterations += 1

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

        # Energy bound check
        if c > upper:
            continue

        # Check resonance with N (only for triples large enough)
        if c >= N:
            result = resonance_check(N, triple)
            if result is not None:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return (min(p, q), max(p, q))

        # Check direct divisibility
        if 1 < a < N and N % a == 0:
            return (min(a, N // a), max(a, N // a))
        if 1 < b < N and N % b == 0:
            return (min(b, N // b), max(b, N // b))

        # Expand children using (m,n) Berggren transforms
        for child in mn_children(current):
            child_key = (child.m, child.n)
            if child_key not in visited and child.m > child.n > 0:
                visited.add(child_key)
                # Pre-check: will this child's triple be too large?
                child_triple = mn_to_triple(child)
                if child_triple.c <= upper:
                    queue.append(child)

    return None