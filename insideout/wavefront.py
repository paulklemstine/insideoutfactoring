"""Wavefront parallel search for Inside-Out factoring.

Instead of evaluating nodes one at a time, group all nodes at increasing
energy distance from the well into wavefronts. Each wavefront can be
evaluated in parallel, with modular resonance filters applied to the
entire batch before individual resonance checks.
"""
from __future__ import annotations

from collections import deque
from math import gcd, isqrt
from typing import Iterator

from .berggren import Triple
from .gaussian import MnPair, mn_to_triple, mn_children
from .energy import is_energy_compatible
from .inside_out import resonance_check, central_well, cf_seeded_well_points
from .cf_guide import cf_factor_check


def expand_wavefront(
    N: int,
    max_batches: int = 100,
    batch_size: int = 1000,
) -> Iterator[list[Triple]]:
    """Generate wavefronts of triples at increasing energy distance.

    Yields lists of triples, where each list represents all triples
    discovered at a given depth from the well. Later batches have
    higher energy (larger hypotenuse).
    """
    # Use CF-seeded well points for better starting coverage
    seed_points = cf_seeded_well_points(N)
    visited: set[tuple[int, int]] = set()
    queue: deque[MnPair] = deque()

    for seed in seed_points:
        key = (seed.m, seed.n)
        if key not in visited:
            queue.append(seed)

    # Upper bound on hypotenuse for energy filtering
    upper = (N * N + 1) // 2

    for _ in range(max_batches):
        batch: list[Triple] = []
        next_queue: deque[MnPair] = deque()

        # Process current level
        processed = 0
        while queue and processed < batch_size:
            current = queue.popleft()
            processed += 1

            key = (current.m, current.n)
            if key in visited:
                continue
            visited.add(key)

            if current.m <= current.n:
                continue

            # Valid PPT parameters
            if (current.m - current.n) % 2 == 1 and gcd(current.m, current.n) == 1:
                triple = mn_to_triple(current)

                # Energy filter: only include triples in the valid range
                # c must be >= N (lower bound) and <= (N^2+1)/2 (upper bound)
                if N <= triple.c <= upper:
                    batch.append(triple)

            # Add children to next level
            for child in mn_children(current):
                if child.m > child.n > 0:
                    child_key = (child.m, child.n)
                    if child_key not in visited:
                        next_queue.append(child)

        if batch:
            yield batch

        queue = next_queue
        if not queue:
            break


def search_wavefront(
    N: int,
    max_radius: int = 1000,
) -> tuple[int, int] | None:
    """Factor N using wavefront search.

    Expands wavefronts from the energy well, checking each triple
    for resonance with N.
    """
    if N < 4:
        return None

    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square detection
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # CF convergent divisibility pre-check
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        return cf_result

    for batch in expand_wavefront(N, max_batches=max_radius):
        for triple in batch:
            result = resonance_check(N, triple)
            if result is not None:
                p, q = result
                if 1 < p < N and p * q == N:
                    return (min(p, q), max(p, q))

            # Also check direct divisibility
            a, b, c = triple
            if a > 1 and N % a == 0 and a < N:
                return (a, N // a)
            if b > 1 and N % b == 0 and b < N:
                return (b, N // b)

    return None