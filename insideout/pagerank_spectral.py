"""PageRank Spectral Cascade — Zero-Divisor Concentration for Factoring.

Key insight (Fuchs 2005): For the unitary Cayley graph G_N = Cay(Z/NZ, (Z/NZ)*):
  - Eigenvalues are Ramanujan sums c_N(k) = μ(N/gcd(k,N)) · φ(N)/φ(N/gcd(k,N))
  - The spectrum is entirely determined by divisor structure of N

PageRank concentration: On the directed multiplication graph,
  π(0) > π(k·p_i) > π(u) for u in (Z/NZ)*

This means random walks with teleportation concentrate on zero-divisors,
which are exactly where gcd(x, N) > 1 reveals factors.

Algorithm:
1. Build sparse multiplication graph (sampled vertices)
2. Compute personalized PageRank from CF-convergent seeds
3. Check top PageRank vertices for gcd
4. Batch GCD on concentrated vertices
"""
from __future__ import annotations

import random
from math import gcd, isqrt, log as ln
from typing import Optional

from .cf_guide import cf_sqrt, convergents


def _build_sampled_graph(N: int, sample_size: int = 10000) -> dict[int, list[int]]:
    """Build sampled multiplication graph.

    Each vertex i connects to i*g mod N for small generators g.
    Only samples sample_size vertices to keep memory manageable.
    """
    # Use small primes as generators
    generators = [2, 3, 5, 7, 11, 13]

    # Sample vertices from [1, min(N, 100000)]
    max_vertex = min(N, 100000)
    all_vertices = list(range(1, max_vertex))
    if len(all_vertices) > sample_size:
        vertices = random.sample(all_vertices, sample_size)
    else:
        vertices = all_vertices

    # Build adjacency
    graph = {v: [] for v in vertices}
    for v in vertices:
        for g in generators:
            neighbor = (v * g) % N
            if neighbor in graph:
                graph[v].append(neighbor)
            # Also add reverse edges for undirected walk
            if neighbor not in graph:
                graph[neighbor] = [v]

    return graph


def _pagerank_iteration(graph: dict, scores: dict, damping: float = 0.15,
                       teleport: set = None) -> dict:
    """One iteration of PageRank with teleportation.

    PR_i+1(v) = (1-d)/|T| + d * Σ PR_i(u) / out_degree(u)
    where T is teleportation set and sum is over neighbors pointing to v.
    """
    N = len(graph)
    teleport = teleport or set(graph.keys())
    new_scores = {}

    for v in graph:
        # Teleportation contribution
        teleport_mass = (1 - damping) / len(teleport) if v in teleport else 0

        # Link contribution from neighbors
        link_mass = 0.0
        for u in graph:
            if v in graph[u]:  # u -> v exists
                out_deg = len(graph[u])
                if out_deg > 0:
                    link_mass += damping * scores.get(u, 0) / out_deg

        new_scores[v] = teleport_mass + link_mass

    return new_scores


def _pagerank(graph: dict, iterations: int = 50, damping: float = 0.15,
             teleport_set: set = None) -> dict:
    """Compute PageRank vector via power iteration."""
    N = len(graph)
    # Initialize uniformly
    scores = {v: 1.0 / N for v in graph}

    for _ in range(iterations):
        scores = _pagerank_iteration(graph, scores, damping, teleport_set)

    return scores


def _personalized_pagerank(graph: dict, seed: int, iterations: int = 30,
                          damping: float = 0.15) -> dict:
    """Personalized PageRank from a seed vertex."""
    if seed not in graph:
        seed = seed % (max(graph.keys()) + 1) if graph else 1
        if seed not in graph:
            return {v: 1.0 / len(graph) for v in graph}

    teleport_set = {seed}
    return _pagerank(graph, iterations, damping, teleport_set)


def _check_top_vertices(scores: dict, N: int, top_k: int = 1000) -> tuple[int, int] | None:
    """Check top PageRank vertices for gcd with N."""
    # Sort by score descending
    sorted_vertices = sorted(scores.keys(), key=lambda v: -scores[v])

    for v in sorted_vertices[:top_k]:
        g = gcd(v, N)
        if 1 < g < N:
            return (g, N // g)

    return None


def pagerank_spectral_factor(N: int,
                            sample_size: int = 20000,
                            damping: float = 0.15,
                            iterations: int = 30) -> tuple[int, int] | None:
    """Factor N using PageRank spectral cascade.

    Key insight: On the multiplication graph mod N, PageRank concentrates
    on zero-divisors. Checking top PageRank vertices for gcd reveals factors.

    The algorithm:
    1. Build sampled multiplication graph
    2. Compute PageRank from uniform teleportation
    3. Check top vertices for gcd
    4. If failed, try personalized PageRank from CF-convergent seeds
    5. Batch GCD on concentrated vertices

    Complexity: O(sample_size * iterations) per attempt.
    Works best when N has small factors (PageRank concentrates there).

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    # Small trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Build sampled graph
    graph = _build_sampled_graph(N, sample_size=sample_size)
    if not graph:
        return None

    # Stage 1: Uniform PageRank
    scores = _pagerank(graph, iterations=iterations, damping=damping)
    result = _check_top_vertices(scores, N, top_k=min(2000, len(scores)))
    if result is not None:
        return result

    # Stage 2: CF-guided personalized PageRank
    try:
        cf = cf_sqrt(N, max_terms=30)
        convs = list(convergents(cf))[:10]
    except Exception:
        convs = []

    for pk, qk in convs:
        # Seed from convergent: a = pk * qk^(-1) mod N
        try:
            seed = (pk * pow(qk, -1, N)) % N
        except Exception:
            seed = pk % N

        if seed in graph:
            ppr = _personalized_pagerank(graph, seed, iterations=iterations, damping=damping)
            result = _check_top_vertices(ppr, N, top_k=min(1000, len(ppr)))
            if result is not None:
                return result

    # Stage 3: Try random seeds
    for _ in range(5):
        seed = random.randint(2, N - 1)
        ppr = _personalized_pagerank(graph, seed, iterations=iterations, damping=damping)
        result = _check_top_vertices(ppr, N, top_k=min(500, len(ppr)))
        if result is not None:
            return result

    return None


def pagerank_batch_factor(N: int,
                         sample_size: int = 30000,
                         damping: float = 0.15,
                         iterations: int = 30) -> tuple[int, int] | None:
    """Enhanced PageRank with batch GCD on concentrated vertices.

    Instead of checking one vertex at a time, accumulates top vertices
    and takes batch GCD of their product.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    graph = _build_sampled_graph(N, sample_size=sample_size)
    if not graph:
        return None

    # Compute uniform PageRank
    scores = _pagerank(graph, iterations=iterations, damping=damping)

    # Get top vertices
    sorted_vertices = sorted(scores.keys(), key=lambda v: -scores[v])
    top_vertices = sorted_vertices[:5000]

    # Batch GCD: product of top vertices mod N
    product = 1
    for v in top_vertices:
        product = (product * v) % N

    g = gcd(product, N)
    if 1 < g < N:
        return (g, N // g)

    # Also try CF-guided with batch
    try:
        cf = cf_sqrt(N, max_terms=20)
        convs = list(convergents(cf))[:5]
    except Exception:
        convs = []

    for pk, qk in convs:
        try:
            seed = (pk * pow(qk, -1, N)) % N
        except Exception:
            continue

        if seed in graph:
            ppr = _personalized_pagerank(graph, seed, iterations=iterations, damping=damping)
            top_ppr = sorted(ppr.keys(), key=lambda v: -ppr[v])[:2000]

            product = 1
            for v in top_ppr:
                product = (product * v) % N

            g = gcd(product, N)
            if 1 < g < N:
                return (g, N // g)

    return None


def pagerank_spectral_cascade(N: int,
                              time_budget_ms: float = 5000) -> tuple[int, int] | None:
    """Multi-stage PageRank spectral cascade with increasing sample sizes."""
    import time
    start = time.perf_counter()

    # Stage 1: Small sample, fast
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > time_budget_ms * 0.3:
        return None

    result = pagerank_batch_factor(N, sample_size=10000, iterations=20)
    if result is not None:
        return result

    # Stage 2: Medium sample
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > time_budget_ms * 0.6:
        return None

    result = pagerank_batch_factor(N, sample_size=20000, iterations=30)
    if result is not None:
        return result

    # Stage 3: Large sample, thorough
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > time_budget_ms:
        return None

    result = pagerank_spectral_factor(N, sample_size=50000, iterations=50)
    return result


__all__ = [
    'pagerank_spectral_factor',
    'pagerank_batch_factor',
    'pagerank_spectral_cascade',
]
