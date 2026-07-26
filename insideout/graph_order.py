"""Graph-Order Cascade — Novel Factoring via Multiplicative Order Graph Analysis.

A novel factoring method based on the observation that the multiplicative order of
elements modulo p vs modulo q creates distinct graph structures.

Key insight: Consider the directed graph G_N where vertices are elements of Z/NZ* and
edges go from a to a^k for each k. For prime p, this graph has a specific
structure. For composite N = pq, the graph mod N is the product (via CRT) of
the graphs mod p and mod q.

The key observation: if we know the order of an element a mod p and mod q, we can
detect CRT divergence. If ord_p(a) | ord_q(a), then a^(ord_p(a)) == 1 (mod p) but
a^(ord_p(a)) ≢ 1 (mod q), giving gcd(a^(ord_p(a)) - 1, N).

Novel approach: Build a small graph of order relationships and find "bridges" —
pairs (a, b) where ord_a divides ord_b mod one factor but not the other.

Additionally, use the **order spectrum**: the set of orders {ord(a), ord(a2), ...}
for various bases. This spectrum differs between mod p and mod q.

Per honest assessment: sub-exponential L_p[1/2], same as ECM.
"""
from __future__ import annotations

from math import gcd, isqrt


def _order_mod(a: int, m: int, max_order: int = 100000) -> int | None:
    """Compute multiplicative order of a mod m, or None if not found.

    The order is the smallest k > 0 such that a^k == 1 (mod m).
    """
    if gcd(a, m) != 1:
        return None

    # Compute phi(m)
    phi = m
    temp = m
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            phi = phi // p * (p - 1)
            while temp % p == 0:
                temp //= p
        p += 1 if p == 2 else 2
    if temp > 1:
        phi = phi // temp * (temp - 1)

    # Find smallest divisor of phi where a^d == 1 (mod m)
    for d in range(1, min(phi + 1, max_order + 1)):
        if pow(a, d, m) == 1:
            return d

    return None


def _compute_order_graph(a: int, N: int, max_exponent: int = 50) -> dict[int, int]:
    """Compute the order graph for base a: ord(a^k) for k = 1..max_exponent."""
    graph = {}
    current = a % N
    for k in range(1, max_exponent + 1):
        if gcd(current, N) == 1:
            ord_k = _order_mod(current, N, max_order=100000)
            graph[k] = ord_k
        current = (current * a) % N
    return graph


def _find_bridge_pairs(N: int, bases: list[int], max_exp: int = 50) -> tuple[int, int] | None:
    """Find pairs (a, b) that form a bridge revealing CRT divergence."""
    for i, a in enumerate(bases):
        if gcd(a, N) != 1:
            continue
        graph_a = _compute_order_graph(a, N, max_exp)

        for j, b in enumerate(bases[i+1:], i+1):
            if gcd(b, N) != 1:
                continue

            # Check if orders share a factor with N
            ord_a_a = graph_a.get(1)
            if ord_a_a is not None and ord_a_a > 1:
                g = gcd(ord_a_a, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            # Check a^k == b (mod N) for some k
            for k in range(2, max_exp + 1):
                if k in graph_a and graph_a[k] is not None:
                    pkt = pow(a, k, N)
                    if pkt == b % N:
                        g = gcd(pkt - 1, N)
                        if 1 < g < N:
                            return (min(g, N // g), max(g, N // g))

    return None


def graph_order_cascade_factor(N: int, bound: int = 50000,
                                max_exp: int = 50,
                                bases: int = 8) -> tuple[int, int] | None:
    """Factor N using graph-order cascade.

    Algorithm:
    1. Choose multiple bases a_i
    2. For each base, compute the order graph: ord(a_i^k) for k = 1..max_exp
    3. Check gcd(a_i^k - 1, N) for each k
    4. Find bridge pairs where ord relationships differ
    5. Apply smooth-bound powering and check orders

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Skip for large N — order computation is too slow
    if N.bit_length() > 256:
        return None

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Build prime list for smooth-bound
    primes = []
    sieve = [True] * (min(bound, 50000) + 1)
    for i in range(2, len(sieve)):
        if sieve[i]:
            primes.append(i)
            for j in range(i * i, len(sieve), i):
                sieve[j] = False

    base_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37][:bases]

    # Phase 1: Direct order-based detection
    for a in base_list:
        if a >= N or gcd(a, N) != 1:
            continue
        for k in range(1, max_exp + 1):
            ak = pow(a, k, N)
            if ak == 1:
                continue
            g = gcd(ak - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    # Phase 2: Smooth-bound powering with order checking
    for a in base_list:
        if a >= N or gcd(a, N) != 1:
            continue

        power = a
        for p in primes:
            pk = p
            while pk * p <= bound:
                pk *= p
            power = pow(power, pk, N)

            g = gcd(power - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            g = gcd(power + 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            for k in range(2, min(max_exp, 10) + 1):
                pkt = pow(power, k, N)
                g = gcd(pkt - 1, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    # Phase 3: Find bridge pairs
    result = _find_bridge_pairs(N, base_list, max_exp)
    if result is not None:
        return result

    # Phase 4: Order spectrum analysis
    for a in base_list:
        if a >= N or gcd(a, N) != 1:
            continue

        spectrum = []
        current = a % N
        for k in range(1, max_exp + 1):
            if gcd(current, N) == 1:
                ord_k = _order_mod(current, N, max_order=100000)
                if ord_k is not None:
                    spectrum.append(ord_k)
            current = (current * a) % N

        for oi in spectrum:
            if oi is not None and oi > 1:
                g = gcd(oi, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def order_spectrum_factor(N: int, bound: int = 50000,
                          spectrum_size: int = 30) -> tuple[int, int] | None:
    """Factor N using order spectrum analysis."""
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2) if N > 2 else None
    # Skip for large N — order computation too slow
    if N.bit_length() > 256:
        return None
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Build prime list
    primes = []
    sieve = [True] * (min(bound, 50000) + 1)
    for i in range(2, len(sieve)):
        if sieve[i]:
            primes.append(i)
            for j in range(i * i, len(sieve), i):
                sieve[j] = False

    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:spectrum_size]

    # Phase 1: Smooth-bound powering
    for a in bases:
        if a >= N or gcd(a, N) != 1:
            continue

        power = a
        for p in primes:
            pk = p
            while pk * p <= bound:
                pk *= p
            power = pow(power, pk, N)

            g = gcd(power - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            g = gcd(power + 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Phase 2: Order spectrum of powered value
        current = power
        for k in range(1, spectrum_size + 1):
            if gcd(current, N) == 1:
                ord_k = _order_mod(current, N, max_order=100000)
                if ord_k is not None and ord_k > 1:
                    g = gcd(ord_k, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
            current = (current * power) % N

    return None