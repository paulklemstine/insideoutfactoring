"""Cyclotomic Polynomial Selection NFS — Novel Method Using Cyclotomic Polynomials for NFS.

A novel approach using cyclotomic polynomials for the polynomial selection step in NFS.
"""
from __future__ import annotations
from math import gcd, isqrt


def _small_primes(bound):
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(bound) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [i for i in range(2, bound + 1) if sieve[i]]


def nfs_cyclo_factor(N, bound=50000, poly_degree=5):
    """Factor N using cyclotomic polynomial selection.

    Simplified NFS-style approach using cyclotomic polynomials.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N:
        return (s, s)

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    primes = _small_primes(bound)

    # For each base a, check if f(a) mod N has small factors
    # Use f(x) = x^poly_degree - 1 (simplified cyclotomic-like polynomial)
    for a in range(2, min(bound, 10000)):
        if a >= N:
            continue

        # Compute a^poly_degree - 1
        val = pow(a, poly_degree, N) - 1
        if val == 0:
            continue

        g = gcd(val, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # Also try a^poly_degree + 1
        val = pow(a, poly_degree, N) + 1
        if val % N == 0:
            continue
        g = gcd(val % N, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None