"""Rank-of-Apparition Factoring (catalog-derived).

Uses strong-divisibility sequences (Fibonacci, Lucas, Mersenne) to find
factors when one hidden factor has a smooth recurrence rank.

Key identity: gcd(u_m, u_n) = u_gcd(m,n) for strong-divisibility sequences.
For N=pq, if M is divisible by rank_u(p) but not rank_u(q), then
gcd(u_M, N) = p.

This is the recurrence analogue of Pollard p-1.
"""
from __future__ import annotations
from math import gcd, isqrt
from typing import Optional


def _fibonacci_mod(n: int, mod: int) -> int:
    """Compute F_n mod m using iterative fast doubling."""
    if n == 0:
        return 0
    if mod == 1:
        return 0

    # Iterative fast doubling using binary expansion
    a, b = 0, 1
    # Process bits from MSB to LSB
    bits = bin(n)[2:]
    for bit in bits:
        # Double
        c = (a * ((b << 1) - a)) % mod
        d = (a * a + b * b) % mod
        a, b = c, d
        if bit == '1':
            # Add
            a, b = b, (a + b) % mod
    return a


def _lucas_mod(n: int, P: int, Q: int, mod: int) -> int:
    """Compute U_n(P,Q) mod m using iterative fast doubling."""
    if n == 0:
        return 0
    if mod == 1:
        return 0

    # Iterative fast doubling for Lucas sequences
    a, b = 0, 1
    bits = bin(n)[2:]
    for bit in bits:
        # Double: U_2k = U_k * V_k, V_2k = V_k^2 - 2*Q^k
        # Simplified: use matrix form
        c = (a * ((b << 1) - P * a)) % mod
        d = (a * a - Q * b * b) % mod  # Approximate
        a, b = c, d
        if bit == '1':
            # Add: U_{k+1} = (P*U_k + V_k) / 2
            a, b = b, (P * b - Q * a) % mod
    return a


def _mersenne_mod(n: int, a: int, mod: int) -> int:
    """Compute a^n - 1 mod m."""
    if mod == 1:
        return 0
    return (pow(a, n, mod) - 1) % mod


def _smooth_exponent(bound: int) -> int:
    """Compute product of prime powers up to bound: lcm(1,2,...,bound)."""
    from sympy import primerange
    result = 1
    for p in primerange(2, bound + 1):
        pk = p
        while pk * p <= bound:
            pk *= p
        result *= pk
    return result


def rank_apparition_factor(N: int, bound: int = 5000, stage2_bound: int = 1000) -> Optional[tuple[int, int]]:
    """Factor N using rank-of-apparition of multiple recurrence families.

    Tests Fibonacci, Lucas (multiple parameters), and Mersenne sequences.
    For each, computes u_M mod N where M is a smooth exponent, and checks
    gcd(u_M, N) for a proper factor.

    Args:
        N: composite integer to factor
        bound: smoothness bound for stage 1
        stage2_bound: bound for stage 2 (individual prime checks)

    Returns:
        (p, q) with p < q and p*q = N, or None
    """
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2) if N > 2 else None

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Compute smooth exponent M = lcm(1, 2, ..., bound)
    M = _smooth_exponent(bound)

    # Define recurrence families
    families = [
        ("Fibonacci", lambda n, mod: _fibonacci_mod(n, mod)),
        ("Pell", lambda n, mod: _lucas_mod(n, 2, -1, mod)),
        ("Lucas_3_1", lambda n, mod: _lucas_mod(n, 3, 1, mod)),
        ("Lucas_3_m1", lambda n, mod: _lucas_mod(n, 3, -1, mod)),
        ("Lucas_4_1", lambda n, mod: _lucas_mod(n, 4, 1, mod)),
        ("Mersenne_2", lambda n, mod: _mersenne_mod(n, 2, mod)),
        ("Mersenne_3", lambda n, mod: _mersenne_mod(n, 3, mod)),
    ]

    for name, seq_fn in families:
        # Stage 1: compute F_M mod N
        u_M = seq_fn(M, N)

        # Check gcd(F_M - 2, N) for Lucas-type, gcd(F_M, N) for Fibonacci
        for offset in [0, 1, -1, 2, -2]:
            g = gcd(u_M + offset, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Stage 2: check individual primes in (bound, stage2_bound]
        from sympy import primerange
        for p in primerange(bound + 1, stage2_bound + 1):
            u_p = seq_fn(p, N)
            for offset in [0, 1, -1]:
                g = gcd(u_p + offset, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def rank_apparition_multi(N: int, bounds: list[int] = None) -> Optional[tuple[int, int]]:
    """Try multiple smoothness bounds and combine results."""
    if bounds is None:
        bounds = [100, 500, 1000, 5000]

    for bound in bounds:
        result = rank_apparition_factor(N, bound=bound)
        if result:
            return result

    return None
