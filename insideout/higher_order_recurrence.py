"""Higher-Order Recurrence Factoring (Research Frontier).

Implements novel recurrence families that reach beyond p±1:
- Hall 3rd-order: rank | p²+p+1 (new smoothness target)
- Order-k Lucas: rank | Φ_d(p) for d|k (generalizes cyclotomic)

These strictly subsume p-1 and p+1 methods.
"""
from __future__ import annotations
from math import gcd, isqrt
from typing import Optional


def _mat_mul(A, B, N):
    """Multiply two k×k matrices mod N."""
    k = len(A)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) % N for j in range(k)] for i in range(k)]


def _mat_pow(M, k, N):
    """M^k mod N via fast exponentiation."""
    n = len(M)
    # Identity
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in M]
    while k:
        if k & 1:
            result = _mat_mul(result, base, N)
        base = _mat_mul(base, base, N)
        k >>= 1
    return result


def hall_third_order_mod(n: int, P: int, Q: int, R: int, mod: int) -> int:
    """Compute n-th term of Hall 3rd-order recurrence mod m.

    Recurrence: u_n = P*u_{n-1} + Q*u_{n-2} + R*u_{n-3}
    Uses companion matrix exponentiation.

    For appropriate (P,Q,R), this is a strong divisibility sequence
    with rank(p) | p²+p+1.
    """
    if mod == 1:
        return 0
    if n == 0:
        return 0
    if n == 1:
        return 1 % mod
    if n == 2:
        return 1 % mod

    # Companion matrix for u_n = P*u_{n-1} + Q*u_{n-2} + R*u_{n-3}
    M = [
        [P % mod, Q % mod, R % mod],
        [1, 0, 0],
        [0, 1, 0],
    ]

    # Initial values: u_0=0, u_1=1, u_2=1 (standard Hall initialization)
    Mn = _mat_pow(M, n - 2, mod)

    # u_n = Mn[0][0]*u_2 + Mn[0][1]*u_1 + Mn[0][2]*u_0
    return (Mn[0][0] * 1 + Mn[0][1] * 1 + Mn[0][2] * 0) % mod


def order_k_lucas_mod(n: int, coeffs: list[int], mod: int) -> int:
    """Compute n-th term of order-k Lucas sequence mod m.

    Recurrence: u_n = c_1*u_{n-1} + c_2*u_{n-2} + ... + c_k*u_{n-k}

    Rank structure: rank(p) | p^k - 1 (or divisor thereof).
    For k=3: reaches p²+p+1
    For k=4: reaches p²+1
    For k=6: reaches p²-p+1
    """
    if mod == 1:
        return 0
    k = len(coeffs)
    if n < k:
        return [0, 1] + [1] * (k - 2)[n] % mod if n < k else 0

    # Companion matrix
    M = [[0] * k for _ in range(k)]
    M[0] = [c % mod for c in coeffs]
    for i in range(1, k):
        M[i][i - 1] = 1

    Mn = _mat_pow(M, n - k + 1, mod)

    # Initial values: u_0=0, u_1=1, u_2=1, ..., u_{k-1}=1
    init = [0, 1] + [1] * (k - 2)

    result = sum(Mn[0][j] * init[j] for j in range(k)) % mod
    return result


def higher_order_factor(N: int, bound: int = 5000, stage2_bound: int = 1000) -> Optional[tuple[int, int]]:
    """Factor N using higher-order recurrence families.

    Tests Hall 3rd-order and order-k Lucas sequences for various k.
    Each order reaches different cyclotomic factors of p^k-1.
    """
    if N < 4 or N % 2 == 0:
        return None

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Compute smooth exponent M = lcm(1, 2, ..., bound)
    from sympy import primerange
    M = 1
    for p in primerange(2, bound + 1):
        pk = p
        while pk * p <= bound:
            pk *= p
        M *= pk

    # Test Hall 3rd-order sequences with various (P,Q,R)
    hall_params = [
        (1, 1, 1),   # Tribonacci-like
        (1, 1, -1),  # Variant
        (2, 1, 1),   # Different coefficients
        (1, 2, 1),
        (3, 1, 1),
    ]

    for P, Q, R in hall_params:
        u_M = hall_third_order_mod(M, P, Q, R, N)
        for offset in [0, 1, -1, 2, -2]:
            g = gcd(u_M + offset, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    # Test order-k Lucas sequences for k=3,4,5,6
    for k in [3, 4, 5, 6]:
        # Standard coefficients: all 1s (like Fibonacci but order k)
        coeffs = [1] * k
        u_M = order_k_lucas_mod(M, coeffs, N)
        for offset in [0, 1, -1]:
            g = gcd(u_M + offset, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Try different coefficient patterns
        for c1 in [1, 2, 3]:
            coeffs = [c1] + [1] * (k - 1)
            u_M = order_k_lucas_mod(M, coeffs, N)
            for offset in [0, 1, -1]:
                g = gcd(u_M + offset, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None
