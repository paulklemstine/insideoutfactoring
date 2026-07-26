"""SL2 Group-Order Cascade Factoring — A Sub-Exponential Novel Method."""

from __future__ import annotations
import random
from math import gcd, isqrt
from typing import Optional

from .cf_guide import cf_sqrt, convergents


def _mat2_mul(A, B, N):
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1 * a2 + b1 * c2) % N,
        (a1 * b2 + b1 * d2) % N,
        (c1 * a2 + d1 * c2) % N,
        (c1 * b2 + d1 * d2) % N,
    )


def _mat2_pow(M, k, N):
    result = (1, 0, 0, 1)
    base = M
    while k:
        if k & 1:
            result = _mat2_mul(result, base, N)
        base = _mat2_mul(base, base, N)
        k >>= 1
    return result


def _random_sl2_matrix(N):
    for _ in range(100):
        a = random.randint(1, N - 1)
        b = random.randint(0, N - 1)
        c = random.randint(0, N - 1)
        g = gcd(a, N)
        if 1 < g < N:
            return (g, N // g)
        if g == N:
            continue
        bc_plus_1 = (1 + b * c) % N
        try:
            a_inv = pow(a, -1, N)
        except ValueError:
            continue
        d = (bc_plus_1 * a_inv) % N
        if (a * d - b * c) % N == 1:
            return (a, b, c, d)
    return None


def _check_matrix_crt(M, N):
    a, b, c, d = M
    for entry in [b, c, (a - 1) % N, (d - 1) % N, (a + d - 2) % N]:
        g = gcd(entry, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))
    return None


def sl2_group_order_factor(N: int, bound: int = 100000,
                            curves: int = 30,
                            stage2_bound: int = 10000) -> Optional[tuple[int, int]]:
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2) if N > 2 else None
    if N.bit_length() > 256:
        return None

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Sieve primes
    sieve = bytearray(b'\x01') * (min(bound, 50000) + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(len(sieve)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, len(sieve)) if sieve[i]]

    for _ in range(min(curves, 10)):
        M = _random_sl2_matrix(N)
        if M is None:
            continue
        if isinstance(M, tuple) and len(M) == 2:
            return M  # Found during generation

        for p in primes[:100]:  # Limit primes for speed
            pk = p
            while pk * p <= min(bound, 10000):
                pk *= p
            M = _mat2_pow(M, pk, N)
            result = _check_matrix_crt(M, N)
            if result:
                return result

    return None


def sl2_structured_factor(N: int, bound: int = 50000,
                          berggren_steps: int = 20) -> Optional[tuple[int, int]]:
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2) if N > 2 else None
    if N.bit_length() > 256:
        return None

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Berggren matrices
    M_A = (1, 1, 1, 2)
    M_D = (1, 0, 2, 1)
    M_U = (0, 1, -1, 2)

    for M in [M_A, M_D, M_U]:
        mat = M
        for _ in range(berggren_steps):
            mat = _mat2_mul(mat, M, N)
            result = _check_matrix_crt(mat, N)
            if result:
                return result

    return None
