"""Pisano Period and Fibonacci-Based Factoring Methods.

Uses the divisibility structure of Fibonacci numbers to factor integers.

Key properties:
- gcd(F_m, F_n) = F_{gcd(m,n)} (proven in Fib_gcd_identity.lean)
- If p | F_n, then the entry point α(p) divides n
- The Pisano period π(N) satisfies π(pq) = lcm(π(p), π(q)) for coprime p,q
- For N = pq, if π(p) != π(q), then gcd(F_{π(p)}, N) gives a factor

This implements Algorithms 29-31 from the SPB Framework's factoring catalog.
"""
from __future__ import annotations

from math import gcd


def fibonacci_mod(n: int, m: int) -> int:
    """Compute F_n mod m using fast doubling.

    Uses the identities:
        F_{2k} = F_k * (2*F_{k+1} - F_k)
        F_{2k+1} = F_k^2 + F_{k+1}^2

    Time complexity: O(log n) multiplications mod m.
    """
    if m == 1:
        return 0
    if n == 0:
        return 0
    if n == 1:
        return 1 % m

    # Fast doubling method
    def _fib_pair(k: int) -> tuple[int, int]:
        """Return (F_k mod m, F_{k+1} mod m)."""
        if k == 0:
            return (0, 1)

        a, b = _fib_pair(k // 2)

        # F_{2k} = F_k * (2*F_{k+1} - F_k)
        c = a * (2 * b - a) % m
        # F_{2k+1} = F_k^2 + F_{k+1}^2
        d = (a * a + b * b) % m

        if k % 2 == 0:
            return (c, d)
        else:
            return (d, (c + d) % m)

    return _fib_pair(n)[0]


def pisano_period(N: int, max_search: int = 100000) -> int | None:
    """Compute the Pisano period π(N): the period of Fibonacci numbers mod N.

    The Pisano period is the smallest k > 0 such that
    F_k == 0 (mod N) and F_{k+1} == 1 (mod N).

    Returns the period, or None if not found within max_search iterations.
    """
    if N == 1:
        return 1

    prev, curr = 0, 1
    for i in range(1, max_search + 1):
        prev, curr = curr, (prev + curr) % N
        if prev == 0 and curr == 1:
            return i
    return None


def entry_point(N: int, max_search: int = 100000) -> int | None:
    """Compute the Fibonacci entry point α(N): the smallest k > 0 with N | F_k.

    For prime p, α(p) divides p-1 (if p == ±1 mod 5) or 2(p+1) (if p == ±2 mod 5).

    Returns the entry point, or None if not found within max_search.
    """
    if N <= 0:
        return None
    if N == 1:
        return 1

    for k in range(1, max_search + 1):
        if fibonacci_mod(k, N) == 0:
            return k
    return None


def pisano_factor(N: int, max_search: int = 100000) -> tuple[int, int] | None:
    """Factor N using the Pisano period method.

    Strategy: compute the Pisano period π(N). For N = pq with coprime p, q,
    π(N) = lcm(π(p), π(q)). If we can find divisors d of π(N) such that
    gcd(F_d mod N, N) reveals a factor, we succeed.

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square fast path
    from math import isqrt
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # Compute Pisano period
    pi_N = pisano_period(N, max_search)
    if pi_N is None:
        return None

    # Try divisors of the Pisano period
    # If d | π(N) and gcd(F_d, N) > 1, we found a factor
    for d in range(1, min(pi_N + 1, 10000)):
        if pi_N % d == 0:
            f_d = fibonacci_mod(d, N)
            if f_d != 0:
                g = gcd(f_d, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    # Also try the period itself
    f_pi = fibonacci_mod(pi_N, N)
    g = gcd(f_pi, N) if f_pi != 0 else 0
    if 1 < g < N:
        return (min(g, N // g), max(g, N // g))

    return None


def fibonacci_entry_point_factor(N: int, max_search: int = 100000) -> tuple[int, int] | None:
    """Factor N using Fibonacci entry points.

    For N = pq, the entry point α(N) = lcm(α(p), α(q)).
    Compute F_{α(N)} mod N and try gcd with N.
    Also try F_k for divisors k of α(N).

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    from math import isqrt
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # Find the entry point
    alpha = entry_point(N, max_search)
    if alpha is None:
        return None

    # Try F_{alpha} directly
    f_alpha = fibonacci_mod(alpha, N)
    g = gcd(f_alpha, N)
    if 1 < g < N:
        return (min(g, N // g), max(g, N // g))

    # Try divisors of the entry point
    for d in range(1, min(alpha + 1, 1000)):
        if alpha % d == 0:
            f_d = fibonacci_mod(d, N)
            if f_d != 0:
                g = gcd(f_d, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def fibonacci_gcd_factor(N: int, bound: int = 10000) -> tuple[int, int] | None:
    """Factor N using Fibonacci GCD method (Pollard rho-like).

    Compute gcd(F_k, N) for k = 1, 2, 3, ... up to bound.
    If p | N and α(p) | k, then p | F_k, so gcd(F_k, N) >= p.

    This is analogous to Pollard's p-1 but using Fibonacci divisibility
    instead of multiplicative order.

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    from math import isqrt
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    prev, curr = 0, 1
    for k in range(1, bound + 1):
        prev, curr = curr, (prev + curr) % N
        if prev == 0:
            continue
        g = gcd(prev, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None