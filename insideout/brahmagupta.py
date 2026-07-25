"""Brahmagupta-Fibonacci Two-Square Factoring Method.

If N can be written as a sum of two squares in two different ways:
    N = a² + b² = c² + d²
then N divides (ad - bc)(ad + bc), and gcd(ad ± bc, N) may reveal factors.

This is Algorithm 18 from the SPB Framework's factoring catalog.
The method is particularly effective for numbers that are products of primes
congruent to 1 mod 4 (since these always have two-square representations by
Fermat's theorem on sums of two squares).
"""
from __future__ import annotations

from math import gcd, isqrt


def find_two_square_representation(N: int) -> tuple[int, int] | None:
    """Find one representation of N as a sum of two squares.

    Uses the method of checking if N - a² is a perfect square for
    increasing a. Returns (a, b) with a ≤ b and a² + b² = N,
    or None if no such representation exists.

    Only works for N where all prime factors of the form 4k+3
    appear to an even power (Fermat's two-square theorem).
    """
    if N < 0:
        return None
    if N == 0:
        return (0, 0)

    for a in range(isqrt(N) + 1):
        remainder = N - a * a
        b = isqrt(remainder)
        if b * b == remainder and a <= b:
            return (a, b)
    return None


def find_all_two_square_representations(N: int, max_reps: int = 10) -> list[tuple[int, int]]:
    """Find all representations of N as a sum of two squares.

    Returns list of (a, b) pairs with a ≤ b and a² + b² = N,
    up to max_reps representations.
    """
    if N < 0:
        return []
    if N == 0:
        return [(0, 0)]

    reps: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for a in range(isqrt(N) + 1):
        remainder = N - a * a
        if remainder < 0:
            break
        b = isqrt(remainder)
        if b * b == remainder and a <= b:
            key = (a, b)
            if key not in seen:
                seen.add(key)
                reps.append(key)
                if len(reps) >= max_reps:
                    break
    return reps


def brahmagupta_fibonacci_factor(N: int) -> tuple[int, int] | None:
    """Factor N using the Brahmagupta-Fibonacci two-square identity.

    If N = a² + b² = c² + d² with (a,b) ≠ (c,d), then:
        N | (ad - bc)(ad + bc)

    So gcd(ad - bc, N) or gcd(ad + bc, N) may yield a nontrivial factor.

    Returns (p, q) with p < q and p*q = N, or None if no factor found.
    """
    if N < 4:
        return None

    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    reps = find_all_two_square_representations(N, max_reps=20)

    if len(reps) < 2:
        # Need at least two distinct representations
        return None

    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i]
            c, d = reps[j]

            # Skip identical representations
            if (a, b) == (c, d):
                continue

            # Brahmagupta-Fibonacci identity:
            # (a² + b²)(c² + d²) = (ac ± bd)² + (ad ∓ bc)²
            # But for factoring, we use:
            # N | (ad - bc)(ad + bc)
            x = a * d - b * c
            y = a * d + b * c

            if x != 0:
                g = gcd(abs(x), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            if y != 0 and y != N:
                g = gcd(abs(y), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            # Also try the other combination: (ac - bd)(ac + bd)
            x2 = a * c - b * d
            y2 = a * c + b * d

            if x2 != 0:
                g = gcd(abs(x2), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            if y2 != 0 and y2 != N:
                g = gcd(abs(y2), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def fermat_difference_of_squares(N: int) -> tuple[int, int] | None:
    """Factor N using Fermat's difference of squares method.

    For N = pq with p ≤ q, if we find s² - t² = N, then
    (s - t)(s + t) = N and p = s - t, q = s + t.

    Starts from s = ceil(sqrt(N)) and searches upward.
    Optimized for close factors (p ~ q) where s is near sqrt(N).
    """
    if N < 4:
        return None

    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s < N:
        s += 1

    # Search for s such that s² - N is a perfect square
    max_s = (N + 1) // 2  # Upper bound: s + t ≤ N
    max_iterations = min(max_s - s + 1, 100000)

    for _ in range(max_iterations):
        t_sq = s * s - N
        if t_sq < 0:
            s += 1
            continue

        t = isqrt(t_sq)
        if t * t == t_sq and t < s:
            # Found: N = (s-t)(s+t)
            p = s - t
            q = s + t
            if 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        s += 1

    return None


def sigma_cryptanalysis(N: int, sigma_N: int | None = None) -> tuple[int, int] | None:
    """Factor N using the divisor sum σ(N).

    For N = pq with p, q distinct primes:
        σ(N) = (1 + p)(1 + q) = 1 + p + q + pq = 1 + p + q + N

    So p + q = σ(N) - N - 1, and pq = N.
    This gives p and q as roots of x² - (p+q)x + N = 0.

    For N = p² (perfect square):
        σ(N) = 1 + p + p², so p = σ(N) - N - 1.

    If sigma_N is not provided, computes it (requires factoring N first,
    so this method is only useful when σ(N) is known from another source).
    """
    if N < 4:
        return None

    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square fast path
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    if sigma_N is None:
        # Without knowing σ(N), we can't use this method
        # (computing σ requires factoring, which defeats the purpose)
        return None

    # p + q = σ(N) - N - 1
    sum_pq = sigma_N - N - 1
    if sum_pq < 2:
        return None

    # p and q are roots of x² - (p+q)x + N = 0
    # discriminant = (p+q)² - 4N = (p-q)²
    discriminant = sum_pq * sum_pq - 4 * N

    if discriminant < 0:
        return None

    sqrt_disc = isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        return None

    # p = ((p+q) + (p-q)) / 2, q = ((p+q) - (p-q)) / 2
    p = (sum_pq + sqrt_disc) // 2
    q = (sum_pq - sqrt_disc) // 2

    if p * q == N and 1 < p < N and 1 < q < N:
        return (min(p, q), max(p, q))

    return None