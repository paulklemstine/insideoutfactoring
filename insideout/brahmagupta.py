"""Brahmagupta-Fibonacci Two-Square Factoring Method.

If N can be written as a sum of two squares in two different ways:
    N = a2 + b2 = c2 + d2
then N divides (ad - bc)(ad + bc), and gcd(ad ± bc, N) may reveal factors.

This is Algorithm 18 from the SPB Framework's factoring catalog.
The method is particularly effective for numbers that are products of primes
congruent to 1 mod 4 (since these always have two-square representations by
Fermat's theorem on sums of two squares).
"""
from __future__ import annotations

from math import gcd, isqrt

# CF-guided Fermat needs cf_sqrt and convergents
try:
    from .cf_guide import cf_sqrt, convergents
    _HAS_CF_GUIDE = True
except ImportError:
    _HAS_CF_GUIDE = False


def find_two_square_representation(N: int) -> tuple[int, int] | None:
    """Find one representation of N as a sum of two squares.

    Uses the method of checking if N - a2 is a perfect square for
    increasing a. Returns (a, b) with a <= b and a2 + b2 = N,
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


def find_all_two_square_representations(N: int, max_reps: int = 10, max_iter: int = 100000) -> list[tuple[int, int]]:
    """Find representations of N as a sum of two squares.

    Returns list of (a, b) pairs with a <= b and a2 + b2 = N,
    up to max_reps representations, stopping after at most max_iter checks.
    """
    if N < 0:
        return []
    if N == 0:
        return [(0, 0)]

    reps: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    # Iterate with a HARD limit to prevent catastrophic slowdown
    # For 56-bit N, isqrt(N) ~= 2.6e8, so we cap at max_iter
    limit = min(isqrt(N) + 1, max_iter)

    for a in range(limit):
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

    If N = a2 + b2 = c2 + d2 with (a,b) != (c,d), then:
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
            # (a2 + b2)(c2 + d2) = (ac ± bd)2 + (ad ∓ bc)2
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


def _cf_fermat_start(N: int, max_convergents: int = 30) -> tuple[int, int, int]:
    """Find CF-guided starting s for Fermat's method.

    Returns (best_s, best_error, num_convergents_checked).
    Uses CF convergents of sqrtN to find a starting s close to the true (p+q)/2.
    """
    if not _HAS_CF_GUIDE:
        s = isqrt(N)
        if s * s < N:
            s += 1
        return s, s, 0

    try:
        cf = cf_sqrt(N, max_terms=max_convergents)
        convs = convergents(cf)
    except Exception:
        s = isqrt(N)
        if s * s < N:
            s += 1
        return s, s, 0

    s0 = isqrt(N)
    if s0 * s0 < N:
        s0 += 1

    best_s = s0
    best_error = abs(s0 * s0 - N)  # squared distance to perfect square

    for p_k, q_k in convs[:max_convergents]:
        # Candidate s from convergent: s = (p_k + q_k) / 2
        if (p_k + q_k) % 2 == 0:
            s_candidate = (p_k + q_k) // 2
        else:
            s_candidate = (p_k + q_k + 1) // 2

        if s_candidate < s0:
            continue

        # Compute t2 = s2 - N
        t_sq = s_candidate * s_candidate - N
        if t_sq < 0:
            continue

        t = isqrt(t_sq)
        if t * t == t_sq:
            # Exact match! Found factor directly
            p = s_candidate - t
            q = s_candidate + t
            if 1 < p < N and 1 < q < N and p * q == N:
                return (min(p, q), 0, 0)  # Signal: direct factor found

        # Track error: how close is s to the solution?
        error = s_candidate - s0
        if error > 0 and error < best_error:
            best_error = error
            best_s = s_candidate

    return best_s, best_error, len(convs[:max_convergents])


def fermat_difference_of_squares(N: int,
                                 max_iterations: int = 100000) -> tuple[int, int] | None:
    """Factor N using Fermat's difference of squares method with CF-guided stepping.

    Enhanced version: uses continued fraction convergents of sqrtN to:
    1. Find a near-optimal starting s (reduces iterations when factors are close)
    2. Adapt step size based on CF convergent gaps
    3. Search in a band around the best convergent when initial search fails

    For N = pq with p <= q, s = (p+q)/2, t = (q-p)/2.
    When factors are close (q-p is small), s is near sqrtN and few iterations suffice.
    When factors are far, the CF-guided band search provides a second chance.

    Returns (p, q) with p < q and p*q = N, or None if no factor found.
    """
    if N < 4:
        return None

    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # CF-guided starting point
    best_s, best_error, num_cf = _cf_fermat_start(N)

    # If CF found a direct factor, return it
    if best_error == 0 and best_s * best_s == N:
        return (best_s, best_s)

    # Adaptive step size: if CF gave a good starting point, use smaller steps
    # near it; otherwise use standard unit steps
    if num_cf > 0 and best_error < 1000:
        # CF gave a close starting point — search carefully in a band
        band_radius = max(100, num_cf * 5)
    else:
        band_radius = 0  # No CF guidance available

    # Phase 1: Search starting from CF-guided s with adaptive band
    if band_radius > 0:
        for delta in range(-band_radius, band_radius + 1):
            s = best_s + delta
            if s < isqrt(N):
                continue

            t_sq = s * s - N
            if t_sq < 0:
                continue

            t = isqrt(t_sq)
            if t * t == t_sq and t < s:
                p = s - t
                q = s + t
                if 1 < p < N and 1 < q < N:
                    return (min(p, q), max(p, q))

    # Phase 2: Standard Fermat search from best_s
    s = best_s
    max_s = (N + 1) // 2
    actual_max = min(max_s, s + max_iterations)

    for _ in range(max(s, 1), actual_max + 1):
        t_sq = s * s - N
        if t_sq < 0:
            s += 1
            continue

        t = isqrt(t_sq)
        if t * t == t_sq and t < s:
            p = s - t
            q = s + t
            if 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        s += 1

    # Phase 3: Last resort — try CF band around s0 = ceil(sqrt(N))
    if band_radius == 0:
        s0 = isqrt(N)
        if s0 * s0 < N:
            s0 += 1
        for delta in range(-1000, 1001):
            s = s0 + delta
            if s < s0:
                continue
            t_sq = s * s - N
            if t_sq < 0:
                continue
            t = isqrt(t_sq)
            if t * t == t_sq and t < s:
                p = s - t
                q = s + t
                if 1 < p < N and 1 < q < N:
                    return (min(p, q), max(p, q))

    return None


def sigma_cryptanalysis(N: int, sigma_N: int | None = None) -> tuple[int, int] | None:
    """Factor N using the divisor sum σ(N).

    For N = pq with p, q distinct primes:
        σ(N) = (1 + p)(1 + q) = 1 + p + q + pq = 1 + p + q + N

    So p + q = σ(N) - N - 1, and pq = N.
    This gives p and q as roots of x2 - (p+q)x + N = 0.

    For N = p2 (perfect square):
        σ(N) = 1 + p + p2, so p = σ(N) - N - 1.

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

    # p and q are roots of x2 - (p+q)x + N = 0
    # discriminant = (p+q)2 - 4N = (p-q)2
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