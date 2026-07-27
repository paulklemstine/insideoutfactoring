"""Top-level factoring API.

Provides the main entry point for the Inside-Out factoring algorithm.
Tries multiple strategies in order of expected speed:

1. Perfect square detection (O(1))
2. CF convergent divisibility pre-check (O(log N))
3. Quick trial division for small factors
4. Brahmagupta-Fibonacci two-square method (for N ≡ 1 mod 4)
5. Fermat difference-of-squares (for close factors)
6. Fibonacci GCD factorization
7. Resonance Cascade (CF-convergent resonance + Möbius descent + squaring conductance)
8. Inside-Out (CF-steered best-first search + BFS)
9. Wavefront search
10. Full trial division
"""
from __future__ import annotations

from math import isqrt

from .inside_out import inside_out_factor
from .wavefront import search_wavefront
from .cf_guide import cf_factor_check
from .brahmagupta import brahmagupta_fibonacci_factor, fermat_difference_of_squares
from .fibonacci_factor import fibonacci_gcd_factor, pisano_factor
from .resonance_cascade import resonance_cascade_factor
from .multi_scale_mobius import multi_scale_mobius_factor
from .character_sum import character_sum_factor
from .rank_apparition import rank_apparition_factor, rank_apparition_multi
from .cf_guided_descent import cf_guided_descent_factor


def factor(N: int) -> tuple[int, int] | None:
    """Factor an integer N into two factors p and q where N = p*q.

    Uses the Inside-Out factoring algorithm with wavefront search
    and trial division as fallback.

    Returns (p, q) with p < q if N is composite, None if N is prime.
    """
    result = factor_with_method(N)
    if result is None:
        return None
    return result[0]


def factor_with_method(N: int, time_budget_s: float = 30.0) -> tuple[tuple[int, int], str] | None:
    """Factor N and return the factors along with the method used.

    Method name is one of: "perfect_square", "cf_precheck",
    "brahmagupta", "fermat", "fibonacci", "resonance_cascade",
    "inside_out", "wavefront", "trial_division".
    """
    import time as _time
    _chain_start = _time.perf_counter()

    if N < 4:
        return None

    # Handle even N
    if N % 2 == 0:
        if N == 2:
            return None
        return ((2, N // 2), "trial_division")

    # Perfect square detection: if N = p^2, then p is a factor
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return ((sqrt_N, sqrt_N), "perfect_square")

    # CF convergent divisibility pre-check
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        p, q = cf_result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "cf_precheck")

    # Quick trial division for small factors (safety net)
    for p in range(3, min(isqrt(N) + 1, 1000), 2):
        if N % p == 0:
            return ((p, N // p), "trial_division")

    # === FAST METHODS FIRST (O(1) for well-separated factors) ===
    # Resonance Cascade: Möbius descent is O(1) when q/p is a perfect square
    rc_result = resonance_cascade_factor(N)
    if rc_result is not None:
        p, q = rc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "resonance_cascade")

    # Multi-Scale Möbius: multi-start Berggren tree descent
    msm_result = multi_scale_mobius_factor(N, num_scales=10, depth=500)
    if msm_result is not None:
        p, q = msm_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "multi_scale_mobius")

    # CF-Guided Descent: CF convergents + Möbius descent
    cfd_result = cf_guided_descent_factor(N, max_convergents=500, depth=2000)
    if cfd_result is not None:
        p, q = cfd_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "cf_guided_descent")

    # Rank of Apparition: Fibonacci/Lucas/Mersenne smooth rank probing
    ra_result = rank_apparition_multi(N, bounds=[100, 500, 1000, 5000])
    if ra_result is not None:
        p, q = ra_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "rank_apparition")

    # Character Sum: Jacobi symbol probing
    cs_result = character_sum_factor(N, max_trials=5000)
    if cs_result is not None:
        p, q = cs_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "character_sum")

    # Check time budget
    if _time.perf_counter() - _chain_start > time_budget_s:
        return None

    # Strategy: Brahmagupta-Fibonacci two-square method
    # Effective for N ≡ 1 mod 4 (products of primes ≡ 1 mod 4)
    bf_result = brahmagupta_fibonacci_factor(N)
    if bf_result is not None:
        p, q = bf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "brahmagupta")

    # Strategy: Fermat difference-of-squares
    # Effective for close factors (p ~ q)
    fermat_result = fermat_difference_of_squares(N)
    if fermat_result is not None:
        p, q = fermat_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "fermat")

    # Strategy: Fibonacci GCD factorization
    # Effective for N where some prime factor p has a small entry point α(p)
    fib_result = fibonacci_gcd_factor(N, bound=5000)
    if fib_result is not None:
        p, q = fib_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "fibonacci")

    # Strategy: Resonance Cascade Factoring
    # Combines CF-convergent resonance, Möbius descent, and squaring conductance
    rc_result = resonance_cascade_factor(N)
    if rc_result is not None:
        p, q = rc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "resonance_cascade")

    # Strategy: Inside-Out (CF-steered best-first search + BFS)
    result = inside_out_factor(N, max_iterations=50000)
    if result is not None:
        p, q = result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "inside_out")

    # Strategy: Wavefront search
    result = search_wavefront(N, max_radius=500)
    if result is not None:
        p, q = result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "wavefront")

    # Fallback: Full trial division (with time check)
    import signal
    limit = isqrt(N) + 1
    # Only trial divide up to a reasonable limit for the time budget
    trial_limit = min(limit, 10_000_000)  # Cap at 10M iterations
    for p in range(3, trial_limit, 2):
        if N % p == 0:
            return ((p, N // p), "trial_division")

    return None