"""Self-Guided Smooth Search — Novel Method Using Early Successes to Guide Search.

A novel approach that uses early smooth finds to guide the search for more smooth relations.

Key insight: If a^B! mod N has small factors, then a^(B!*k) mod N for various k
might also have small factors. The early smooth finds create a "guidance pattern"
that biases the search toward similar values.

This is different from standard methods because:
1. Standard smooth-search is random (vary the base a)
2. Self-guided search uses discovered smooth values to construct new candidates

Algorithm:
1. Run standard smooth search (e.g., cyclotomic cascade)
2. Record the values that worked: which bases, which cyclotomic orders, which powers
3. Construct new candidates based on combinations of successful patterns
4. Use these to find more smooth relations faster

This is a heuristic improvement — no theoretical complexity change.
"""
from __future__ import annotations

from math import gcd, isqrt
from collections import defaultdict


def _small_primes(bound: int) -> list[int]:
    """Generate primes up to bound."""
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(bound) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [i for i in range(2, bound + 1) if sieve[i]]


def self_guided_factor(N: int, bound: int = 50000,
                        base_points: int = 10) -> tuple[int, int] | None:
    """Factor N using self-guided smooth search.

    Uses early smooth finds to guide the search for more relations.

    Returns (p, q) with p < q and p*q = N, or None.
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

    # Track successful patterns: (base, power, order, smoothness_score)
    successful_patterns = []

    # Phase 1: Initial smooth search (like cyclotomic cascade)
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:base_points]:
        if a >= N:
            continue

        power = a
        for prime in primes:
            pk = prime
            while pk * prime <= bound:
                pk *= prime
            power = pow(power, pk, N)

            # Check various cyclotomic orders
            for order_name, check_fn in [
                ('m1', lambda v: gcd(abs(v - 1), N)),
                ('m2', lambda v: gcd(abs(v + 1), N)),
            ]:
                val = check_fn(power)
                if 1 < val < N:
                    successful_patterns.append((a, power, order_name))
                    p_factor = min(val, N // val)
                    if 1 < p_factor < N:
                        q_factor = N // p_factor
                        if p_factor * q_factor == N:
                            return (p_factor, q_factor)

            # Check higher orders
            power2 = pow(power, 2, N)
            for check_val, order_name in [
                (power2 + power + 1, 'm3'),
                (power2 + 1, 'm4'),
                (power2 - power + 1, 'm6'),
            ]:
                g = gcd(abs(check_val), N)
                if 1 < g < N:
                    successful_patterns.append((a, power, order_name))
                    p_factor = min(g, N // g)
                    if 1 < p_factor < N:
                        q_factor = N // p_factor
                        if p_factor * q_factor == N:
                            return (p_factor, q_factor)

    if not successful_patterns:
        return None

    # Phase 2: Guided search using successful patterns
    # Construct new candidates based on combinations of successful patterns
    for iteration in range(3):  # Multiple guided iterations
        new_candidates = []

        # Pattern 1: Product of successful bases
        for i, (a1, p1, o1) in enumerate(successful_patterns):
            for a2, p2, o2 in successful_patterns[i+1:]:
                # Combine two successful patterns
                combined_base = (a1 * a2) % N
                combined_power = pow(combined_base, 1, N)  # Reset power

                # Try powering with smooth bound
                for prime in primes[:len(primes)//2]:  # Only half the primes
                    pk = prime
                    while pk * prime <= bound // 10:
                        pk *= prime
                    combined_power = pow(combined_power, pk, N)

                # Check this guided candidate
                for check_val, _ in [
                    (combined_power - 1, 'guided_m1'),
                    (combined_power + 1, 'guided_m2'),
                    (pow(combined_power, 2, N) + 1, 'guided_m4'),
                ]:
                    g = gcd(abs(check_val), N)
                    if 1 < g < N:
                        p_factor = min(g, N // g)
                        if 1 < p_factor < N:
                            q_factor = N // p_factor
                            if p_factor * q_factor == N:
                                return (p_factor, q_factor)

        # Pattern 2: Power of successful base
        for a, power, order_name in successful_patterns:
            for multiplier in [2, 3, 5]:
                guided_base = pow(a, multiplier, N)
                guided_power = guided_base

                for prime in primes[:len(primes)//2]:
                    pk = prime
                    while pk * prime <= bound // 10:
                        pk *= prime
                    guided_power = pow(guided_power, pk, N)

                g = gcd(abs(guided_power - 1), N)
                if 1 < g < N:
                    p_factor = min(g, N // g)
                    if 1 < p_factor < N:
                        q_factor = N // p_factor
                        if p_factor * q_factor == N:
                            return (p_factor, q_factor)

                g = gcd(abs(guided_power + 1), N)
                if 1 < g < N:
                    p_factor = min(g, N // g)
                    if 1 < p_factor < N:
                        q_factor = N // p_factor
                        if p_factor * q_factor == N:
                            return (p_factor, q_factor)

        # Pattern 3: CRT-guided: construct x such that x == successful_base (mod p)
        # We don't know p, but we can use the pattern
        for a, power, _ in successful_patterns[:3]:  # Top 3 patterns
            # Try x = a^k for various k
            for k in [2, 3, 5, 7, 11]:
                guided_base = pow(a, k, N)
                guided_power = guided_base

                for prime in primes[:len(primes)//3]:
                    pk = prime
                    while pk * prime <= bound // 20:
                        pk *= prime
                    guided_power = pow(guided_power, pk, N)

                g = gcd(abs(guided_power - 1), N)
                if 1 < g < N:
                    p_factor = min(g, N // g)
                    if 1 < p_factor < N:
                        q_factor = N // p_factor
                        if p_factor * q_factor == N:
                            return (p_factor, q_factor)

    return None