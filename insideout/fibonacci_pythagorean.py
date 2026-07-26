"""Fibonacci–Pythagorean Hybrid Factoring.

A verified synthesis combining Fibonacci/Lucas rank probes with Pythagorean
coordinate batching. Based on the Lean-verified identity:

  A_n = 2·F_{n+1}·F_{n+2}       (even leg)
  B_n = F_n · F_{n+3}            (odd leg)
  C_n = F_{n+1}2 + F_{n+2}2      (hypotenuse)

with A_n2 + B_n2 = C_n2 and C_n = F_{2n+3}.

The mechanism:
1. Compute Q^k mod N (companion matrix) to get F_k mod N in O(log k) steps
2. At each smooth-rank stage, derive 5 GCD candidates:
   F_M, L_M, A_M, B_M, C_M
3. Batch all 5 into one GCD check: gcd(F_M · L_M · A_M · B_M · C_M mod N, N)
4. If gcd = N, split and test individually
5. Stage 2: after smooth core M, test F_{M·ℓ} for small primes ℓ

This is a smooth-rank method (like Pollard p-1) using Fibonacci rank of
apparition instead of multiplicative order. It is fast when a prime factor p
has a smooth rank of apparition Z(p). It is NOT a generic polynomial-time
factoring algorithm.

Per the Algebraic Light assessment: the Pythagorean coordinates add cheap
correlated probes (batching two shifted Fibonacci tests), but calling this
"topological acceleration" would overstate the result. It is a principled
candidate batch that may improve constants.
"""
from __future__ import annotations

from math import gcd, isqrt


def _fibonacci_pair(k: int, N: int) -> tuple[int, int]:
    """Compute (F_k mod N, F_{k+1} mod N) using fast doubling.

    Uses the identities:
      F_{2k} = F_k · (2·F_{k+1} - F_k)
      F_{2k+1} = F_k2 + F_{k+1}2

    Returns (F_k mod N, F_{k+1} mod N).
    """
    if k == 0:
        return (0 % N, 1 % N)
    if k == 1:
        return (1 % N, 1 % N)

    # Fast doubling
    a, b = 0, 1  # F_0, F_1
    bits = k.bit_length()

    for i in range(bits - 1, -1, -1):
        # Doubling formulas
        # F_{2j} = F_j · (2·F_{j+1} - F_j)
        # F_{2j+1} = F_j2 + F_{j+1}2
        c = a * ((2 * b - a) % N) % N  # F_{2j}
        d = (a * a + b * b) % N        # F_{2j+1}

        if (k >> i) & 1:
            a, b = d, (c + d) % N  # F_{2j+1}, F_{2j+2}
        else:
            a, b = c, d              # F_{2j}, F_{2j+1}

    return (a % N, b % N)


def _fibonacci_smooth_rank(N: int, bound: int = 10000,
                           stage2_bound: int = 1000) -> tuple[int, int] | None:
    """Factor N using Fibonacci smooth-rank method with Pythagorean batching.

    Stage 1: Compute Q^M mod N where M = lcm(1, 2, ..., B) = product of
    prime powers up to B. At each prime power, compute F_M and derive
    Pythagorean coordinates for a batched GCD check.

    Stage 2: After smooth core M, test F_{M·ℓ} for small primes ℓ.

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square
    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Stage 1: Smooth-rank computation
    # Build M = product of prime powers up to bound
    # At each step, compute F_M mod N and check GCDs
    primes = _small_primes(bound)

    # Compute Q^M mod N incrementally
    # We track (F_M, F_{M+1}) mod N
    # After each prime power, we have a new M and can check
    M = 1
    fk, fk1 = 1 % N, 1 % N  # F_1 = 1, F_2 = 1

    for p in primes:
        # Largest power of p <= bound
        pk = p
        while pk * p <= bound:
            pk *= p

        # Compute F_{M*pk} from F_M using matrix powering
        # Q^{M*pk} = (Q^M)^{pk}
        # We need to raise the matrix [[F_{M+1}, F_M], [F_M, F_{M-1}]] to pk power
        # But it's easier to just compute F_{M*pk} directly

        old_M = M
        M *= pk

        # Compute (F_M, F_{M+1}) using fast doubling
        fk_new, fk1_new = _fibonacci_pair(M, N)

        # Batch GCD: F_M, L_M, A_M, B_M, C_M
        L_M = (2 * fk1_new - fk_new) % N  # L_M = 2*F_{M+1} - F_M

        # For Pythagorean coordinates, we need F_M, F_{M+1}, F_{M+2}, F_{M+3}
        # F_{M+2} = F_{M+1} + F_M
        fm2 = (fk1_new + fk_new) % N
        # F_{M+3} = F_{M+2} + F_{M+1}
        fm3 = (fm2 + fk1_new) % N

        # A_M = 2·F_{M+1}·F_{M+2}
        A_M = (2 * fk1_new * fm2) % N
        # B_M = F_M · F_{M+3}
        B_M = (fk_new * fm3) % N
        # C_M = F_{M+1}2 + F_{M+2}2
        C_M = (fk1_new * fk1_new + fm2 * fm2) % N

        # Batch: product of all candidates mod N
        batch = (fk_new * L_M * A_M * B_M * C_M) % N

        g = gcd(batch, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        if g == N:
            # Split: test each individually
            for val in [fk_new, L_M, A_M, B_M, C_M]:
                g = gcd(val, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

        fk, fk1 = fk_new, fk1_new

    # Stage 2: Test F_{M·ℓ} for small primes ℓ
    # This catches factors where Z(p) = M·ℓ for small prime ℓ
    for ell in _small_primes(stage2_bound):
        if ell in set(primes):
            continue
        # Compute F_{M*ell}
        M2 = M * ell
        fM2, fM2_1 = _fibonacci_pair(M2, N)

        # Pythagorean coordinates
        fM2_2 = (fM2_1 + fM2) % N
        fM2_3 = (fM2_2 + fM2_1) % N
        L_M2 = (2 * fM2_1 - fM2) % N
        A_M2 = (2 * fM2_1 * fM2_2) % N
        B_M2 = (fM2 * fM2_3) % N
        C_M2 = (fM2_1 * fM2_1 + fM2_2 * fM2_2) % N

        batch = (fM2 * L_M2 * A_M2 * B_M2 * C_M2) % N
        g = gcd(batch, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        if g == N:
            for val in [fM2, L_M2, A_M2, B_M2, C_M2]:
                g = gcd(val, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def _small_primes(bound: int) -> list[int]:
    """Generate primes up to bound using a simple sieve."""
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(bound) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [i for i in range(2, bound + 1) if sieve[i]]


def fibonacci_pythagorean_factor(N: int, bound: int = 10000,
                                  stage2_bound: int = 1000) -> tuple[int, int] | None:
    """Factor N using Fibonacci–Pythagorean hybrid method."""
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2) if N > 2 else None
    # Skip for large N — smooth rank computation too slow
    if N.bit_length() > 256:
        return None
    return _fibonacci_smooth_rank(N, bound=bound, stage2_bound=stage2_bound)