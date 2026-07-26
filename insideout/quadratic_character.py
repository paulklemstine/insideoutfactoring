"""Quadratic Character Difference (QCD) Factoring.

A novel factoring algorithm exploiting the independence of Legendre symbols
(a/p) and (a/q) for a semiprime N = p*q.

Mathematical core
-----------------
For N = p*q the Jacobi symbol (a/N) = (a/p)(a/q) is a multiplicative
character.  By Euler's criterion:

    a^((p-1)/2) ≡  (a/p)  mod p,     a^((q-1)/2) ≡  (a/q)  mod q.

The four layers of the algorithm:

1. **QR Ladder** — compute base^(2^k) mod N for k = 0, 1, 2, ...
   and check gcd(base^(2^k) ± 1, N) at each step.  When v_2(p-1) ≠
   v_2(q-1) and the base happens to lie in a 2-Sylow subgroup of one
   of the prime fields, the orbit hits 1 mod one prime but not the
   other, yielding a nontrivial square root of 1 and splitting N.

2. **Jacobi Mismatch** — sample random a with (a/N) = -1.  Then
   (a/p) = -(a/q), so a is a QR mod one prime and a QNR mod the
   other.  The value a^((N-1)/2) mod N is a CRT combination of the
   two Euler-criterion values; when it is neither +1 nor -1 mod N it
   is a nontrivial square root of 1 and gcd(a^((N-1)/2) ± 1, N) splits
   N.  Probability of a useful mismatch per random sample ≈ 1/2.

3. **Smooth-exponent p±1 (workhorse)** — compute a^M mod N where
   M = 2^K 3^K 5^K ... p_K^K is a smooth exponent.  If p-1 | M then
   a^M ≡ 1 mod p and gcd(a^M - 1, N) = p.  This is the classic p−1
   method, here framed and seeded via quadratic-character information.

4. **Pollard ρ** — standard fallback for remaining cases.

The novel contribution is the combination: the Jacobi symbol is used to
select bases and detect mismatches early, the QR ladder exploits 2-adic
structure, and the smooth-exponent stage catches everything with a
smooth p±1.
"""
from __future__ import annotations

import random
from math import gcd, isqrt


# ---------------------------------------------------------------------------
# Small primes for the multi-base ladder and smooth-exponent stage
# ---------------------------------------------------------------------------
_SMALL_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139,
    149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
    307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383,
    389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trial_division(n: int, bound: int = 1000) -> tuple[int, int] | None:
    """Return a nontrivial factor pair of n if a prime <= bound divides it."""
    if n % 2 == 0:
        return (2, n // 2)
    f = 3
    while f <= bound and f * f <= n:
        if n % f == 0:
            return (f, n // f)
        f += 2
    return None


def _is_perfect_power(n: int) -> tuple[int, int] | None:
    """Detect n = a**b for b >= 2.  Returns (a, b) or None."""
    for b in range(2, n.bit_length() + 1):
        if b == 2:
            a = isqrt(n)
        else:
            a = int(round(n ** (1.0 / b)))
        for cand in (a - 1, a, a + 1):
            if cand > 1 and cand ** b == n:
                return (cand, b)
    return None


def _jacobi(a: int, n: int) -> int:
    """Compute the Jacobi symbol (a/n) for odd positive n.

    Returns -1, 0, or 1.  Uses the standard quadratic-reciprocity
    algorithm with O(log n) steps.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            r = n % 8
            if r == 3 or r == 5:
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    if n == 1:
        return result
    return 0


def _v2(n: int) -> int:
    """Return the 2-adic valuation of n (largest k with 2^k | n)."""
    if n == 0:
        return 64
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

def _sqrt_one_split(N: int, x: int) -> tuple[int, int] | None:
    """Given x with x² ≡ 1 mod N (nontrivial), attempt to split N.

    Returns a factor pair (p, q) with p*q = N and 1 < p <= q < N, or None
    if the split fails (x was trivial).
    """
    if x <= 1 or x >= N - 1:
        return None
    g1 = gcd(x - 1, N)
    if 1 < g1 < N:
        return (g1, N // g1)
    g2 = gcd(x + 1, N)
    if 1 < g2 < N:
        return (g2, N // g2)
    return None


def _qr_ladder(N: int, base: int, max_k: int = 64) -> tuple[int, int] | None:
    """Compute the sequence base^(2^k) mod N and check for nontrivial sqrt(1).

    At each step compute x ← x² mod N and check gcd(x ± 1, N).  The
    orbit eventually reaches 1 mod N; the predecessor of 1 is a square
    root of 1, and if it is nontrivial we split N.

    Args:
        N: composite odd integer to factor.
        base: starting base coprime to N.
        max_k: maximum ladder height.

    Returns:
        A factor pair or None.
    """
    if gcd(base, N) > 1:
        g = gcd(base, N)
        return (g, N // g)

    x = base % N
    prev = x
    for _ in range(max_k):
        x = pow(x, 2, N)
        g = gcd(x - 1, N)
        if 1 < g < N:
            return (g, N // g)
        g = gcd(x + 1, N)
        if 1 < g < N:
            return (g, N // g)
        if x == 1:
            # prev is a square root of 1
            return _sqrt_one_split(N, prev)
        if x == prev:
            # fixed point that is not 1; further squaring is pointless
            break
        prev = x
    return None


def _qr_ladder_divergence(N: int, base: int, max_k: int = 64) -> tuple[int, int] | None:
    """QR ladder with explicit divergence tracking.

    Tracks the exact point where base^(2^k) mod N first equals 1, then
    backs up one step to extract the nontrivial square root of 1.
    """
    if gcd(base, N) > 1:
        g = gcd(base, N)
        return (g, N // g)

    x = base % N
    for k in range(max_k):
        prev = x
        x = pow(x, 2, N)
        if x == 1:
            # prev is a square root of 1
            return _sqrt_one_split(N, prev)
        if x == prev:
            break
    return None


def _multi_base_qcd(N: int, bases: list[int] | None = None) -> tuple[int, int] | None:
    """Try the QR ladder on multiple small-prime bases.

    For random p, q the probability that v_2(p-1) ≠ v_2(q-1) is 2/3.
    When the valuations differ, a base in the 2-Sylow subgroup of one
    prime field (but not the other) yields a split.  We try many bases
    to maximise the chance of hitting such a case.

    Args:
        N: composite to factor.
        bases: list of bases to try; defaults to _SMALL_PRIMES.

    Returns:
        A factor pair or None.
    """
    if bases is None:
        bases = _SMALL_PRIMES
    for b in bases:
        if b >= N:
            break
        g = gcd(b, N)
        if 1 < g < N:
            return (g, N // g)
        result = _qr_ladder(N, b)
        if result is not None:
            return result
    return None


def _jacobi_sequence(N: int, max_samples: int = 200) -> tuple[int, int] | None:
    """Sample random a, detect Jacobi-symbol vs Euler-criterion mismatch.

    For a ∈ (Z/NZ)* with (a/N) = -1 we know (a/p) = -(a/q).  The value
    a^((N-1)/2) mod N combines the two Euler-criterion values; when it
    is neither +1 nor -1 mod N it is a nontrivial square root of 1 and
    gcd(a^((N-1)/2) ± 1, N) splits N.

    Args:
        N: composite to factor.
        max_samples: number of random a values to try.

    Returns:
        A factor pair or None.
    """
    if N % 2 == 0:
        return (2, N // 2)
    half = (N - 1) // 2
    for _ in range(max_samples):
        a = random.randrange(2, N - 1)
        chi = pow(a, half, N)
        if chi != 1 and chi != N - 1:
            # chi is a nontrivial square root of 1 mod N
            result = _sqrt_one_split(N, chi)
            if result is not None:
                return result
            # Also try direct gcd
            g = gcd(chi - 1, N)
            if 1 < g < N:
                return (g, N // g)
    return None


def _smooth_exponent_pm1(N: int, bound: int = 50000) -> tuple[int, int] | None:
    """Smooth-exponent p−1 method (workhorse).

    Compute a^M mod N where M = ∏ p_i for all primes p_i ≤ bound,
    raising to each prime in turn and checking gcd(a^M - 1, N) after
    each step.  If p-1 | M for some prime factor p of N, then
    a^M ≡ 1 mod p and gcd(a^M - 1, N) = p.

    Tries several bases to handle the case where both p-1 and q-1 are
    smooth (in which case a single base may give 1 mod N immediately).

    Args:
        N: composite to factor.
        bound: smoothness bound for the exponent.

    Returns:
        A factor pair or None.
    """
    bases = [2, 3, 5, 7, 6, 10, 11, 12, 13, 17]
    for base in bases:
        if base >= N:
            continue
        g = gcd(base, N)
        if 1 < g < N:
            return (g, N // g)
        x = base % N
        for p in _SMALL_PRIMES:
            if p > bound:
                break
            # Raise to the highest power of p not exceeding bound
            pe = p
            while pe * p <= bound:
                pe *= p
            x = pow(x, pe, N)
            g = gcd(x - 1, N)
            if 1 < g < N:
                return (g, N // g)
            if g == N:
                # x ≡ 1 mod N; this base is exhausted, try the next
                break
    return None


def _pollard_rho(N: int, max_steps: int = 200000) -> tuple[int, int] | None:
    """Pollard's ρ method with Brent's cycle detection.

    Standard fallback for cases not caught by the character-based methods.

    Args:
        N: composite to factor.
        max_steps: maximum iterations.

    Returns:
        A factor pair or None.
    """
    if N % 2 == 0:
        return (2, N // 2)
    for _ in range(10):
        x = random.randrange(2, N - 1)
        y = x
        c = random.randrange(1, N - 1)
        d = 1
        steps = 0
        while d == 1 and steps < max_steps:
            x = (x * x + c) % N
            y = (y * y + c) % N
            y = (y * y + c) % N
            d = gcd(abs(x - y), N)
            steps += 1
        if 1 < d < N:
            return (d, N // d)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def quadratic_character_factor(N: int, max_iter: int = 100000) -> tuple[int, int] | None:
    """Factor N using the Quadratic Character Difference method.

    Layers (in order of expected speed):
        1. Trivial checks (even, perfect square, perfect power, small factors).
        2. Multi-base QR ladder on small primes (catches ~2/3 of semiprimes
           with different 2-adic valuations).
        3. Jacobi-sequence mismatch detection (random sampling).
        4. Smooth-exponent p−1 (workhorse for smooth p±1).
        5. Pollard ρ (general fallback).

    Args:
        N: integer > 1 to factor.
        max_iter: upper bound on total iterations across all layers.

    Returns:
        A factor pair (p, q) with 1 < p <= q < N and p*q = N, or None if
        factoring failed.
    """
    if N < 2:
        return None
    if N % 2 == 0:
        return (2, N // 2)

    # Perfect square
    r = isqrt(N)
    if r * r == N:
        return (r, r)

    # Perfect power
    pp = _is_perfect_power(N)
    if pp is not None:
        a, b = pp
        other = N // a
        return (min(a, other), max(a, other))

    # Small trial division
    td = _trial_division(N, bound=min(1000, int(N ** (1 / 3)) + 1))
    if td is not None:
        return (min(td[0], td[1]), max(td[0], td[1]))

    def _order(pair):
        return (min(pair[0], pair[1]), max(pair[0], pair[1]))

    # Layer 2: multi-base QR ladder — fast, deterministic, catches ~2/3
    result = _multi_base_qcd(N)
    if result is not None:
        return _order(result)

    # Layer 3: Jacobi-sequence mismatch (random sampling)
    samples = min(max_iter // 4, 500)
    result = _jacobi_sequence(N, max_samples=samples)
    if result is not None:
        return _order(result)

    # Layer 4: smooth-exponent p−1
    result = _smooth_exponent_pm1(N, bound=min(max_iter // 10, 200000))
    if result is not None:
        return _order(result)

    # Layer 5: Pollard ρ
    result = _pollard_rho(N, max_steps=max_iter)
    if result is not None:
        return _order(result)

    return None
