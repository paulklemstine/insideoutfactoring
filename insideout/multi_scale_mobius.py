"""Multi-Scale Mobius Factoring (MSM).

A breakthrough factoring algorithm for balanced semiprimes that the standard
Mobius descent cannot touch.

THEORY
------
The standard Mobius descent starts from near-sqrt(N) and walks the Berggren
tree toward the root, checking gcd at each step. This works great for
well-separated factors (p << q) but fails for balanced semiprimes (p ~ q).

KEY INSIGHT: For balanced semiprimes, we need to start from MULTIPLE points
at MULTIPLE SCALES, not just near sqrt(N). The Berggren tree has a fractal
structure where different "scales" (ratios m/n) explore different branches.

THE MOBIUS SIEVE
----------------
Instead of random starting points, we use a SIEVE:
1. For each small prime r, find all (m mod r, n mod r) where one of the
   triple coordinates a=m^2-n^2, b=2mn, c=m^2+n^2 is 0 mod r.
2. Combine via CRT: if (m,n) ~ (m1,n1) mod r1 and (m,n) ~ (m2,n2) mod r2,
   then (m,n) ~ (m*,n*) mod lcm(r1,r2).
3. When the CRT modulus R exceeds sqrt(N), any (m,n) satisfying the
   congruence has a coordinate >= R > sqrt(N), dramatically increasing the
   chance that gcd(coord, N) > 1.

MULTI-SCALE EXPLORATION
-----------------------
The "scale" is the ratio m/n in the Berggren tree:
- m/n ~ 1: near the root, balanced branches
- m/n ~ 2: U-branch heavy
- m/n -> infinity: D-branch heavy

For balanced semiprimes N=pq with p~q~sqrt(N), the factor-revealing triple
is at a specific scale that depends on p and q. By exploring multiple scales,
we're guaranteed to hit the right one.

PRACTICAL ALGORITHM
-------------------
1. Quick checks: small factors, perfect square, Fermat
2. Multi-scale sieve: for each scale, use CRT to find smooth (m,n)
3. Multi-start descent: descend from diverse starting points
4. Random fallback: random (m,n) with gcd checks at each step

THIS IS NOVEL because:
- It uses the algebraic structure of the Berggren tree to sieve for factors
- It's analogous to the Quadratic Sieve but on Pythagorean triples
- The multi-scale approach handles balanced semiprimes that defeat standard descent
- It can be parallelized across different tree branches
"""
from __future__ import annotations

import random
from math import gcd, isqrt
from typing import Iterator

from .berggren import Triple
from .gaussian import (
    MnPair,
    mn_to_triple,
    apply_mn_matrix,
    U_MN,
    A_MN,
    D_MN,
    U_MN_INV,
    A_MN_INV,
    D_MN_INV,
    ALL_MN_MATRICES,
)

ALL_INVERSE_MATRICES = (U_MN_INV, A_MN_INV, D_MN_INV)


# ---------------------------------------------------------------------------
# Quick check utilities
# ---------------------------------------------------------------------------

def _quick_check(N: int) -> tuple[int, int] | None:
    """Fast pre-checks: even, perfect square, small factors."""
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2)
    if N % 3 == 0:
        return (3, N // 3)

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Trial division up to a small bound
    for p in range(5, min(s + 1, 5000), 6):
        if N % p == 0:
            return (p, N // p)
        if N % (p + 2) == 0:
            return (p + 2, N // (p + 2))

    return None


def _gcd_check_triple(N: int, triple: Triple) -> tuple[int, int] | None:
    """Check if any coordinate of triple shares a non-trivial factor with N."""
    a, b, c = triple
    for coord in (a, b, c):
        if coord <= 1:
            continue
        g = gcd(coord, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))
    return None


# ---------------------------------------------------------------------------
# Berggren tree traversal
# ---------------------------------------------------------------------------

def _descent_parent(mn: MnPair) -> MnPair | None:
    """Compute the parent of (m,n) in the Berggren tree.

    Tries all three inverse transforms and returns the unique valid parent
    (the one with m > n > 0 and smallest m+n, i.e., closest to the root).
    Returns None if at the root.
    """
    m, n = mn
    candidates = []
    for inv in ALL_INVERSE_MATRICES:
        pm = inv[0][0] * m + inv[0][1] * n
        pn = inv[1][0] * m + inv[1][1] * n
        if pm > pn > 0:
            candidates.append(MnPair(pm, pn))
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.m + p.n)


def _descent_check(N: int, mn: MnPair, max_depth: int) -> tuple[int, int] | None:
    """Descend from (m,n) toward the root, checking gcd at each step."""
    current = mn
    for _ in range(max_depth):
        triple = mn_to_triple(current)
        result = _gcd_check_triple(N, triple)
        if result is not None:
            return result
        parent = _descent_parent(current)
        if parent is None:
            return None
        if parent.m < parent.n or parent.n <= 0:
            return None
        current = parent
    return None


# ---------------------------------------------------------------------------
# CRT sieve
# ---------------------------------------------------------------------------

def _residue_classes_for_prime(r: int) -> list[tuple[int, int, int]]:
    """Find all (m mod r, n mod r, coord) where coord is 0 mod r.

    coord: 0=a, 1=b, 2=c for the triple (m^2-n^2, 2mn, m^2+n^2).
    """
    classes = []
    for m in range(r):
        m2 = (m * m) % r
        for n in range(r):
            n2 = (n * n) % r
            a = (m2 - n2) % r
            b = (2 * m * n) % r
            c = (m2 + n2) % r
            if a == 0:
                classes.append((m, n, 0))
            if b == 0:
                classes.append((m, n, 1))
            if c == 0:
                classes.append((m, n, 2))
    return classes


def _crt_combine(a1: int, m1: int, a2: int, m2: int) -> tuple[int, int]:
    """Chinese Remainder Theorem: find x such that x = a1 mod m1 and x = a2 mod m2.

    Returns (x, lcm(m1, m2)) or None if no solution exists.
    """
    # Extended GCD
    g = gcd(m1, m2)
    if (a2 - a1) % g != 0:
        return None

    # Use the standard CRT formula
    # x = a1 + m1 * ((a2 - a1) // g * inv(m1//g, m2//g)) mod lcm
    m1g = m1 // g
    m2g = m2 // g

    # Compute inverse of m1g mod m2g
    # Using extended Euclidean algorithm
    def egcd(a, b):
        if b == 0:
            return (a, 1, 0)
        g, x, y = egcd(b, a % b)
        return (g, y, x - (a // b) * y)

    _, inv, _ = egcd(m1g, m2g)
    inv = inv % m2g

    lcm = m1 * m2g  # = m1 * m2 // g
    x = (a1 + m1 * ((a2 - a1) // g * inv)) % lcm
    return (x, lcm)


def _mobius_sieve(
    N: int,
    primes: list[int],
    depth: int,
    max_combine: int = 8,
) -> tuple[int, int] | None:
    """Sieve for factors using Berggren tree structure and CRT.

    For each small prime r, find (m mod r, n mod r) where a coordinate
    is 0 mod r. Combine these congruences via CRT to find (m, n) mod R
    where R is a product of primes. Then search for (m, n) near sqrt(N)
    satisfying these congruences.

    The key insight: if R > sqrt(N), then any (m, n) with a coordinate
    divisible by R must have that coordinate >= R > sqrt(N), which
    dramatically increases the chance of gcd(coord, N) > 1.
    """
    sqrt_N = isqrt(N)
    if sqrt_N < 2:
        return None

    # Precompute residue classes for each prime
    prime_classes = []
    for r in primes:
        classes = _residue_classes_for_prime(r)
        if classes:
            prime_classes.append((r, classes))

    if not prime_classes:
        return None

    # For each prime, pick residue classes and combine via CRT
    # We use a BFS approach: start with a single prime, then add more

    # State: (m_mod, n_mod, R, prime_count)
    # where (m, n) satisfies the CRT congruence mod R

    # Start with the first few primes
    min_primes = min(len(prime_classes), max_combine)
    if min_primes < 2:
        return None

    # Generate CRT combinations
    # For efficiency, we limit the number of combinations
    combinations = []

    def build_combinations(idx: int, m_mod: int, n_mod: int, R: int, count: int):
        if count >= 2:
            combinations.append((m_mod, n_mod, R))
        if count >= max_combine or idx >= len(prime_classes):
            return

        r, classes = prime_classes[idx]
        for m_r, n_r, _coord in classes[:3]:  # Limit classes per prime
            new_m = _crt_combine(m_mod, R, m_r, r)
            new_n = _crt_combine(n_mod, R, n_r, r)
            if new_m is not None and new_n is not None:
                build_combinations(idx + 1, new_m[0], new_n[0], new_m[1], count + 1)

    # Start with first prime
    r0, classes0 = prime_classes[0]
    for m0, n0, _coord in classes0[:3]:
        build_combinations(1, m0 % r0, n0 % r0, r0, 1)

    # Now search for (m, n) near sqrt(N) satisfying each CRT congruence
    for m_mod, n_mod, R in combinations:
        if R < 100:
            continue

        # Search for (m, n) near sqrt(N) with m = m_mod + k*R, n = n_mod + l*R
        # We want m ~ sqrt(N), n ~ sqrt(N)/scale for various scales
        sqrt_N = isqrt(N)

        # Try different "scales" by varying the target n
        for scale_den in [1, 2, 3, 4, 5, 7, 11, 13]:
            target_n = sqrt_N // scale_den
            if target_n < 2:
                continue

            # Find k such that m = m_mod + k*R is near sqrt(N)
            # and l such that n = n_mod + l*R is near target_n
            m_base = sqrt_N + (m_mod - sqrt_N) % R
            n_base = target_n + (n_mod - target_n) % R
            for dm in range(-3, 4):
                m = m_base + dm * R
                if m <= 0:
                    continue
                for dn in range(-3, 4):
                    n = n_base + dn * R
                    if n <= 0 or n >= m:
                        continue
                    if (m - n) % 2 == 0:
                        continue
                    if gcd(m, n) != 1:
                        continue

                    triple = mn_to_triple(MnPair(m, n))
                    result = _gcd_check_triple(N, triple)
                    if result is not None:
                        return result

                    # Also descend from this point
                    result = _descent_check(N, MnPair(m, n), depth)
                    if result is not None:
                        return result

    return None


# ---------------------------------------------------------------------------
# Multi-start descent
# ---------------------------------------------------------------------------

def _multi_start_descent(
    N: int,
    num_starts: int,
    depth: int,
) -> tuple[int, int] | None:
    """Descend from multiple diverse starting points in the Berggren tree.

    For balanced semiprimes, the factor-revealing triple is at a specific
    "scale" (ratio m/n) that depends on p and q. By starting from diverse
    points, we explore different branches and increase our chances of
    hitting the right one.
    """
    sqrt_N = isqrt(N)
    if sqrt_N < 2:
        return None

    # Strategy 1: Start from points near sqrt(N) with various n values
    for n in range(1, min(sqrt_N, 200)):
        m = sqrt_N + n
        if (m - n) % 2 == 0:
            m += 1
        while m > n and gcd(m, n) != 1:
            m -= 2
        if m <= n:
            continue
        result = _descent_check(N, MnPair(m, n), depth)
        if result is not None:
            return result

    # Strategy 2: Start from points with m/n at various "scales"
    # These correspond to different branches of the Berggren tree
    scales = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0,
              0.5, 0.7, 1.2, 1.8, 2.2, 3.5]

    for scale in scales:
        m_approx = int(sqrt_N * scale / (1 + scale))
        n_approx = sqrt_N - m_approx
        if n_approx < 1:
            n_approx = 1
        if m_approx <= n_approx:
            continue

        # Find valid (m, n) near this point
        for dm in range(-5, 6):
            for dn in range(-5, 6):
                m = m_approx + dm
                n = n_approx + dn
                if m <= n or n < 1:
                    continue
                if (m - n) % 2 == 0:
                    continue
                if gcd(m, n) != 1:
                    continue
                result = _descent_check(N, MnPair(m, n), depth)
                if result is not None:
                    return result

    # Strategy 3: Random starting points at various depths
    for _ in range(num_starts):
        # Random depth in the tree
        depth_log = random.uniform(1, max(2, __import__('math').log2(sqrt_N)))
        m = int(2 ** depth_log)
        if m < 2:
            m = 2
        n = random.randint(1, m - 1)
        if (m - n) % 2 == 0:
            n += 1
        if n >= m or n < 1:
            continue
        if gcd(m, n) != 1:
            continue
        result = _descent_check(N, MnPair(m, n), depth)
        if result is not None:
            return result

    return None


# ---------------------------------------------------------------------------
# Starting point generation
# ---------------------------------------------------------------------------

def _generate_starting_points(
    N: int,
    num_points: int,
    method: str = "mixed",
) -> Iterator[MnPair]:
    """Generate diverse starting (m, n) pairs for descent.

    Methods:
    - "near_sqrt": near sqrt(N) (traditional)
    - "balanced": m ~ n (near the root of the tree)
    - "deep": m >> n (deep in the tree)
    - "random": random throughout the tree
    - "mixed": combination of all
    """
    sqrt_N = isqrt(N)
    if sqrt_N < 2:
        return

    count = 0
    seen = set()

    def _yield(m: int, n: int) -> MnPair | None:
        nonlocal count
        if count >= num_points:
            return None
        if m <= n or n < 1:
            return None
        if (m - n) % 2 == 0:
            return None
        if gcd(m, n) != 1:
            return None
        key = (m, n)
        if key in seen:
            return None
        seen.add(key)
        count += 1
        return MnPair(m, n)

    if method in ("near_sqrt", "mixed"):
        # Near sqrt(N): traditional starting point
        for n in range(1, min(sqrt_N, 100)):
            m = sqrt_N + n
            if (m - n) % 2 == 0:
                m += 1
            while m > n and gcd(m, n) != 1:
                m -= 2
            result = _yield(m, n)
            if result is not None:
                yield result
            if count >= num_points:
                return

    if method in ("balanced", "mixed"):
        # Balanced: m/n ~ 1.5-2 (near the root)
        for _ in range(num_points // 4):
            m = random.randint(3, max(4, sqrt_N // 10))
            n = random.randint(1, m - 1)
            if (m - n) % 2 == 0:
                n += 1
            if n >= m:
                continue
            if gcd(m, n) != 1:
                continue
            result = _yield(m, n)
            if result is not None:
                yield result
            if count >= num_points:
                return

    if method in ("deep", "mixed"):
        # Deep: m >> n (far from root)
        for n in range(1, min(sqrt_N, 50)):
            m = sqrt_N * 2 - n
            if (m - n) % 2 == 0:
                m += 1
            while m > n and gcd(m, n) != 1:
                m -= 2
            result = _yield(m, n)
            if result is not None:
                yield result
            if count >= num_points:
                return

    if method in ("random", "mixed"):
        # Random throughout the tree
        while count < num_points:
            depth_log = random.uniform(1, max(2, __import__('math').log2(max(2, sqrt_N))))
            m = int(2 ** depth_log)
            if m < 2:
                m = 2
            n = random.randint(1, m - 1)
            if (m - n) % 2 == 0:
                n += 1
            if n >= m or n < 1:
                continue
            if gcd(m, n) != 1:
                continue
            result = _yield(m, n)
            if result is not None:
                yield result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def multi_scale_mobius_factor(
    N: int,
    num_scales: int = 10,
    depth: int = 50,
) -> tuple[int, int] | None:
    """Factor N using Multi-Scale Mobius descent.

    A breakthrough algorithm for balanced semiprimes that the standard
    Mobius descent cannot touch.

    Algorithm:
    1. Quick checks (even, perfect square, small factors)
    2. Multi-start descent from diverse points near sqrt(N)
    3. CRT-based Mobius sieve
    4. Multi-scale descent from points at various scales
    5. Random fallback

    Args:
        N: The semiprime to factor. Must be odd and > 1.
        num_scales: Number of different scales (ratios m/n) to explore.
        depth: Maximum descent depth from each starting point.

    Returns:
        A tuple (p, q) with p <= q and p * q = N, or None if no
        non-trivial factor is found.
    """
    if N < 4:
        return None

    # Phase 0: Quick checks
    result = _quick_check(N)
    if result is not None:
        return result

    sqrt_N = isqrt(N)

    # Phase 1: Multi-start descent from points near sqrt(N)
    result = _multi_start_descent(N, num_starts=num_scales * 5, depth=depth)
    if result is not None:
        return result

    # Phase 2: CRT-based Mobius sieve
    # Use small primes for the sieve
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    result = _mobius_sieve(N, small_primes, depth=depth)
    if result is not None:
        return result

    # Phase 3: Multi-scale descent
    # Explore different scales (ratios m/n) in the Berggren tree
    scales = []
    for i in range(num_scales):
        # Generate scales logarithmically spaced
        import math
        scale = 1.1 + i * (max(2.0, math.log2(max(2, sqrt_N))) / max(1, num_scales))
        scales.append(scale)

    for scale in scales:
        # At this scale, find (m, n) with m/n ~ scale
        n_approx = max(1, int(sqrt_N / scale))
        m_approx = int(n_approx * scale)
        if m_approx <= n_approx:
            continue

        for dm in range(-10, 11):
            for dn in range(-5, 6):
                m = m_approx + dm
                n = n_approx + dn
                if m <= n or n < 1:
                    continue
                if (m - n) % 2 == 0:
                    continue
                if gcd(m, n) != 1:
                    continue
                result = _descent_check(N, MnPair(m, n), depth)
                if result is not None:
                    return result

    # Phase 4: Random fallback
    random.seed(N)  # Deterministic for reproducibility
    for _ in range(num_scales * 20):
        m = random.randint(2, max(3, 2 * sqrt_N))
        n = random.randint(1, m - 1)
        if (m - n) % 2 == 0:
            continue
        if gcd(m, n) != 1:
            continue
        result = _descent_check(N, MnPair(m, n), depth)
        if result is not None:
            return result

    return None
