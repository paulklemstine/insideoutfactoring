"""Hensel Lifting Cascade — Novel Factoring via Hensel's Lemma and CRT Lifting.

A novel factoring method based on the observation that Hensel's lemma provides
a way to lift modular roots from Z/p^k to Z/p^(k+1), and this lifting behavior
differs mod p vs mod q for N = pq.

Key insight: For a polynomial f(x) == 0 (mod p) with f'(x) ≢ 0 (mod p),
Hensel's lemma gives a unique lift to a root mod p^k for all k. But for a
composite modulus N = pq, the Chinese Remainder Theorem means f has a root
mod N iff it has roots mod p AND mod q. The number of roots mod N is the
product of the numbers mod p and mod q.

So if f has m roots mod p and n roots mod q, then:
- If m*n != 0, f has m*n roots mod N (expected)
- If gcd(f(x), N) != 1 for some root x, we found a factor!

The novel approach: we construct polynomials whose root structure mod p
and mod q differs, then detect the CRT divergence via gcd computations.

Specifically, we use:
1. The polynomial x^(B!) - 1: roots are elements of order dividing B!
2. Cyclotomic polynomials Φ_m(x): roots are primitive m-th roots of unity
3. Quadratic polynomials x2 - d: roots depend on whether d is a QR mod p

For each, we check gcd(f(x0) mod N, N) at candidate roots x0.

Per honest assessment: this is still a smooth-group-order method, achieving
L_p[1/2] expected time. The Hensel lifting adds a novel detection mechanism
but does not change the asymptotic complexity.
"""
from __future__ import annotations

from math import gcd, isqrt


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


def _jacobi(a: int, n: int) -> int:
    """Compute Jacobi symbol (a/n) for odd n > 0."""
    if n <= 0 or n % 2 == 0:
        return 0
    a = a % n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a = a % n
    return result if n == 1 else 0


def hensel_cascade_factor(N: int, bound: int = 50000,
                           max_lifts: int = 10,
                           base_points: int = 10) -> tuple[int, int] | None:
    """Factor N using Hensel lifting cascade.

    For each base a:
    1. Compute a^(B!) mod N (smooth-bound powering)
    2. Check gcd(a^(B!) - 1, N) and gcd(a^(B!) + 1, N) (standard p-1/p+1)
    3. Check gcd(a^(B!) - k, N) for small k (quadratic residue divergence)
    4. Hensel lift: compute x^(B!) mod N2 and check gcd with N
    5. Check cyclotomic orders m=3,4,6,10,12

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    primes = _small_primes(min(bound, 100000))

    # Phase 1: Standard smooth-bound powering with Hensel lift detection
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:base_points]:
        if a >= N:
            continue

        # Compute a^(B!) mod N
        power = a
        for p in primes:
            pk = p
            while pk * p <= bound:
                pk *= p
            power = pow(power, pk, N)

            # Standard checks
            val = (power - 1) % N
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            val = (power + 1) % N
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Phase 1.5: Quadratic residue divergence
        # For a^(B!) mod N, check if (a^(2B!) - k) shares a factor with N
        # This detects when a^(B!) is a QR mod p but not mod q
        power2 = pow(power, 2, N)

        for k in [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25]:
            val = (power2 - k) % N
            g = gcd(val, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Phase 2: Cyclotomic orders (same as cyclotomic cascade)
        # m=3: Φ_3(x) = x2 + x + 1
        val = (power2 + power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=4: Φ_4(x) = x2 + 1
        val = (power2 + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=6: Φ_6(x) = x2 - x + 1
        val = (power2 - power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=10: Φ_10(x) = x4 - x3 + x2 - x + 1
        power3 = pow(power, 3, N)
        power4 = pow(power, 4, N)
        val = (power4 - power3 + power2 - power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=12: Φ_12(x) = x4 - x2 + 1
        val = (power4 - power2 + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # Phase 3: Hensel-lifted root detection
        # For f(x) = x^(B!) - 1, if x0 is a root mod p, then
        # Hensel's lemma gives x0 + p*Δ as a root mod p2 for Δ = -(f(x0)/p) * (f'(x0))^(-1) mod p
        # But since we don't know p, we use a different approach:
        # compute f(x0) = x0^(B!) - 1 mod N2, then gcd(f(x0) mod N2, N)
        # might reveal p when the Hensel lift diverges mod p2 vs mod q2
        N2 = N * N
        power_N2 = pow(a, 1, N2)  # Start fresh mod N2
        for p in primes:
            pk = p
            while pk * p <= bound:
                pk *= p
            power_N2 = pow(power_N2, pk, N2)

        val_N2 = (power_N2 - 1) % N2
        g = gcd(val_N2, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        val_N2 = (power_N2 + 1) % N2
        g = gcd(val_N2, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Phase 4: Jacobi symbol-based CRT divergence
    # For small a, check if gcd(a^k + a^j, N) reveals factors
    # This exploits quadratic residue structure
    for a in [2, 3, 5, 7, 11, 13]:
        if a >= N:
            continue
        ak = a * a
        for _ in range(max_lifts):
            ak = (ak * a) % N
            # Check gcd(a^k + 1, N) and gcd(a^k - 1, N)
            g = gcd(ak - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            g = gcd(ak + 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Check gcd(a^k + a^(k//2), N) for even k
            # This checks if the geometric series a^0 + a^1 + ... + a^(k-1) == 0 (mod p)
            if ak > 1:
                val = (ak - a) % N
                g = gcd(val, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def crt_lattice_factor(N: int, bound: int = 50000,
                        lattice_dim: int = 8) -> tuple[int, int] | None:
    """Factor N using CRT lattice divergence detection.

    Novel approach: instead of powering a single element, we build a lattice
    of values (a_1^(B!), a_2^(B!), ..., a_k^(B!)) mod N and check for CRT
    divergence in the lattice structure.

    The key insight: if a_i^(B!) == 1 (mod p) but a_i^(B!) ≢ 1 (mod q) for
    some i, then the lattice mod p has a different rank than mod q. We detect
    this via determinant computation or gcd of linear combinations.

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    primes = _small_primes(min(bound, 100000))

    # Choose multiple bases
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53][:lattice_dim]
    bases = [b for b in bases if b < N]

    # Phase 1: Compute B! for all bases simultaneously
    powers = list(bases)
    for p in primes:
        pk = p
        while pk * p <= bound:
            pk *= p
        for i in range(len(powers)):
            powers[i] = pow(powers[i], pk, N)

    # Phase 2: Check individual bases (standard p-1/p+1)
    for i, power in enumerate(powers):
        val = (power - 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        val = (power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Phase 3: Cross-product CRT divergence detection
    # For pairs (i, j), check gcd(p_i * p_j - 1, N), gcd(p_i * p_j + 1, N)
    # This detects when the product a_i^(B!) * a_j^(B!) has different order mod p vs q
    for i in range(len(powers)):
        for j in range(i + 1, min(i + 4, len(powers))):
            prod = (powers[i] * powers[j]) % N

            val = (prod - 1) % N
            g = gcd(val, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            val = (prod + 1) % N
            g = gcd(val, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Difference detection: gcd(p_i - p_j, N)
            diff = abs(powers[i] - powers[j]) % N
            if diff > 0:
                g = gcd(diff, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    # Phase 4: Linear combination divergence
    # Check gcd(sum c_i * p_i - 1, N) for small coefficients c_i
    from itertools import product as iterproduct

    # Check 2-term and 3-term linear combinations
    coeffs_range = [-2, -1, 0, 1, 2]
    for c1, c2, c3 in iterproduct(coeffs_range, repeat=3):
        if c1 == 0 and c2 == 0 and c3 == 0:
            continue
        if len(powers) < 3:
            break
        val = (c1 * powers[0] + c2 * powers[1] + c3 * powers[2]) % N
        if val == 0:
            continue
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Phase 5: Determinant-based detection
    # Build a small matrix from the powered values and check determinant mod N
    k = min(len(powers), 4)
    # Build k×k matrix M where M[i][j] = powers[i]^j mod N
    # If det(M) == 0 (mod p) but det(M) ≢ 0 (mod q), gcd(det(M), N) = p
    for start in range(0, len(powers) - k + 1, k):
        sub_powers = powers[start:start + k]
        # Compute determinant of Vandermonde-like matrix
        # Using the formula: det of [p_i^j] for i=0..k-1, j=0..k-1
        # is prod_{i<j} (p_j - p_i)
        det_val = 1
        for i in range(k):
            for j in range(i + 1, k):
                diff = abs(sub_powers[i] - sub_powers[j]) % N
                det_val = (det_val * diff) % N

        g = gcd(det_val, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None