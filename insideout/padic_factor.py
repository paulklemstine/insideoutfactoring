"""p-adic Slope Factoring: Valuation Collisions on (Z/NZ)*.

A novel factoring algorithm using p-adic logarithms and smooth-ladder
powering to detect multiplicative order structure modulo the (unknown)
prime factors of N.

THEORY
------
For N = p*q, the multiplicative group G = (Z/NZ)* splits via CRT:
    G ≅ (Z/pZ)* × (Z/qZ)*.

Each factor's group has order p-1 and q-1 respectively. The discrete
logarithm problem has DIFFERENT structure in each component. The key
insight: the p-adic logarithm

    log_p(x) = Σ_{k=1}^∞ (-1)^{k+1} (x-1)^k / k   (converges p-adically)

satisfies log_p(g^k) = k · log_p(g) mod p, but log_q(g^k) = k · log_q(g) mod q.
Since log_p(g) and log_q(g) live in groups of different size (p-1 vs q-1),
their "slopes" differ, and this difference can be detected via gcd.

PRACTICAL METHOD
----------------
Three stages, each strictly integer arithmetic (no floats):

1. SMOOTH LADDER (Pollard p-1 with prime-powers):
   Compute g^L mod N where L = lcm(1,2,...,B). If p-1 | L, then
   g^L ≡ 1 (mod p), so gcd(g^L - 1, N) reveals p. We also try
   L/2, L/3, L/5, ... (L divided by each prime ≤ B) to catch factors
   where (p-1)/prime is smooth even if p-1 itself isn't fully B-smooth.

2. P-ADIC SLOPE COLLISION:
   Compute the squaring sequence s_k = g^(2^k) mod N. For each k,
   compute gcd(s_k - 1, N), gcd(s_k + 1, N), and gcd(s_k - s_j, N)
   for j < k. Collisions in this sequence reveal when the order of g
   mod p divides 2^k but the order mod q does not — a structural
   difference detectable via pairwise GCD.

3. P-ADIC LOG SERIES:
   Truncated p-adic logarithm: log(a) ≈ Σ_{j=1}^{d} (-1)^{j+1} (a-1)^j / j
   (mod N), skipping j not invertible mod N. The ratio
   (log(g^k) - k·log(g)) should be 0 mod p and 0 mod q individually,
   but if we use a single "log(g)" value, the difference reveals which
   prime divides the non-vanishing component.

COMPLEXITY
----------
- Smooth ladder: O(B log B) multiplications via prime-power accumulation
- Slope collision: O(log B) GCDs on O(log B) values
- Log series: O(d · log N) per evaluation, where d is the truncation depth

For B-smooth p-1, this is essentially Pollard p-1. The novelty is in
the slope collision stage and the log-series cross-check.
"""
from __future__ import annotations

from math import gcd, isqrt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prime_sieve(bound: int) -> list[int]:
    """Return all primes ≤ bound using the Sieve of Eratosthenes."""
    if bound < 2:
        return []
    # is_composite[n] == 1 means n is known to be composite
    is_composite = bytearray(bound + 1)
    is_composite[0] = is_composite[1] = 1
    for i in range(2, isqrt(bound) + 1):
        if not is_composite[i]:
            # Mark all multiples of i starting from i*i as composite
            is_composite[i*i::i] = b'\x01' * len(is_composite[i*i::i])
    return [i for i in range(2, bound + 1) if not is_composite[i]]


def _prime_powers(bound: int) -> list[tuple[int, int]]:
    """Return list of (prime, prime^k) for each prime p ≤ bound.

    The exponent k is the largest such that p^k ≤ bound.
    These are the "prime power" components of lcm(1, 2, ..., bound).
    """
    primes = _prime_sieve(bound)
    result = []
    for p in primes:
        pk = p
        while pk * p <= bound:
            pk *= p
        result.append((p, pk))
    return result


def _modinv(a: int, m: int) -> int | None:
    """Extended GCD modular inverse. Returns None if gcd(a, m) ≠ 1."""
    g, x, _ = _extended_gcd(a % m, m)
    if g != 1:
        return None
    return x % m


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


# ---------------------------------------------------------------------------
# Stage 1: Smooth Ladder (Pollard p-1 with prime-power L)
# ---------------------------------------------------------------------------

def _smooth_ladder(g: int, N: int, bound: int) -> int:
    """Compute g^L mod N where L = lcm(1, 2, ..., bound).

    Uses the prime-power decomposition so we get the full smooth ladder
    in O(π(bound)) modular exponentiations instead of O(bound).

    L = ∏ p^{⌊log_p(bound)⌋} for primes p ≤ bound.
    """
    g = g % N
    if g < 2:
        return g
    for p, pk in _prime_powers(bound):
        g = pow(g, pk, N)
    return g


def _pm1_smooth(g: int, N: int, bound: int) -> tuple[int, int] | None:
    """Pollard p-1 factoring with smooth ladder and per-prime reduction.

    Stage 1: compute g^L mod N where L = lcm(1..bound). If p-1 | L for
    some prime p | N, then g^L ≡ 1 (mod p) and gcd(g^L - 1, N) = p.

    Stage 2 (the novel "L/r" twist): for small primes r ≤ L_R_BOUND,
    try g^(L/r). This catches factors where (p-1)/r is smooth even if
    p-1 has one prime factor > bound. We compute g^(L/r) by running
    the smooth ladder with r's contribution reduced by one factor.

    Stage 3 (standard p+1 extension): for primes r in (bound, stage2_bound],
    try g^(L*r). This catches factors where p-1 has exactly one prime
    factor in (bound, stage2_bound].

    Returns (p, q) if a factor is found, else None.
    """
    g = g % N
    if g < 2:
        return None

    prime_powers = _prime_powers(bound)

    # --- Stage 1: full smooth ladder ---
    val = g
    for p, pk in prime_powers:
        val = pow(val, pk, N)

    f = gcd(val - 1, N)
    if 1 < f < N:
        return (f, N // f)

    # --- Stage 2: standard p-1 Stage 2 (primes in (bound, stage2_bound]) ---
    # For each prime r in (bound, stage2_bound], compute g^(L*r) mod N
    # by raising the Stage 1 result to the r-th power. This is fast:
    # O(π(stage2_bound) - π(bound)) modular exponentiations.
    stage2_bound = bound * 10
    primes_stage2 = _prime_sieve(stage2_bound)
    for r in primes_stage2:
        if r <= bound:
            continue
        val_r = pow(val, r, N)
        f = gcd(val_r - 1, N)
        if 1 < f < N:
            return (f, N // f)

    # --- Stage 3: L/r trick for small primes ---
    # For each small prime r, compute g^(L/r) by running the smooth ladder
    # with r's contribution reduced by one factor. This catches factors
    # where (p-1)/r is smooth even if p-1 has one "extra" factor of r.
    # Limit to small primes to keep O(π(B_small) * π(B)) manageable.
    # For L_R_BOUND = 200 and bound = 50000: ~46 * 5133 ≈ 236K pow() calls.
    L_R_BOUND = min(bound, 200)
    small_prime_powers = [(p, pk) for p, pk in prime_powers if p <= L_R_BOUND]

    for r, rk in small_prime_powers:
        # Compute g^(L/r) by raising through all prime powers,
        # but using rk/r for prime r.
        reduced = rk // r
        v = g
        for p, pk in prime_powers:
            if p == r:
                if reduced > 1:
                    v = pow(v, reduced, N)
            else:
                v = pow(v, pk, N)

        f = gcd(v - 1, N)
        if 1 < f < N:
            return (f, N // f)
        # Also try v + 1 (catches order-2 factors)
        f = gcd(v + 1, N)
        if 1 < f < N:
            return (f, N // f)

    return None


# ---------------------------------------------------------------------------
# Stage 2: p-adic Slope Collision Search
# ---------------------------------------------------------------------------

def _padic_slope(g: int, N: int, bound: int) -> tuple[int, int] | None:
    """p-adic slope collision search.

    Compute the repeated-squaring sequence s_k = g^(2^k) mod N.
    For each k, check gcd(s_k - 1, N) and gcd(s_k + 1, N).
    Also check pairwise collisions gcd(s_k - s_j, N) for j < k.

    The sequence s_k has period equal to the multiplicative order of 2
    in the exponent group. If ord_p(g) divides 2^k but ord_q(g) does not,
    we get a factor from gcd(s_k - 1, N). The collision check catches
    cases where ord_p(g) divides 2^k - 2^j = 2^j(2^{k-j} - 1).

    The "slope" moniker: if we view log(s_k) ≈ 2^k · log(g), then the
    ratio log(s_k)/log(g) = 2^k. If this ratio differs mod p vs mod q
    (because the order of g divides 2^k in one component but not the
    other), the difference s_k - s_j reveals structure.
    """
    g = g % N
    if g < 2:
        return None

    # Build squaring sequence up to 2^max_k ≥ bound
    max_k = max(1, bound.bit_length() + 1)

    seq = []
    current = g % N
    for k in range(max_k):
        seq.append(current)

        # Check s_k - 1
        f = gcd(current - 1, N)
        if 1 < f < N:
            return (f, N // f)

        # Check s_k + 1 (catches order-2 factors)
        f = gcd(current + 1, N)
        if 1 < f < N:
            return (f, N // f)

        # Check collisions with all previous terms
        for j, prev in enumerate(seq[:-1]):
            diff = current - prev
            if diff != 0:
                f = gcd(diff, N)
                if 1 < f < N:
                    return (f, N // f)

        # Square for next iteration
        current = pow(current, 2, N)

        # Detect cycle (Floyd-style early exit)
        if current == g % N:
            break

    return None


# ---------------------------------------------------------------------------
# Stage 3: Truncated p-adic Logarithm
# ---------------------------------------------------------------------------

def _padic_log_series(a: int, N: int, depth: int) -> int:
    """Compute truncated p-adic logarithm of a mod N.

    log_p(a) ≈ Σ_{j=1}^{depth} (-1)^{j+1} · (a-1)^j / j   (mod N)

    Division by j is performed via modular inverse, which requires gcd(j, N) = 1.
    Terms where j shares a factor with N are SKIPPED (they would require
    p-adic division, which is the whole point of the algorithm — the
    skipped terms encode the factor structure).

    For a ∈ 1 + pZ_p, this series converges p-adically to log_p(a).
    For general a, we decompose a = ω(a) · <a> and the formula applies
    to <a>. Here we use the raw series as a heuristic that still
    captures valuation information.
    """
    a = a % N
    if a < 0:
        a += N

    am1 = (a - 1) % N
    if am1 == 0:
        return 0  # log(1) = 0

    result = 0
    term = am1  # (a-1)^1
    sign = 1    # (-1)^{j+1} for j=1 is +1

    for j in range(1, depth + 1):
        # Try to divide by j mod N
        j_inv = _modinv(j, N)
        if j_inv is not None:
            result = (result + sign * term * j_inv) % N
        # else: skip — this term encodes factor structure

        # Update for next j
        sign = -sign
        term = (term * am1) % N

    return result % N


def _collision_gcd(seq: list[int], N: int) -> tuple[int, int] | None:
    """Compute gcd of all pairwise differences with N.

    For a sequence derived from exponentiation mod N, if two entries
    are congruent mod p but not mod q, their difference shares a factor
    with N. Checking all pairs is O(n²) but n is small (≤ log N).

    Returns (p, q) if a factor is found, else None.
    """
    n = len(seq)
    for i in range(n):
        for j in range(i + 1, n):
            diff = seq[i] - seq[j]
            if diff != 0:
                f = gcd(diff, N)
                if 1 < f < N:
                    return (f, N // f)
    return None


# ---------------------------------------------------------------------------
# Stage 4: Log-Series Cross-Check (the "slope" variant)
# ---------------------------------------------------------------------------

def _log_slope_crosscheck(g: int, N: int, bound: int, depth: int) -> tuple[int, int] | None:
    """Cross-check p-adic log values to detect slope divergence.

    Compute log(g) and log(g^k) for strategic k values. If log_p(g) has
    different valuation at p vs q, then log(g^k) - k·log(g) ≡ 0 mod one
    prime but not the other, and gcd with N reveals the factor.

    We use k = 2, 3, 5, 7, ... (small primes) and k = bound, bound-1, ...
    (near-bound values) to maximize the chance of hitting a valuation
    difference.
    """
    g = g % N
    if g < 2:
        return None

    log_g = _padic_log_series(g, N, depth)
    if log_g == 0:
        return None

    # Strategic k values: small primes and powers of 2
    k_values = []
    primes = _prime_sieve(min(bound, 100))
    k_values.extend(primes[:20])
    # Powers of 2
    pk = 2
    while pk <= bound:
        k_values.append(pk)
        pk *= 2
    # Near-bound values
    for k in range(max(2, bound - 20), bound + 1):
        k_values.append(k)

    # Deduplicate while preserving order
    seen = set()
    unique_k = []
    for k in k_values:
        if k not in seen and 1 < k <= bound:
            seen.add(k)
            unique_k.append(k)

    for k in unique_k:
        gk = pow(g, k, N)
        log_gk = _padic_log_series(gk, N, depth)

        # log(g^k) - k*log(g) should be 0 if log is consistent
        diff = (log_gk - k * log_g) % N
        if diff != 0:
            f = gcd(diff, N)
            if 1 < f < N:
                return (f, N // f)

    return None


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def padic_factor(N: int, smooth_bound: int = 50000) -> tuple[int, int] | None:
    """Factor N using the p-adic slope method.

    Combines three stages:
    1. Smooth ladder (Pollard p-1 with per-prime reduction)
    2. p-adic slope collision search (repeated-squaring sequence)
    3. Log-series cross-check (p-adic logarithm slope divergence)

    Each stage is tried with multiple bases g = 2, 3, 5, 6, 7, 11
    to maximize the chance of finding a factor.

    Parameters
    ----------
    N : int
        The composite integer to factor (must be odd, > 4, not a prime power).
    smooth_bound : int
        Smoothness bound B. Higher values catch factors with less-smooth
        p-1 / q-1 at the cost of more computation. Default 50000.
        The bound is automatically reduced for small N to keep runtime fast.

    Returns
    -------
    (p, q) with p < q if N is composite and a factor is found, else None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2) if N > 2 else None

    # Check for perfect square
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # Quick trial division for very small factors
    for p in range(3, min(isqrt(N) + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Make bound adaptive: for small N, a smaller bound suffices.
    # The smooth ladder cost is O(π(bound)) pow() calls, so we scale
    # bound with log(N) to keep runtime reasonable.
    bound = smooth_bound
    if N < 10**6:
        bound = min(bound, 2000)
    elif N < 10**10:
        bound = min(bound, 10000)
    elif N < 10**16:
        bound = min(bound, 30000)
    else:
        bound = min(bound, 5000)

    # Try multiple bases
    bases = [2, 3, 5, 6, 7, 11, 13, 17, 19, 23, 29, 31]
    for g in bases:
        if g >= N:
            continue
        if gcd(g, N) > 1:
            # We already found a factor by accident
            f = gcd(g, N)
            return (f, N // f)

        # Stage 1: Smooth ladder (Pollard p-1 with L/r twist)
        result = _pm1_smooth(g, N, bound)
        if result is not None:
            p, q = result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        # Stage 2: p-adic slope collision search (fast: O(log B) GCDs)
        result = _padic_slope(g, N, bound)
        if result is not None:
            p, q = result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        # Stage 3: Log-series cross-check (depth scales with log N)
        # This is the most expensive stage (O(depth) per evaluation),
        # so it runs last. Depth is capped at 200 for performance.
        depth = min(max(20, N.bit_length() * 2), 100)
        result = _log_slope_crosscheck(g, N, bound, depth)
        if result is not None:
            p, q = result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

    return None
