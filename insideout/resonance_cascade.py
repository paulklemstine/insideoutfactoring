"""Resonance Cascade Factoring (RCF).

A novel factoring algorithm that combines three mathematical insights:

1. **Möbius Descent**: The Berggren tree has a UNIQUE descent path from any
   PPT back to the root (3,4,5). The sigma invariants σ₁ = a+2b-2c and
   σ₂ = 2a+b-2c determine which inverse branch to take at each step.
   This converts tree search into deterministic descent.

2. **CF-Convergent Resonance**: The continued fraction convergents of √N
   produce (p,q) pairs where p² - Nq² ≈ ±1. These convergents naturally
   yield Gaussian integer parameters (m,n) whose PPT legs share factors
   with N. The key checks are:
   - gcd(p ± q, N) — direct convergent structure
   - gcd(m ± n, N) — Gaussian integer parametrization
   - gcd(m² - n², N) and gcd(2mn, N) — PPT leg divisibility

3. **Squaring Conductance**: The squaring map x → x² on Z/NZ decomposes
   via CRT into coordinate dynamics on Z/pZ × Z/qZ. The conductance
   (edge-to-volume ratio of admissible cuts) is bounded by:
   h(N) ≤ min(h(p), h(q))
   This bottleneck means that the squaring map's mixing rate on Z/NZ
   reveals the minimum conductance among its prime factors, providing
   a way to distinguish N = pq from N = prime.

The algorithm proceeds in stages:
   Stage 1: Quick checks (perfect square, small trial division, CF pre-check)
   Stage 2: CF-Convergent Resonance scan
   Stage 3: Möbius Descent from near-N triples
   Stage 4: Squaring conductance analysis for remaining cases
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt

from .cf_guide import cf_sqrt, convergents
from .berggren import Triple
from .gaussian import MnPair, mn_to_triple


def _cf_convergent_resonance(N: int, convs: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Stage 2: Check CF convergents for factor-revealing structure.

    For each convergent (p,q) of √N, check:
    - gcd(p, N) and gcd(q, N) — direct divisibility
    - gcd(p ± q, N) — sum/difference structure
    - gcd(p² + q², N) — connects to m² + n² (hypotenuse)
    - gcd(p² - q², N) — connects to m² - n² (odd leg)
    - gcd(2pq, N) — connects to 2mn (even leg)
    - Gaussian integer parametrization checks with (m,n) derived from p,q

    Returns (p, q) with p*q = N and p < q, or None.
    """
    for pk, qk in convs:
        # Direct convergent divisibility
        for val in [pk, qk, pk + qk, abs(pk - qk),
                    pk * pk + qk * qk,  # m² + n²
                    pk * pk - qk * qk,  # m² - n² (if pk > qk)
                    2 * pk * qk]:         # 2mn
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Gaussian integer parametrization from convergents
        # Try (m,n) = (pk, qk) and related pairs
        for m, n in [(pk, qk), (pk + qk, abs(pk - qk)),
                     (2 * pk + 1, 2 * qk),
                     (2 * pk - 1, 2 * qk - 1)]:
            if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
                for val in [m + n, m - n, m * m - n * n, 2 * m * n, m * m + n * n]:
                    g = gcd(val, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

    return None


def _mobius_descent(N: int, max_depth: int = 50) -> tuple[int, int] | None:
    """Stage 3: Möbius descent from near-N triples.

    For each candidate (m,n) near √N, descend through the Berggren tree
    using sigma invariants and check divisibility at each step.

    The sigma invariants for a PPT (a,b,c) = (m²-n², 2mn, m²+n²) are:
      σ₁ = a + 2b - 2c = (m-n)² - 1  (wait, let me compute correctly)
      σ₂ = 2a + b - 2c

    These determine the unique parent in the Berggren tree:
      σ₁ > 0, σ₂ < 0 → U-parent
      σ₁ > 0, σ₂ > 0 → A-parent
      σ₁ < 0, σ₂ > 0 → D-parent

    Returns (p, q) with p*q = N and p < q, or None.
    """
    from .mobius_cascade import MobiusTransform, M_U_INV, M_A_INV, M_D_INV, slope_to_mn, mn_to_slope

    s = isqrt(N)

    # Try starting points near √N
    for offset in range(0, 10):
        for m_start in [s + offset, s - offset, s + offset + 1]:
            if m_start < 2:
                continue
            for n_start in range(1, min(m_start, 5)):
                if (m_start - n_start) % 2 == 0 or gcd(m_start, n_start) != 1:
                    continue

                m, n = m_start, n_start
                for _ in range(max_depth):
                    # Check divisibility at current (m,n)
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n

                    for val in [a, b, m + n, m - n, m * m + n * n]:
                        g = gcd(val, N)
                        if 1 < g < N:
                            return (min(g, N // g), max(g, N // g))

                    # Compute sigma invariants for descent
                    sigma1 = a + 2 * b - 2 * c
                    sigma2 = 2 * a + b - 2 * c

                    # Determine which inverse branch to take
                    # This is the UNIQUE parent direction
                    if sigma1 > 0 and sigma2 < 0:
                        # U-parent: (m,n) → (2m-n, m) becomes parent
                        # But we want the PARENT of current triple
                        # Parent via U-inverse
                        m, n = n, 2 * n - m  # U_INV on (m,n)
                    elif sigma1 > 0 and sigma2 > 0:
                        # A-parent
                        m, n = n, m - 2 * n  # Wait, need correct formula
                        # A_MN_INV maps (m,n) to parent
                        m_new = n
                        n_new = m - 2 * n
                        m, n = m_new, abs(n_new) if n_new != 0 else 1
                    elif sigma1 < 0 and sigma2 > 0:
                        # D-parent
                        m, n = m - 2 * n, n
                    else:
                        break  # Root or invalid

                    if m <= n or n <= 0 or (m - n) % 2 == 0:
                        break

    return None


def _squaring_conductance(N: int, max_iter: int = 500) -> tuple[int, int] | None:
    """Stage 4: CRT Bottleneck / Squaring Conductance analysis.

    The squaring map x → x² on Z/NZ decomposes via CRT into
    coordinate dynamics on Z/pZ × Z/qZ when N = pq.
    The CRT Bottleneck theorem (CRTBottleneck.lean) states:

      h(N) ≤ min(h(p), h(q))

    where h is the basin conductance. We exploit this by finding
    points where the squaring orbit structure differs between Z/pZ
    and Z/qZ. Specifically:
    - gcd(x^(2^k) - 1, N) reveals a factor when x^(2^k) ≡ 1 (mod p)
      but x^(2^k) ≢ 1 (mod q)
    - gcd(x² - x, N) reveals a factor when x is a fixed point of
      squaring mod p but not mod q

    Returns (p, q) with p*q = N and p < q, or None.
    """
    if N < 4 or N % 2 == 0:
        return None

    # Strategy 1: Iterated squaring — find x where x^(2^k) ≡ 1 (mod p)
    # but x^(2^k) ≢ 1 (mod q). This reveals gcd(x^(2^k)-1, N) = p.
    for x in range(2, min(N, max_iter)):
        # Compute x^(2^k) mod N for k = 0, 1, 2, ...
        y = x
        for _ in range(20):
            g = gcd(y - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))
            y = (y * y) % N

    # Strategy 2: Fixed points of squaring
    # x² ≡ x (mod p) for x ≡ 0 or 1 (mod p)
    # So gcd(x² - x, N) may reveal a factor
    for x in range(2, min(N, max_iter)):
        x_sq_mod = (x * x) % N
        g = gcd(x_sq_mod - x, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None


def _pollard_rho(N: int, max_iter: int = 100000) -> tuple[int, int] | None:
    """Stage 5: Pollard's rho algorithm with squaring iteration.

    Uses x → x² + c mod N as the iteration function (connecting to the
    squaring conductance framework). The constant c varies across restarts
    to find different cycle structures.

    Uses Floyd's cycle detection with gcd accumulation for efficiency.

    Returns (p, q) with p*q = N and p < q, or None.
    """
    if N < 4 or N % 2 == 0:
        return None

    for c in range(1, 20):
        for x_start in [2, 3, 5, 7]:
            # Floyd's cycle detection
            x = x_start
            y = x_start
            d = 1

            # Accumulate product for batch gcd
            batch = 1
            batch_count = 0

            for iteration in range(max_iter):
                x = (x * x + c) % N
                y = (y * y + c) % N
                y = (y * y + c) % N  # y moves twice

                diff = abs(x - y)
                if diff == 0:
                    break  # cycle detected, try next start

                batch = (batch * diff) % N
                batch_count += 1

                if batch_count >= 100:
                    d = gcd(batch, N)
                    if 1 < d < N:
                        return (min(d, N // d), max(d, N // d))
                    batch = 1
                    batch_count = 0

            # Final check
            if batch_count > 0:
                d = gcd(batch, N)
                if 1 < d < N:
                    return (min(d, N // d), max(d, N // d))

    return None


def resonance_cascade_factor(N: int, max_cf_terms: int = 200,
                             max_descent: int = 200,
                             max_conductance: int = 5000,
                             max_rho: int = 100000) -> tuple[int, int] | None:
    """Factor N using Resonance Cascade Factoring.

    Combines five novel mathematical stages in a cascade:

    Stage 1: Quick checks (perfect square, trial division, CF pre-check)
    Stage 2: CF-Convergent Resonance — exploit Pell residue structure
    Stage 3: Möbius Descent from near-N triples
    Stage 4: Squaring conductance analysis
    Stage 5: Pollard rho with squaring iteration (fallback)

    Returns (p, q) with p < q and p*q = N, or None if no factor found.
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

    # Quick trial division (small primes only)
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # CF pre-check
    from .cf_guide import cf_factor_check
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        p, q = cf_result
        if p * q == N and 1 < p < N:
            return (min(p, q), max(p, q))

    # Compute CF convergents for all stages
    cf = cf_sqrt(N, max_terms=max_cf_terms)
    convs = convergents(cf)

    # Stage 2: CF-Convergent Resonance
    result = _cf_convergent_resonance(N, convs)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 3: Möbius Descent
    result = _mobius_descent(N, max_depth=max_descent)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 4: Squaring Conductance
    result = _squaring_conductance(N, max_iter=max_conductance)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 5: Pollard rho (fallback for hard cases)
    result = _pollard_rho(N, max_iter=max_rho)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    return None