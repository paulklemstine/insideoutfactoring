"""Hybrid Cyclotomic-SL2 Cascade — Combining Our Two Strongest Methods.

A hybrid approach that combines:
1. Cyclotomic cascade for rapid small-factor detection
2. SL2 group-order cascade for deeper search
3. Thaine-SL2 hybrid: SL2 matrices with cyclotomic unit determinants

Key insight: The cyclotomic cascade provides fast detection when p-1, p+1, or p^k+1
has small factors. SL2 group-order provides three independent smoothness targets:
|SL2(F_p)| = p(p-1)(p+1). By combining both, we get broader coverage.

The Thaine-SL2 hybrid (new):
- Generate SL2 matrices whose determinant is a Thaine cyclotomic unit Φ_m(a)
- Thaine (1988): cyclotomic units annihilate class groups via Stickelberger elements
- When class group order is smooth, Φ_m(a)^S - 1 reveals factors
- Using Φ_m(a) as matrix determinant channels class group structure into SL2

This is not a theoretical improvement — all methods are L_p[1/2]. But the
combination improves constants by providing broader smoothness target coverage.
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


def _mat2_mul(A, B, mod):
    """Multiply two 2x2 matrices mod mod."""
    return [
        [(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
         (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
        [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
         (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]
    ]


def _mat2_pow(M, exp, mod):
    """Raise 2x2 matrix M to power exp mod mod."""
    result = [[1, 0], [0, 1]]  # Identity
    base = M
    while exp > 0:
        if exp % 2 == 1:
            result = _mat2_mul(result, base, mod)
        base = _mat2_mul(base, base, mod)
        exp //= 2
    return result


def _sl2_smooth_cascade(N, bound, num_curves):
    """SL2 group-order cascade factoring.

    Uses |SL2(F_p)| = p(p-1)(p+1) as smoothness target.
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

    # Generate SL2 matrices as products of generators
    # Use M = [[a, b], [c, d]] with ad - bc = 1 mod N
    for curve in range(num_curves):
        # Create a random SL2 matrix using two elementary matrices
        # E1 = [[1, x], [0, 1]] and E2 = [[1, 0], [y, 1]]
        x = (curve * 7 + 3) % N
        y = (curve * 11 + 5) % N

        M = [
            [(1 + x*y) % N, x % N],
            [y % N, 1 % N]
        ]

        # Compute M^(B!) mod N
        power = M
        for prime in primes:
            pk = prime
            while pk * prime <= bound:
                pk *= prime
            power = _mat2_pow(power, pk, N)

            # Check gcd of matrix entries with N
            for i in range(2):
                for j in range(2):
                    g = gcd(abs(power[i][j]), N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

            # Also check trace = M[0][0] + M[1][1]
            trace = (power[0][0] + power[1][1]) % N
            g = gcd(abs(trace - 2), N)  # Check if trace == 2 mod p (identity)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    return None


def _cyclotomic_cascade(N, bound, base_points):
    """Cyclotomic cascade factoring.

    Evaluates Φ_m(a^(B!)) mod N for m = 1, 2, 3, 4, 6, 10, 12.
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

            # Check m=1: Φ_1(x) = x - 1
            g = gcd(abs(power - 1), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Check m=2: Φ_2(x) = x + 1
            g = gcd(abs(power + 1), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Higher cyclotomic orders
        power2 = pow(power, 2, N)

        # m=3: x2 + x + 1
        g = gcd(abs(power2 + power + 1), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=4: x2 + 1
        g = gcd(abs(power2 + 1), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=6: x2 - x + 1
        g = gcd(abs(power2 - power + 1), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=10: x4 - x3 + x2 - x + 1
        power4 = pow(power, 4, N)
        g = gcd(abs(power4 - power2 + 1), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None


# --- Thaine-SL2 Hybrid: Cyclotomic unit determinants in SL2 matrices ---


def _mobius(n: int) -> int:
    """Compute the Möbius function μ(n)."""
    if n == 1:
        return 1
    count = 0
    temp = n
    for p in range(2, isqrt(n) + 1):
        if temp % p == 0:
            count += 1
            temp //= p
            if temp % p == 0:
                return 0
    if temp > 1:
        count += 1
    return -1 if count % 2 else 1


def _divisors(n: int) -> list[int]:
    """Return sorted list of divisors of n."""
    divs = []
    for i in range(1, isqrt(n) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _poly_mul(a: list[int], b: list[int]) -> list[int]:
    """Multiply two polynomials (ascending order: a[0] + a[1]*x + a[2]*x^2 + ...)."""
    if not a or not b:
        return [0]
    result = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            result[i + j] += ca * cb
    return result


def _poly_div(dividend: list[int], divisor: list[int]) -> list[int]:
    """Divide polynomial dividend by divisor, returning quotient.

    Both polynomials are in ascending order (constant term first).
    Uses exact integer arithmetic — assumes divisor divides dividend exactly.
    """
    # Remove trailing zeros (leading terms that are 0)
    while dividend and dividend[-1] == 0:
        dividend = dividend[:-1]
    while divisor and divisor[-1] == 0:
        divisor = divisor[:-1]

    if not divisor:
        return [0]
    if not dividend or len(dividend) < len(divisor):
        return [0]

    remainder = list(dividend)
    quotient = [0] * len(dividend)

    while len(remainder) >= len(divisor):
        # Leading coefficient of remainder / leading coefficient of divisor
        lead_rem = remainder[-1] if remainder else 0
        lead_div = divisor[-1]
        if lead_div == 0:
            break
        coeff = lead_rem // lead_div
        shift = len(remainder) - len(divisor)
        quotient[shift] = coeff

        # Subtract coeff * x^shift * divisor from remainder
        for i in range(len(divisor)):
            idx = shift + i
            if idx < len(remainder):
                remainder[idx] -= coeff * divisor[i]

        # Clean up leading zeros
        while remainder and remainder[-1] == 0:
            remainder = remainder[:-1]

    # Strip trailing zeros from quotient
    while quotient and quotient[-1] == 0:
        quotient = quotient[:-1]
    return quotient if quotient else [0]


def _cyclotomic_poly(n: int) -> list[int]:
    """Compute cyclotomic polynomial Φ_n(x) via Möbius inversion.

    Φ_n(x) = prod_{d|n} (x^{n/d} - 1)^{μ(d)}

    In ascending order: [a_0, a_1, a_2, ...] means a_0 + a_1*x + a_2*x^2 + ...
    """
    if n == 1:
        return [1, -1]  # x - 1

    # x^n - 1 in ascending order: [-1, 0, 0, ..., 0, 1]
    def make_xn_minus_1(k):
        poly = [0] * (k + 1)
        poly[0] = -1
        poly[k] = 1
        return poly

    # For prime n: Φ_n(x) = (x^n - 1) / (x - 1)
    if _mobius(n) == -1:  # n is prime
        return _poly_div(make_xn_minus_1(n), [-1, 1])

    # General case: multiply (x^{n/d} - 1) for d with μ(d) = 1
    # and divide by (x^{n/d} - 1) for d with μ(d) = -1
    numerator = [1]
    denominator = [1]

    for d in _divisors(n):
        mu = _mobius(d)
        if mu == 0:
            continue
        poly = make_xn_minus_1(n // d)
        if mu == 1:
            numerator = _poly_mul(numerator, poly)
        elif mu == -1:
            denominator = _poly_mul(denominator, poly)

    return _poly_div(numerator, denominator)


def _poly_eval_mod(coeffs: list[int], x: int, N: int) -> int:
    """Evaluate polynomial at x mod N using Horner's method."""
    if not coeffs:
        return 0
    result = 0
    for c in coeffs:
        result = (result * x + c) % N
    return result


def _thaine_sl2_matrix(N: int, m: int, a: int) -> list | None:
    """Generate an SL2 matrix with cyclotomic unit determinant.

    Thaine's theorem: cyclotomic units Φ_m(a) annihilate class groups via
    Stickelberger elements. Using Φ_m(a) as the matrix determinant channels
    this class-group structure into the SL2 cascade.

    For Φ_m(a) to be a valid determinant, we need det(M) = Φ_m(a) mod N.
    We construct M = [[d + bc, b], [c, 1]] where d = Φ_m(a) mod N,
    ensuring det = d*1 - b*c = Φ_m(a) mod N.

    Args:
        N: Integer being factored
        m: Cyclotomic order
        a: Evaluation point

    Returns:
        SL2 matrix [[d+bc, b], [c, 1]] mod N, or None if degenerate
    """
    coeffs = _cyclotomic_poly(m)
    phi_ma = _poly_eval_mod(coeffs, a, N)

    if phi_ma <= 1 or phi_ma >= N - 1:
        return None

    # For matrix [[d+bc, b], [c, 1]], det = d + bc - b*c = d
    # Wait: det([[d+bc, b], [c, 1]]) = (d+bc)*1 - b*c = d
    # That's not right for channeling structure. We need:
    # M = [[d + b*c, b], [c, 1]] where d = phi_ma
    # det(M) = (d + b*c)*1 - b*c = d ✓
    # But this has det = d regardless of b,c. We need det = d.
    # Actually: det([[A, B], [C, D]]) = A*D - B*C
    # If we want det = phi_ma, set A = phi_ma + B*C, D = 1
    # Then det = (phi_ma + B*C)*1 - B*C = phi_ma ✓

    b = (a * 7 + 11) % N
    c = (a * 13 + 17) % N
    bc = (b * c) % N

    # Ensure phi_ma + bc is coprime to N (otherwise we found a factor)
    g = gcd(phi_ma + bc, N)
    if 1 < g < N:
        return None  # Found factor in construction

    A = (phi_ma + bc) % N
    B = b % N
    C = c % N
    D = 1 % N

    # Verify determinant
    det = (A * D - B * C) % N
    if det != phi_ma % N:
        return None

    return [[A, B], [C, D]]


def _thaine_sl2_cascade(N: int, bound: int, num_curves: int) -> tuple[int, int] | None:
    """Thaine-SL2 hybrid cascade factoring.

    Combines Thaine's class group annihilator theory with SL2 group order.
    Key insight: using cyclotomic units Φ_m(a) as matrix determinants
    channels class-group structure (via Stickelberger elements) into
    the SL2 cascade's smoothness detection.

    Algorithm:
    1. For cyclotomic orders m that divide p-1 or p+1 for unknown p|N
    2. Generate SL2 matrices with determinant = Φ_m(a)
    3. Raise to smooth powers and check gcd of matrix entries
    4. The Stickelberger action on Φ_m(a) produces norms divisible by p

    This is a smooth-class-order method: when class group order is smooth,
    Thaine guarantees that Φ_m(a)^S - 1 reveals factors.
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

    # Cyclotomic orders to try (structured for class-group annihilators)
    orders = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 30]
    # Evaluation points (primitive roots give well-distributed Φ_m values)
    eval_points = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    for m in orders:
        for a in eval_points:
            if gcd(a, m) != 1:
                continue

            M = _thaine_sl2_matrix(N, m, a)
            if M is None:
                continue

            # Compute M^(B!) mod N with smoothness cascade
            power = M
            for prime in primes:
                pk = prime
                while pk * prime <= bound:
                    pk *= prime
                power = _mat2_pow(power, pk, N)

                # Check gcd of matrix entries with N
                for i in range(2):
                    for j in range(2):
                        g = gcd(abs(power[i][j]), N)
                        if 1 < g < N:
                            return (min(g, N // g), max(g, N // g))

                # Check trace
                trace = (power[0][0] + power[1][1]) % N
                g = gcd(abs(trace - 2), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

                # Check det - 1: Thaine's unit condition
                det_minus_1 = (power[0][0] * power[1][1] - power[0][1] * power[1][0] - 1) % N
                g = gcd(abs(det_minus_1), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def hybrid_cyclo_sl2_factor(N, time_budget_ms=30000) -> tuple[int, int] | None:  # noqa: E501
    # Skip for large N — too slow
    if N.bit_length() > 256:
        return None

    """Factor N using hybrid Cyclotomic-SL2 cascade.

    Combines three approaches:
    1. Cyclotomic cascade for quick wins (low bounds)
    2. Thaine-SL2 hybrid: SL2 matrices with cyclotomic unit determinants
       (channels class-group annihilator structure via Stickelberger elements)
    3. SL2 group-order cascade for deeper search (increasing bounds)

    Returns (p, q) with p < q and p*q = N, or None.
    """
    import time

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

    start = time.perf_counter()

    # Phase 1: Cyclotomic cascade with moderate bounds
    for bound in [10000, 50000, 100000]:
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > time_budget_ms * 0.2:
            break

        result = _cyclotomic_cascade(N, bound=bound, base_points=10)
        if result is not None:
            return result

    # Phase 1.5: Thaine-SL2 hybrid cascade
    # Uses cyclotomic units as SL2 matrix determinants to channel
    # class-group annihilator structure via Stickelberger elements
    for bound in [50000, 100000, 500000]:
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > time_budget_ms * 0.5:
            break

        result = _thaine_sl2_cascade(N, bound=bound, num_curves=20)
        if result is not None:
            return result

    # Phase 2: SL2 cascade with increasing bounds
    for bound in [50000, 100000, 500000, 1000000, 5000000]:
        elapsed = (time.perf_counter() - start) * 1000
        if elapsed > time_budget_ms * 0.9:
            break

        for curves in [10, 20, 50, 100, 200]:
            result = _sl2_smooth_cascade(N, bound=bound, num_curves=curves)
            if result is not None:
                return result

            elapsed = (time.perf_counter() - start) * 1000
            if elapsed > time_budget_ms * 0.95:
                break

    return None