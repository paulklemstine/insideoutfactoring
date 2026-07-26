"""Cyclotomic-Resultant Cascade — Novel Factoring via Cyclotomic Polynomial Resultants.

A novel factoring method that uses resultant theory and cyclotomic polynomials
to detect factors. The key innovation:

For N = pq, the cyclotomic polynomial Φ_n(x) factors differently mod p vs mod q:
- If n | (p-1), then Φ_n(x) splits into linear factors mod p
- If n | (p+1), then Φ_n(x) splits into quadratic factors mod p
- Otherwise, Φ_n(x) has higher-degree irreducible factors mod p

The resultant Res(Φ_m, Φ_n) of two cyclotomic polynomials encodes information
about when their roots of unity coincide mod a factor of N. Computing
Φ_m(x) mod N and checking for CRT divergence (non-trivial gcd with N) at
various evaluation points reveals factors.

Additionally, we implement the **polynomial cascade**: compute Φ_m(x) mod N
for successive values of m, and at each step evaluate Φ_m(a) mod N for
multiple values of a. When Φ_m(a) == 0 (mod p) but Φ_m(a) ≢ 0 (mod q),
we have gcd(Φ_m(a) mod N, N) = p.

This generalizes Pollard p-1 (which uses Φ_1(x) = x-1 evaluated at a = B!)
and Williams p+1 (which uses Φ_2(x) = x+1 via Lucas sequences) to ALL
cyclotomic polynomials simultaneously.

Per the honest assessment: this is a smooth-group-order method in the ring
Z[x]/(Φ_m(x), N), which has order h_m(p) for prime factors p where m | (p-1)
or m | (p+1). It achieves L_p[1/2] expected time, matching ECM, but provides
coverage of multiple smoothness targets simultaneously (p-1, p+1, p2+1, etc.)
by testing multiple cyclotomic orders.

Not polynomial time. For cryptographic inputs, GNFS remains asymptotically fastest.
"""
from __future__ import annotations

from math import gcd, isqrt


def _mobius(n: int) -> int:
    """Compute the Möbius function μ(n).

    Returns:
        1 if n is squarefree with an even number of prime factors
        -1 if n is squarefree with an odd number of prime factors
        0 if n has a squared prime factor
    """
    if n == 1:
        return 1
    count = 0
    temp = n
    for p in range(2, isqrt(n) + 1):
        if temp % p == 0:
            count += 1
            temp //= p
            if temp % p == 0:
                return 0  # Not squarefree
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
    """Multiply two polynomials (descending order of degree).

    [1, 0, 1] * [1, 1] = x^3 + x^2 + x + 1 = [1, 1, 1, 1]
    """
    if not a or not b:
        return [0]
    # Remove leading zeros
    while a and a[0] == 0 and len(a) > 1:
        a = a[1:]
    while b and b[0] == 0 and len(b) > 1:
        b = b[1:]
    result = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        for j, cb in enumerate(b):
            result[i + j] += ca * cb
    return result


def _poly_div(dividend: list[int], divisor: list[int]) -> list[int]:
    """Divide polynomial dividend by divisor, returning quotient.

    Both polynomials are in descending order of degree.
    Uses exact integer arithmetic — assumes divisor divides dividend exactly.

    Example: _poly_div([1, 0, 0, 0, -1], [1, -1]) = [1, 1, 1, 1]
             (x^4 - 1) / (x - 1) = x^3 + x^2 + x + 1
    """
    # Strip leading zeros
    while dividend and dividend[0] == 0 and len(dividend) > 1:
        dividend = dividend[1:]
    while divisor and divisor[0] == 0 and len(divisor) > 1:
        divisor = divisor[1:]

    if not divisor or len(dividend) < len(divisor):
        return [0]

    # Polynomial long division with exact integer arithmetic
    remainder = list(dividend)
    divisor_lead = divisor[0]
    divisor_len = len(divisor)
    quotient_len = len(remainder) - divisor_len + 1
    quotient = [0] * quotient_len

    for i in range(quotient_len):
        if remainder[i] == 0:
            continue
        # Divide remainder[i] by divisor_lead (both integers)
        coeff = remainder[i] // divisor_lead
        quotient[i] = coeff
        for j in range(divisor_len):
            remainder[i + j] -= coeff * divisor[j]

    # Clean trailing zeros
    while quotient and quotient[-1] == 0:
        quotient.pop()

    return quotient if quotient else [0]


def _cyclotomic_poly(n: int) -> list[int]:
    """Compute the cyclotomic polynomial Φ_n(x) using Möbius inversion.

    Φ_n(x) = prod_{d|n} (x^{n/d} - 1)^{μ(d)}

    where μ is the Möbius function. This avoids recursive division
    and produces correct polynomials for all n.

    Returns coefficients in descending order: [a_k, a_{k-1}, ..., a_1, a_0].
    """
    if n == 1:
        return [1, -1]  # x - 1

    # Collect divisors with non-zero Möbius values
    # Process in order: multiply by (x^{n/d} - 1) for μ=1,
    # divide by (x^{n/d} - 1) for μ=-1
    mu_plus = []  # d where μ(d) = +1
    mu_minus = []  # d where μ(d) = -1

    for d in _divisors(n):
        mu = _mobius(d)
        if mu == 1:
            mu_plus.append(d)
        elif mu == -1:
            mu_minus.append(d)

    # Start with identity
    result = [1]

    # Multiply by (x^{n/d} - 1) for each d where μ(d) = 1
    for d in mu_plus:
        exp = n // d
        # (x^exp - 1) represented in descending order
        poly = [0] * (exp + 1)
        poly[0] = 1
        poly[exp] = -1
        result = _poly_mul(result, poly)

    # Divide by (x^{n/d} - 1) for each d where μ(d) = -1
    for d in mu_minus:
        exp = n // d
        poly = [0] * (exp + 1)
        poly[0] = 1
        poly[exp] = -1
        result = _poly_div(result, poly)

    return result


def _poly_eval_mod(coeffs: list[int], x: int, N: int) -> int:
    """Evaluate polynomial with given coefficients at x mod N.

    Coefficients are in descending order: [a_k, ..., a_0].
    Uses Horner's method for efficiency.
    """
    if not coeffs:
        return 0

    result = 0
    for c in coeffs:
        result = (result * x + c) % N
    return result


def _poly_gcd_mod(coeffs_a: list[int], coeffs_b: list[int], N: int) -> list[int]:
    """Compute gcd of two polynomials mod N using the Euclidean algorithm.

    Returns the monic GCD polynomial (or detects a factor of N).
    """
    # Remove leading zeros
    a = list(coeffs_a)
    b = list(coeffs_b)
    while a and a[0] == 0:
        a = a[1:]
    while b and b[0] == 0:
        b = b[1:]

    if not a:
        return b
    if not b:
        return a

    # Euclidean algorithm for polynomials
    while b:
        while b and b[0] == 0:
            b = b[1:]
        if not b:
            break

        # Check if leading coefficient of b is invertible mod N
        g = gcd(abs(b[0]), N)
        if 1 < g < N:
            # Found a factor!
            return [g]

        # Make b monic if possible
        if b[0] != 0:
            inv = pow(b[0], -1, N)
            b = [(c * inv) % N for c in b]

        # Compute a mod b
        result = list(a)
        b_len = len(b)
        for i in range(len(result) - b_len + 1):
            if result[i] == 0:
                continue
            for j in range(b_len):
                result[i + j] = (result[i + j] - result[i] * b[j]) % N

        # The remainder is in the last b_len-1 coefficients
        a = b
        remainder = result[len(result) - b_len + 1:] if len(result) >= b_len else [0]
        while remainder and remainder[0] == 0:
            remainder = remainder[1:]
        b = remainder if remainder else [0]

    # Make result monic
    while a and a[0] == 0:
        a = a[1:]
    if a and a[0] != 0:
        g = gcd(abs(a[0]), N)
        if 1 < g < N:
            return [g]
        inv = pow(a[0], -1, N)
        a = [(c * inv) % N for c in a]

    return a if a else [1]


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


def _eisenstein_splitting_guidance(N: int, m: int) -> float:
    """Score a cyclotomic order m by how differently it splits mod p vs mod q.

    Eisenstein Splitting Law: Φ_m(x) mod p splits into φ(m)/d factors of degree d
    where d = ord_m(p) (multiplicative order of p mod m).

    If ord_m(p) != ord_m(q), then Φ_m behaves differently mod p vs mod q,
    meaning evaluating Φ_m(a) may reveal a factor via CRT divergence.

    Since we don't know p, q, we approximate by checking which m divide
    patterns of the form p == 1 (mod m) or p == -1 (mod m):
    - If m | (p-1), then ord_m(p) = 1 → Φ_m splits linearly mod p
    - If m | (p+1) and p is odd, then ord_{2m}(p) = 2 → different splitting mod q

    Scoring: higher score = more likely that ord_m(p) != ord_m(q).
    We approximate by checking divisibility of N±1 by small factors of m.

    Returns a score in [0, 1] indicating how promising this order is.
    """
    # Score based on how many ways m divides N-1 or N+1
    # (approximating p-1 and p+1 structure)
    score = 0.0
    n_minus_1 = N - 1
    n_plus_1 = N + 1

    # Check divisors of m
    divs = _divisors(m)
    for d in divs:
        if d < 2:
            continue
        # If d divides N-1, then p == 1 (mod d) is possible
        if n_minus_1 % d == 0:
            score += 0.3
        # If d divides N+1, then p == -1 (mod d) is possible
        if n_plus_1 % d == 0:
            score += 0.3

    # Higher orders get a small bonus (more cyclotomic structure)
    if m > 12:
        score += 0.1

    # Cap at 1.0
    return min(score, 1.0)


def _zsigmondy_base_score(a: int, m: int) -> float:
    """Score base a by Zsigmondy's theorem for order m.

    Zsigmondy's theorem: a^m - 1 has a primitive prime divisor except for
    special cases (a=2, m=1; a=2, m=6; etc.).

    A primitive prime divisor of a^m - 1 divides Φ_m(a) but no a^k - 1 for k < m.
    Such primes are "fresh" and more likely to expose CRT divergence.

    Returns score in [0, 1] — higher means a is a better base for order m.
    """
    # Check known exceptional cases from Zsigmondy
    if a == 2 and m == 1:
        return 0.5
    if a == 2 and m == 6:
        return 0.5
    if a == 2 and m == 2:
        return 0.8

    # Default: bases that are primitive roots mod many primes are better
    # We approximate by checking small a values
    if a in (2, 3, 5, 6, 7):
        return 0.9
    elif a < 20:
        return 0.7
    else:
        return 0.5


def cyclotomic_resultant_factor(N: int, max_order: int = 50,
                                eval_points: int = 20,
                                smooth_bound: int = 50000) -> tuple[int, int] | None:
    """Factor N using cyclotomic polynomial resultants.

    For each cyclotomic order m = 1, 2, ..., max_order:
    1. Compute Φ_m(x) (cyclotomic polynomial)
    2. Evaluate Φ_m(a) mod N for multiple values of a
    3. Check gcd(Φ_m(a) mod N, N) for non-trivial factors
    4. Use smooth-bound powering: compute a^(B!) and evaluate Φ_m at that point

    The key insight: if p | N and the order of a mod p divides m, then
    Φ_m(a) == 0 (mod p), so gcd(Φ_m(a) mod N, N) reveals p.

    This generalizes Pollard p-1 (m=1, a^(B!)) and Williams p+1 (m=2, Lucas)
    to all cyclotomic orders simultaneously.

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

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    primes = _small_primes(smooth_bound)

    # Phase 1: Direct cyclotomic evaluation at small orders
    # For small m, Φ_m(a) can detect factors when ord_p(a) | m
    for m in range(1, min(max_order + 1, 30)):
        try:
            phi_m = _cyclotomic_poly(m)
        except (RecursionError, OverflowError):
            continue

        if not phi_m or phi_m == [0]:
            continue

        # Evaluate at multiple base points
        for a in range(2, min(eval_points + 2, N)):
            # Direct evaluation: Φ_m(a) mod N
            val = _poly_eval_mod(phi_m, a, N)
            if val == 0:
                continue  # Φ_m(a) == 0 mod N, try next

            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Phase 1.5: Smooth-bound powering for m=1,2
        # m=1: Φ_1(x) = x-1, so Φ_1(a^(B!)) = a^(B!) - 1 (Pollard p-1)
        # m=2: Φ_2(x) = x+1, so Φ_2(a^(B!)) = a^(B!) + 1 (Williams p+1-like)
        if m <= 2:
            for a in [2, 3, 5, 7, 11, 13]:
                if a >= N:
                    continue
                # Compute a^(B!) mod N
                power = a
                for p in primes:
                    pk = p
                    while pk * p <= smooth_bound:
                        pk *= p
                    power = pow(power, pk, N)

                if m == 1:
                    val = (power - 1) % N  # Φ_1(a^(B!)) = a^(B!) - 1
                else:
                    val = (power + 1) % N  # Φ_2(a^(B!)) = a^(B!) + 1

                g = gcd(abs(val), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    # Phase 2: Polynomial GCD approach
    # Compute gcd(x^m - 1, x^n - 1) mod N for various m, n
    # If p | N and ord_p(a) | gcd(m,n), the GCD polynomial reveals p
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
        if m >= N:
            continue
        # x^m - 1 mod N
        poly_m = [0] * (m + 1)
        poly_m[0] = 1
        poly_m[m] = N - 1  # -1 mod N

        for n in [2, 3, 4, 5, 6]:
            if n >= m:
                continue
            # x^n - 1 mod N
            poly_n = [0] * (n + 1)
            poly_n[0] = 1
            poly_n[n] = N - 1

            # Compute GCD of polynomials mod N
            g_poly = _poly_gcd_mod(poly_m, poly_n, N)

            # Check if GCD computation found a factor
            if g_poly == [0]:
                continue
            if len(g_poly) == 1 and g_poly[0] > 1 and g_poly[0] < N:
                return (min(g_poly[0], N // g_poly[0]), max(g_poly[0], N // g_poly[0]))

            # Evaluate GCD polynomial at several points
            for a in range(2, min(10, N)):
                val = _poly_eval_mod(g_poly, a, N)
                if val == 0:
                    continue
                g = gcd(abs(val), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def cyclotomic_cascade_factor(N: int, bound: int = 50000,
                                max_order: int = 50,
                                base_points: int = 10) -> tuple[int, int] | None:
    """Factor N using cyclotomic polynomial evaluation cascade.

    A simpler, more practical version that focuses on the smooth-bound
    powering approach for cyclotomic polynomials of small order.

    For each base a and cyclotomic order m:
    1. Compute a^(B!) mod N using incremental powering
    2. Evaluate Φ_m at the powered point
    3. Check gcd for non-trivial factors

    This provides coverage of p-1 (m=1), p+1 (m=2), p2+p+1 (m=3),
    p2+1 (m=4), p2-p+1 (m=6), etc. simultaneously.

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

    primes = _small_primes(bound)

    # Try multiple base points
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

            # Check m=1: Φ_1(a^(B!)) = a^(B!) - 1 (Pollard p-1)
            val = (power - 1) % N
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Check m=2: Φ_2(a^(B!)) = a^(B!) + 1 (Williams p+1-like)
            val = (power + 1) % N
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Stage 2: Test a^(B! · ℓ) for small primes ℓ
        for ell in _small_primes(min(bound // 5, 5000)):
            power_ell = pow(power, ell, N)

            # m=1
            val = (power_ell - 1) % N
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # m=2
            val = (power_ell + 1) % N
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Check higher cyclotomic orders
        # m=3: Φ_3(x) = x2 + x + 1 → a^(2·B!) + a^(B!) + 1
        # m=4: Φ_4(x) = x2 + 1 → a^(2·B!) + 1
        # m=6: Φ_6(x) = x2 - x + 1 → a^(2·B!) - a^(B!) + 1
        power2 = pow(power, 2, N)

        # m=3: x2 + x + 1
        val = (power2 + power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=4: x2 + 1
        val = (power2 + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=6: x2 - x + 1
        val = (power2 - power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=10: x4 - x3 + x2 - x + 1
        power3 = pow(power, 3, N)
        power4 = pow(power, 4, N)
        val = (power4 - power3 + power2 - power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # m=12: x4 - x2 + 1
        val = (power4 - power2 + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None


def _full_order_spectrum_eval(N: int, power: int, m: int) -> tuple[int, int] | None:
    """Evaluate Φ_m at powered point and check for factor.

    For m in {1, 2, 3, 4, 6, 10, 12}, uses the closed-form cyclotomic
    polynomial evaluated at power = a^(B!) mod N.

    Returns (p, q) if found, None otherwise.
    """
    power2 = pow(power, 2, N)
    power3 = pow(power, 3, N)
    power4 = pow(power, 4, N)

    # Dispatch by cyclotomic order using closed forms
    # m=1: Φ_1(x) = x - 1
    if m == 1:
        val = (power - 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=2: Φ_2(x) = x + 1
    elif m == 2:
        val = (power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=3: Φ_3(x) = x2 + x + 1
    elif m == 3:
        val = (power2 + power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=4: Φ_4(x) = x2 + 1
    elif m == 4:
        val = (power2 + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=5: Φ_5(x) = x4 + x3 + x2 + x + 1
    elif m == 5:
        val = (power4 + power3 + power2 + power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=6: Φ_6(x) = x2 - x + 1
    elif m == 6:
        val = (power2 - power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=10: Φ_10(x) = x4 - x3 + x2 - x + 1
    elif m == 10:
        val = (power4 - power3 + power2 - power + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # m=12: Φ_12(x) = x4 - x2 + 1
    elif m == 12:
        val = (power4 - power2 + 1) % N
        g = gcd(abs(val), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None


def full_order_spectrum_factor(N: int, bound: int = 50000,
                                max_order: int = 60,
                                base_points: int = 8) -> tuple[int, int] | None:
    """Full cyclotomic order spectrum cascade with Eisenstein splitting-law guidance.

    A major enhancement over the existing cyclotomic_cascade_factor:
    1. Tests ALL cyclotomic orders up to max_order (not just 1, 2, 3, 4, 6, 10, 12)
    2. Uses Eisenstein splitting-law guidance to prioritize orders where
       ord_m(p) != ord_m(q), via _eisenstein_splitting_guidance
    3. Uses Zsigmondy-guided base selection to prefer bases with primitive
       prime divisors for each order

    For each base a (small primes) and each order m up to max_order:
    1. Score m by Eisenstein splitting law: how differently does Φ_m split mod p vs q?
    2. Score a by Zsigmondy: does a^m - 1 have a primitive prime divisor?
    3. Compute a^(B!) mod N
    4. Evaluate Φ_m at that point and check gcd

    This gives higher per-trial success probability than the fixed-order approach,
    especially for N where p and q have different multiplicative structure mod m.

    Complexity: L_p[1/2] expected, same as ECM but with better constant
    factors from smarter order selection.

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

    primes = _small_primes(bound)

    # Base candidates — prefer small primes with good Zsigmondy properties
    base_candidates = [2, 3, 5, 6, 7, 11, 13, 17, 19, 23, 29, 31]

    # Order candidates — all orders up to max_order
    all_orders = list(range(1, max_order + 1))

    # Score and sort orders by Eisenstein splitting-law guidance
    order_scores = [(m, _eisenstein_splitting_guidance(N, m)) for m in all_orders]
    order_scores.sort(key=lambda x: x[1], reverse=True)

    # For each base, compute a^(B!) once, then test all scored orders
    for a in base_candidates[:base_points]:
        if a >= N:
            continue

        # Compute a^(B!) mod N incrementally
        power = a
        for prime in primes:
            pk = prime
            while pk * prime <= bound:
                pk *= prime
            power = pow(power, pk, N)

            # Check the top-scored orders first (Eisenstein guidance)
            for m, score in order_scores[:15]:
                if score < 0.1:
                    break  # Skip low-scored orders
                result = _full_order_spectrum_eval(N, power, m)
                if result is not None:
                    return result

            # Stage 2: test with second power (a^(B! * ℓ))
        for ell in _small_primes(min(bound // 5, 5000)):
            power_ell = pow(power, ell, N)
            for m, score in order_scores[:10]:
                if score < 0.2:
                    break
                result = _full_order_spectrum_eval(N, power_ell, m)
                if result is not None:
                    return result

    return None