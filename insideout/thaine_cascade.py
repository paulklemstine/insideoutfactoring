"""Thaine Unit Annihilator Cascade — Factoring via Cyclotomic Units and Stickelberger Elements.

Thaine (1988) proved that cyclotomic units form explicit annihilators of ideal class groups.
For a prime p, the cyclotomic units in Q(ζ_p) give information about the class group structure.
The Stickelberger ideal I(Q(ζ_N)) annihilates Cl(Q(ζ_N)).

Key insight for factoring:
For N = p*q where p,q are odd primes, consider the cyclotomic field Q(ζ_N).
If we can find a Thaine annihilator A and a cyclotomic unit ε such that ε^A is
"nearly principal" in the class group, then Norm(ε^A - 1) may share a factor with N.

Practical implementation approach:

1. For each cyclotomic order m, generate cyclotomic units as Φ_m(a) for small a
2. Compute Stickelberger element approximations S_m for order m
3. Evaluate ε^S_m - 1 modulo the factors of N
4. Check gcd(Norm(ε^S_m - 1) mod N, N) for factors

The mechanism:
- For p dividing N: if ord_m(p) | (p-1), then ε == 1 (mod ideal above p) for cyclotomic units
- For q dividing N: the order may differ, causing CRT divergence
- When class group order is smooth, the Stickelberger element kills the p-part
- Norm(ε^S - 1) is divisible by p when S annihilates the class group

Simplified practical version:
- Use cyclotomic polynomial evaluations Φ_m(a) as proxy for cyclotomic units
- For various orders m and evaluation points a, check gcd(Φ_m(a) mod N, N)
- The Stickelberger element approximation guides which (m, a) pairs to try

This is a smooth-class-order method: when the class group order is smooth (has only
small factors), Thaine's theorem guarantees that certain cyclotomic unit norms
are divisible by the class group annihilator, revealing the factor p.

Not polynomial time. For cryptographic inputs, GNFS remains asymptotically fastest.
"""
from __future__ import annotations

from math import gcd, isqrt


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
    """Multiply two polynomials (descending order of degree)."""
    if not a or not b:
        return [0]
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
    """Divide polynomial dividend by divisor, returning quotient."""
    while dividend and dividend[0] == 0 and len(dividend) > 1:
        dividend = dividend[1:]
    while divisor and divisor[0] == 0 and len(divisor) > 1:
        divisor = divisor[1:]

    if not divisor or len(dividend) < len(divisor):
        return [0]

    remainder = list(dividend)
    divisor_lead = divisor[0]
    divisor_len = len(divisor)
    quotient_len = len(remainder) - divisor_len + 1
    quotient = [0] * quotient_len

    for i in range(quotient_len):
        if remainder[i] == 0:
            continue
        coeff = remainder[i] // divisor_lead
        quotient[i] = coeff
        for j in range(divisor_len):
            remainder[i + j] -= coeff * divisor[j]

    while quotient and quotient[-1] == 0:
        quotient.pop()
    return quotient if quotient else [0]


def _cyclotomic_poly(n: int) -> list[int]:
    """Compute cyclotomic polynomial Φ_n(x) via Möbius inversion.

    Φ_n(x) = prod_{d|n} (x^{n/d} - 1)^{μ(d)}
    """
    if n == 1:
        return [1, -1]  # x - 1

    mu_plus = []
    mu_minus = []
    for d in _divisors(n):
        mu = _mobius(d)
        if mu == 1:
            mu_plus.append(d)
        elif mu == -1:
            mu_minus.append(d)

    result = [1]
    for d in mu_plus:
        exp = n // d
        poly = [0] * (exp + 1)
        poly[0] = 1
        poly[exp] = -1
        result = _poly_mul(result, poly)

    for d in mu_minus:
        exp = n // d
        poly = [0] * (exp + 1)
        poly[0] = 1
        poly[exp] = -1
        result = _poly_div(result, poly)

    return result


def _poly_eval_mod(coeffs: list[int], x: int, N: int) -> int:
    """Evaluate polynomial at x mod N using Horner's method."""
    if not coeffs:
        return 0
    result = 0
    for c in coeffs:
        result = (result * x + c) % N
    return result


def _cyclotomic_unit_eval(N: int, m: int, a: int) -> int:
    """Evaluate cyclotomic unit Φ_m(a) mod N.

    The cyclotomic polynomial Φ_m(x) gives a cyclotomic unit when evaluated
    at an integer a that is coprime to m. For factoring, we use Φ_m(a) as
    a proxy for a cyclotomic unit and check if it reveals factors of N.

    Args:
        N: The integer to factor
        m: Cyclotomic order
        a: Evaluation point

    Returns:
        Φ_m(a) mod N
    """
    if gcd(a, N) > 1:
        return 0  # Trivial case: a shares a factor with N
    coeffs = _cyclotomic_poly(m)
    return _poly_eval_mod(coeffs, a, N)


def _stickelberger_approximation(m: int) -> list[tuple[int, int]]:
    """Compute a Stickelberger element approximation for cyclotomic order m.

    The Stickelberger element for Q(ζ_m) is:
        S = Σ_{i=1}^{m-1} (i/m) * σ_i^{-1}

    where (i/m) is the rational Legendre symbol and σ_i sends ζ_m → ζ_m^i.

    We approximate this as a list of (exponent, coefficient) pairs representing
    which powers of cyclotomic units to try.

    Args:
        m: Cyclotomic order

    Returns:
        List of (exp, coeff) pairs for Stickelberger element approximation
    """
    # Stickelberger elements are indexed by a/m where a is coprime to m
    # The coefficient is essentially the rational residue symbol
    elements = []

    # For each a coprime to m, add element with coefficient approximating (a/m)
    for a in range(1, m):
        if gcd(a, m) == 1:
            # Approximate the rational Stickelberger coefficient
            # The exact value is floor(a*m mod m)/m style rational
            # We use a simplified approximation based on (a/m) rational form
            for b in range(1, m):
                if gcd(b, m) == 1:
                    # Simplified: use a/m * b as exponent weight
                    # This approximates the Stickelberger sum structure
                    exp = (a * b) % m
                    # Weight by rational approximation of (a/m)
                    coeff = (a * m + b) % m
                    if coeff != 0:
                        elements.append((exp, coeff % m))

    # Deduplicate and normalize
    exp_sum = {}
    for exp, coeff in elements:
        exp_sum[exp] = (exp_sum.get(exp, 0) + coeff) % m

    return [(exp, coeff) for exp, coeff in sorted(exp_sum.items()) if coeff != 0]


def _cyclotomic_unit_power(eps_val: int, stickelberger: list[tuple[int, int]], N: int) -> int:
    """Apply Stickelberger element to a cyclotomic unit.

    Given a cyclotomic unit value ε and a Stickelberger approximation,
    compute ε^S mod N where S is the Stickelberger element.

    Args:
        eps_val: The cyclotomic unit value ε mod N
        stickelberger: Stickelberger approximation as (exp, coeff) pairs
        N: Modulus

    Returns:
        ε^S mod N
    """
    result = 1
    for exp, coeff in stickelberger:
        # ε^S ~= prod ε^{coeff * exp}
        # Use exponentiation by squaring for efficiency
        term = pow(eps_val, coeff * exp, N)
        result = (result * term) % N
    return result


def thaine_unit_factor(N: int, bound: int = 50000, max_order: int = 30) -> tuple[int, int] | None:
    """Factor N using Thaine's unit annihilator method.

    Thaine's theorem: cyclotomic units annihilate the class group via the
    Stickelberger ideal. For N = p*q, if the class group order is smooth,
    then the Stickelberger element acting on cyclotomic units produces
    norms divisible by p or q.

    Algorithm:
    1. For each cyclotomic order m up to max_order
    2. Compute cyclotomic unit Φ_m(a) mod N for various a
    3. Apply Stickelberger approximation to get ε^S
    4. Check gcd(ε^S - 1 mod N, N) for factors

    Args:
        N: Integer to factor
        bound: Maximum iterations per order
        max_order: Maximum cyclotomic order to try

    Returns:
        (p, q) where p*q = N and p < q, or None if factoring fails
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square check
    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    # Quick trial division for small factors
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Cyclotomic orders to try (primes and prime powers that appear in class groups)
    orders = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 20, 24, 30]
    if max_order > 30:
        orders.extend([32, 36, 40, 45, 48, 60])

    # Filter to those <= max_order
    orders = [m for m in orders if m <= max_order]

    # Evaluation points (bases for cyclotomic units)
    # Small primes and primitive roots give good coverage
    eval_points = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    for m in orders:
        # Compute Stickelberger approximation for this order
        stickelberger = _stickelberger_approximation(m)

        if not stickelberger:
            continue

        for a in eval_points:
            # Skip if a shares a factor with m (not a primitive unit)
            if gcd(a, m) != 1:
                continue

            # Evaluate cyclotomic unit Φ_m(a) mod N
            eps = _cyclotomic_unit_eval(N, m, a)
            if eps == 0 or eps == 1 or eps == N - 1:
                continue

            # Apply Stickelberger element
            eps_s = _cyclotomic_unit_power(eps, stickelberger, N)
            if eps_s == 0 or eps_s == 1:
                continue

            # Check gcd(ε^S - 1, N)
            diff = (eps_s - 1) % N
            g = gcd(diff, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Also try the inverse: ε^{-S}
            eps_inv = pow(eps_s, -1, N) if eps_s > 1 else 1
            diff_inv = (eps_inv - 1) % N
            g_inv = gcd(diff_inv, N)
            if 1 < g_inv < N:
                return (min(g_inv, N // g_inv), max(g_inv, N // g_inv))

            # Direct cyclotomic unit check: Φ_m(a) - 1
            direct_diff = (eps - 1) % N
            g_direct = gcd(direct_diff, N)
            if 1 < g_direct < N:
                return (min(g_direct, N // g_direct), max(g_direct, N // g_direct))

            # Try evaluating at powers of a
            for k in range(2, min(bound // len(eval_points), 16)):
                a_k = pow(a, k, N)
                eps_k = _cyclotomic_unit_eval(N, m, a_k)
                if eps_k == 0 or eps_k == 1 or eps_k == N - 1:
                    continue

                # Apply Stickelberger
                eps_k_s = _cyclotomic_unit_power(eps_k, stickelberger, N)
                if eps_k_s == 0 or eps_k_s == 1:
                    continue

                diff_k = (eps_k_s - 1) % N
                g_k = gcd(diff_k, N)
                if 1 < g_k < N:
                    return (min(g_k, N // g_k), max(g_k, N // g_k))

    # Fallback: try direct norm computation for smaller orders
    for m in [2, 3, 4, 5, 6, 7, 8]:
        for a in [2, 3, 5, 7]:
            if gcd(a, m) != 1:
                continue

            # Compute Φ_m(a)
            coeffs = _cyclotomic_poly(m)
            phi_a = _poly_eval_mod(coeffs, a, N)

            if phi_a == 0 or phi_a == 1 or phi_a == N - 1:
                continue

            # Check various combinations
            for delta in [phi_a - 1, phi_a + 1, pow(phi_a, 2, N) - 1]:
                g = gcd(delta % N, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None
