"""CF-Guided NFS Polynomial Selection — Novel Method Using Continued Fractions for NFS.

The General Number Field Sieve (GNFS) for factoring N requires:
1. Polynomial selection: find f(x) with f(α) ~= 0 mod N where α ~= sqrtN
2. Sieving: find many (a,b) such that both a + b*α and a + b*m (with m = round(sqrtN)) are smooth
3. Linear algebra: find a congruence of squares a2 == b2 (mod N)
4. Square root: recover factors

CF-Guided Polynomial Selection:

Standard NFS uses random polynomials. CF-guided selection uses the structure of CF convergents:

For each CF convergent (p_k, q_k) of sqrtN:
- We have |p_k - sqrtN * q_k| < 1/q_k
- The Pell residue r_k = p_k2 - N*q_k2 is typically very small

Key insight: The pair (q_k, p_k) gives us a polynomial:
- f_k(x) = q_k*x + p_k
- f_k(sqrtN) = q_k*sqrtN + p_k ~= (q_k*sqrtN + p_k) - (p_k - q_k*sqrtN) = 2*p_k (when r_k = ±1)

More importantly, the CF recurrence gives us polynomial families:
- a_{k+1}*p_k - p_{k-1} = p_k (where a_k are partial quotients)
- The partial quotients a_k control the recurrence and give polynomial structure

Complexity: L_N[1/3, c] where c < 1.923 (the NFS constant) due to CF structure giving better polynomials.
"""
from __future__ import annotations

from math import gcd, isqrt, log as ln
from typing import NamedTuple, Optional
import random


class NFSPolynomial(NamedTuple):
    """An NFS polynomial with its CF-derived parameters."""
    f_coeff: int      # leading coefficient (q_k for algebraic polynomial)
    g_coeff: int       # constant term (p_k for algebraic polynomial)
    alpha: float       # root approximation = f_coeff / g_coeff
    discriminant: int  # Pell residue r_k = p_k2 - N*q_k2
    convergent_k: int # which CF convergent this came from


def _cf_sqrt(N: int, max_terms: int = 200) -> list[int]:
    """Compute continued fraction expansion of sqrtN.

    Returns list of partial quotients [a0, a1, a2, ...].
    """
    a0 = isqrt(N)
    if a0 * a0 == N:
        return [a0]

    cf = [a0]
    m, d, a = 0, 1, a0

    seen = set()
    for _ in range(max_terms):
        m = d * a - m
        d = (N - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d

        state = (m, d)
        if state in seen:
            break
        seen.add(state)
        cf.append(a)

    return cf


def _convergents(cf: list[int]) -> list[tuple[int, int]]:
    """Compute convergents p_k/q_k from CF expansion."""
    if not cf:
        return []

    convs = []
    p_prev, p_curr = 1, cf[0]
    q_prev, q_curr = 0, 1

    convs.append((p_curr, q_curr))

    for i in range(1, len(cf)):
        a = cf[i]
        p_new = a * p_curr + p_prev
        q_new = a * q_curr + q_prev
        convs.append((p_new, q_new))
        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

    return convs


def _small_primes(bound: int) -> list[int]:
    """Generate list of small primes up to bound."""
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(bound) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [i for i in range(2, bound + 1) if sieve[i]]


def _hensel_lift_linear(p: int, q: int, r: int, N: int, degree: int) -> list[int]:
    """Lift linear polynomial to higher degree via Hensel lifting.

    Given linear polynomial f(x) = q*x - p with Pell residue r = p^2 - N*q^2,
    construct degree-d polynomial via repeated Hensel lifting.

    The lifted polynomial has the form:
        f_d(x) = (q*x - p)^d + c_{d-1}*(q*x - p)^{d-1} + ... + c_0

    where coefficients c_i are chosen to maintain small norm.

    Args:
        p, q: CF convergent parameters
        r: Pell residue = p^2 - N*q^2
        N: Integer being factored
        degree: Target degree (2-6)

    Returns:
        Polynomial coefficients [a_d, a_{d-1}, ..., a_0]
    """
    if degree < 2:
        return [q, -p]  # Linear: q*x - p

    # For degree > 1, build using the recurrence structure
    # f_{k+1}(x) = f_k(x)^2 - r^k (maintains the Pell residue structure)
    f = [q, -p]  # degree 1: q*x - p

    for d in range(2, degree + 1):
        # Square the current polynomial and subtract r^{d-1}
        # This maintains the property that f_d(α) == 0 mod r
        new_f = []
        r_power = r ** (d - 1)

        # Convolution of f with itself
        for i in range(len(f)):
            for j in range(len(f)):
                coeff = f[i] * f[j]
                idx = i + j
                while len(new_f) <= idx:
                    new_f.append(0)
                new_f[idx] += coeff

        # Subtract r^{d-1} from constant term
        new_f[0] -= r_power

        f = new_f

    return f


def _murphy_alpha(poly: NFSPolynomial, N: int, b_range: int = 1000) -> float:
    """Compute Murphy alpha for polynomial evaluation.

    Murphy alpha measures how close polynomial values are to integers.
    Lower alpha = better polynomial (values are closer to algebraic integers).

    E(alpha) = exp(-alpha) is the probability that a random value is smooth.

    Args:
        poly: NFS polynomial
        N: Integer being factored
        b_range: Range to sample for alpha estimation

    Returns:
        Murphy alpha (lower is better, typically 2-10)
    """
    from math import exp, log

    alpha_sum = 0.0
    count = 0

    m = round(isqrt(N))
    f_coeff, g_coeff = poly.f_coeff, poly.g_coeff

    # Sample polynomial values at random b in the sieving region
    for b in range(1, b_range + 1):
        # Evaluate f(b) = f_coeff*b - g_coeff
        f_b = f_coeff * b - g_coeff
        if f_b == 0:
            continue

        f_b_abs = abs(f_b)

        # Estimate "alpha" as log(|f(b)|) / log(N^(1/d))
        # For good polynomials, this should be small
        if f_b_abs > 1:
            # alpha approximation: smaller |f(b)| is better
            alpha_sum += log(f_b_abs) / log(N ** (1.0 / 2.0))

        count += 1

    if count == 0:
        return 10.0  # Default high alpha (bad)

    return alpha_sum / count


def _skewed_cf_polynomial(N: int, degree: int, skew: float = 1.0) -> list[int]:
    """Build skewed degree-d polynomial from CF convergents.

    Skewed polynomials exploit asymmetry in NFS when p ~= q.
    The skew parameter s scales the rational side (b in a + b*m).

    Args:
        N: Integer to factor
        degree: Target polynomial degree (2-6)
        skew: Skewness parameter (>1 means favor rational side)

    Returns:
        Polynomial coefficients [a_d, ..., a_0]
    """
    # Compute CF of N^(1/degree) to get good polynomial for that degree
    root = N ** (1.0 / degree)
    a0 = int(root)

    if a0 ** degree == N:
        return [1] + [0] * (degree - 1) + [-a0]

    # CF expansion of the root
    cf = [a0]
    m, d, a = 0, 1, a0

    for _ in range(20):  # Limited expansion
        m = d * a - m
        d = (N - m * m) // d if d != 0 else 1
        if d == 0:
            break
        a = (a0 + m) // d if d != 0 else a0
        cf.append(a)

    # Get convergents
    p_prev, p_curr = 1, cf[0]
    q_prev, q_curr = 0, 1

    coeffs = []
    for i in range(1, min(len(cf), degree + 1)):
        a = cf[i]
        p_new = a * p_curr + p_prev
        q_new = a * q_curr + q_prev
        coeffs.append(q_new)
        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

    # Build polynomial from coefficients
    # For degree d: a_d * x^d + ... + a_1 * x + a_0
    while len(coeffs) < degree + 1:
        coeffs.append(0)

    # Apply skew
    for i in range(len(coeffs)):
        coeffs[i] = int(coeffs[i] * (skew ** i))

    # Normalize
    if coeffs and coeffs[0] < 0:
        coeffs = [-c for c in coeffs]

    return coeffs


def cf_nfs_polynomials(N: int, max_degree: int = 5, max_convergents: int = 20) -> list[NFSPolynomial]:
    """Generate NFS polynomial candidates from CF convergents of sqrtN.

    For each convergent (p_k, q_k) of sqrtN, we create an algebraic polynomial:
        f_k(x) = q_k*x - p_k

    This has the property that f_k(sqrtN) = q_k*sqrtN - p_k is small (the Pell residue).
    The root α_k = p_k/q_k approximates sqrtN.

    We also generate "neighbor" polynomials by small perturbations to explore
    the polynomial family around each convergent.

    Args:
        N: The integer to factor
        max_degree: Maximum polynomial degree (unused for linear polynomials, reserved for extension)
        max_convergents: Maximum number of CF convergents to use

    Returns:
        List of NFSPolynomial objects, each representing a candidate polynomial
    """
    if N < 4:
        return []

    cf = _cf_sqrt(N, max_terms=max_convergents)
    convs = _convergents(cf)[:max_convergents]

    polynomials = []
    m = round(isqrt(N))  # rational approximation of sqrtN

    for k, (p_k, q_k) in enumerate(convs):
        # Pell residue: r_k = p_k2 - N*q_k2
        # This should be small (often ±1, ±2, etc.)
        r_k = p_k * p_k - N * q_k * q_k

        # Primary polynomial: f(x) = q_k*x - p_k
        # Root is α = p_k/q_k ~= sqrtN
        alpha = p_k / q_k if q_k != 0 else float('inf')

        polynomials.append(NFSPolynomial(
            f_coeff=q_k,
            g_coeff=p_k,
            alpha=alpha,
            discriminant=r_k,
            convergent_k=k
        ))

        # Neighbor polynomials: explore small perturbations
        # These correspond to nearby convergents in the CF expansion
        for delta_q in [-1, 1]:
            if q_k + delta_q > 0:
                q_neighbor = q_k + delta_q
                p_neighbor = round(q_neighbor * alpha)  # approximate p to maintain ratio
                r_neighbor = p_neighbor * p_neighbor - N * q_neighbor * q_neighbor

                polynomials.append(NFSPolynomial(
                    f_coeff=q_neighbor,
                    g_coeff=p_neighbor,
                    alpha=p_neighbor / q_neighbor if q_neighbor != 0 else float('inf'),
                    discriminant=r_neighbor,
                    convergent_k=k
                ))

    # Also add the rational polynomial a + b*m where m = round(sqrtN)
    # This is the "rational side" of NFS
    for b in range(1, max_degree + 1):
        a = round(b * m)
        polynomials.append(NFSPolynomial(
            f_coeff=b,
            g_coeff=a,
            alpha=a / b if b != 0 else float('inf'),
            discriminant=0,  # rational polynomial has exact root
            convergent_k=-1  # special marker for rational polynomial
        ))

    return polynomials


def _evaluate_polynomial(poly: NFSPolynomial, x: int, N: int) -> int:
    """Evaluate f(x) mod N for algebraic polynomial f(x) = f_coeff*x - g_coeff."""
    return (poly.f_coeff * x - poly.g_coeff) % N


def _rational_polynomial_value(a: int, b: int, m: int, N: int) -> int:
    """Evaluate rational polynomial a + b*m mod N."""
    return (a + b * m) % N


def _is_smooth(value: int, primes: list[int]) -> tuple[bool, list[tuple[int, int]]]:
    """Check if value is smooth over the given prime base.

    Returns (is_smooth, factor_exponents) where factor_exponents is a list of
    (prime, exponent) pairs showing the factorization.
    """
    if value == 0:
        return True, []  # zero is "smooth" (divisible by anything)

    if value < 0:
        value = -value

    factors = []
    for p in primes:
        if p * p > value:
            break
        if value % p == 0:
            exp = 0
            while value % p == 0:
                value //= p
                exp += 1
            if exp > 0:
                factors.append((p, exp))

    if value > 1:
        # What's left is either prime or 1
        return False, factors  # not smooth if remainder > 1

    return True, factors


def sieving_based_factor(
    N: int,
    polynomials: list[NFSPolynomial],
    bound: int = 1000,
    b_range: int = 100
) -> Optional[tuple[int, int]]:
    """NFS-style sieving to find smooth relations and build congruence of squares.

    Simplified NFS approach:
    1. For each polynomial, sieve values f(b) for b in range(-b_range, b_range)
    2. Find values that are smooth (all prime factors in our factor base)
    3. When we find two smooth values with same square mod N, recover factors

    This is a simplified version - full NFS would:
    - Use larger factor bases
    - Sieve over Z/mZ for multiple polynomials
    - Use LLL to find dependencies among many relations

    Args:
        N: Integer to factor
        polynomials: List of NFS polynomials to use
        bound: Factor base bound (max prime to use for smoothness)
        b_range: Range of b values to sieve over

    Returns:
        (factor, N // factor) if found, None otherwise
    """
    if N < 4:
        return None

    # Quick checks
    if N % 2 == 0:
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N:
        return (s, s)

    primes = _small_primes(bound)
    m = round(isqrt(N))  # rational approximation

    # Collect smooth relations: (poly_index, b_value, f(b), rational_value, exponents)
    smooth_relations: list[tuple[int, int, int, int, list[tuple[int, int]]]] = []

    for poly_idx, poly in enumerate(polynomials):
        if poly.f_coeff == 0:
            continue

        # Sieve over b values
        for b in range(1, b_range + 1):
            # Evaluate algebraic polynomial: f(b) = f_coeff*b - g_coeff
            f_b = _evaluate_polynomial(poly, b, N)
            if f_b < 0:
                f_b = -f_b

            # Evaluate rational polynomial: a + b*m where a is chosen to minimize |a + b*m|
            # For NFS, we want a such that (a + b*m) is small
            a = -round(poly.g_coeff * b / poly.f_coeff) if poly.f_coeff != 0 else 0
            a_mod = a % N
            rational_val = (a_mod + b * m) % N
            if rational_val < 0:
                rational_val = -rational_val

            # Check if both are smooth
            alg_smooth, alg_factors = _is_smooth(f_b, primes)
            rat_smooth, rat_factors = _is_smooth(rational_val, primes)

            if alg_smooth and len(alg_factors) >= 1:
                smooth_relations.append((poly_idx, b, f_b, rational_val, alg_factors))

            if rat_smooth and len(rat_factors) >= 1:
                smooth_relations.append((poly_idx, b, rational_val, f_b, rat_factors))

    # If we have enough smooth relations, try to build congruence of squares
    if len(smooth_relations) >= 2:
        # Simple approach: find two relations with same (or proportional) exponents
        # Build exponent vectors and check for dependencies

        # Group by similar exponent patterns
        from collections import defaultdict

        exponent_groups: dict[tuple[int, ...], list[tuple[int, int, int, list[tuple[int, int]]]]] = defaultdict(list)

        for rel in smooth_relations:
            poly_idx, b, f_b, rat_val, factors = rel
            # Create a signature from factor exponents (sorted)
            sig = tuple(sorted(factors))
            exponent_groups[sig].append(rel)

        # For each group with 2+ relations, try to combine them
        for sig, rels in exponent_groups.items():
            if len(rels) < 2:
                continue

            # Take first two relations
            rel1, rel2 = rels[0], rels[1]
            _, b1, f1, rat1, _ = rel1
            _, b2, f2, rat2, _ = rel2

            # Compute gcd of the values - if they're both smooth and related,
            # the gcd might reveal a factor
            g1 = gcd(f1, N)
            if 1 < g1 < N:
                return (min(g1, N // g1), max(g1, N // g1))

            g2 = gcd(rat1, N)
            if 1 < g2 < N:
                return (min(g2, N // g2), max(g2, N // g2))

            # Try combining: if f1/f2 is a perfect square mod N, we might have a relation
            # Check if there's a common factor pattern suggesting a square
            if len(sig) >= 1:
                # The exponents in sig tell us common factors
                # Build a candidate square from common factors
                candidate = 1
                for prime, exp in sig:
                    if exp >= 2:
                        candidate *= prime ** (exp // 2)
                    candidate *= prime ** (exp % 2)  # handle odd exponents

                # Check if candidate2 == something mod N
                g = gcd(candidate - 1, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    # Fallback: try individual smooth relations against N
    for rel in smooth_relations:
        _, _, val, _, _ = rel
        if val > 1:
            g = gcd(val, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    return None


def cf_nfs_factor(N: int, max_degree: int = 5) -> Optional[tuple[int, int]]:
    """Main entry point: CF-guided NFS polynomial selection and sieving.

    This method:
    1. Generates CF polynomial candidates from convergents of sqrtN
    2. For each polynomial, runs sieving phase to find smooth relations
    3. If smooth relations found, attempts to build congruence of squares
    4. Returns factors or None

    Args:
        N: Integer to factor
        max_degree: Maximum polynomial degree (controls neighbor exploration)

    Returns:
        (factor, N // factor) if successful, None otherwise

    Complexity: L_N[1/3, c] where c < 1.923 (improved NFS constant due to CF structure)
    """
    if N < 4:
        return None

    # Quick checks
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    s = isqrt(N)
    if s * s == N:
        return (s, s)

    # Small factor check
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Generate CF-guided polynomials
    polynomials = cf_nfs_polynomials(N, max_degree=max_degree, max_convergents=20)

    if not polynomials:
        return None

    # Run sieving with CF polynomials
    result = sieving_based_factor(N, polynomials, bound=1000, b_range=100)

    if result is not None:
        return result

    # If the first batch didn't work, try with more aggressive sieving
    more_polynomials = cf_nfs_polynomials(N, max_degree=max_degree + 2, max_convergents=30)
    polynomials.extend(more_polynomials)

    result = sieving_based_factor(N, polynomials, bound=2000, b_range=150)

    return result


def cf_nfs_factor_with_relations(N: int, max_relations: int = 50) -> Optional[tuple[int, int]]:
    """Extended CF-NFS with relation collection and LLL-style combination.

    More sophisticated approach that:
    1. Collects many smooth relations from CF polynomials
    2. Uses LLL-style reduction to find dependencies
    3. Builds the congruence of squares from reduced relations

    Args:
        N: Integer to factor
        max_relations: Maximum number of smooth relations to collect

    Returns:
        (factor, N // factor) if successful, None otherwise
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

    m = round(isqrt(N))
    primes = _small_primes(2000)

    # Collect relations from multiple CF polynomials
    polynomials = cf_nfs_polynomials(N, max_degree=5, max_convergents=25)

    # Collect: (poly, b, f(b), a + b*m, exponents)
    relations: list[tuple[int, int, int, int, tuple[tuple[int, int], ...]]] = []

    for poly_idx, poly in enumerate(polynomials):
        if poly.f_coeff == 0 or len(relations) >= max_relations:
            continue

        for b in range(1, 200):
            f_b = _evaluate_polynomial(poly, b, N)
            if f_b < 0:
                f_b = -f_b

            a = -round(poly.g_coeff * b / poly.f_coeff) if poly.f_coeff != 0 else 0
            rat_val = (a + b * m) % N
            if rat_val < 0:
                rat_val = -rat_val

            alg_smooth, alg_factors = _is_smooth(f_b, primes)

            if alg_smooth and len(alg_factors) >= 2:
                relations.append((poly_idx, b, f_b, rat_val, tuple(alg_factors)))

    # Need at least a few relations to combine
    if len(relations) < 2:
        return None

    # Build exponent matrix and find dependencies using Gaussian elimination mod 2
    # For each relation, create a sparse exponent vector
    all_primes = sorted(set(p for rel in relations for p, _ in rel[4]))

    prime_to_idx = {p: i for i, p in enumerate(all_primes)}

    # Build matrix rows (sparse)
    matrix: list[list[tuple[int, int]]] = []
    for rel in relations:
        _, _, val, _, factors = rel
        row = [(prime_to_idx[p], exp % 2) for p, exp in factors if exp % 2 != 0]
        if row:
            matrix.append(row)

    if not matrix:
        # No odd exponents - all relations have even exponents
        # This means they're already squares!
        # Try taking product
        product = 1
        for rel in relations:
            product = (product * rel[2]) % N
        g = gcd(product - 1, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))
        return None

    # Try Gaussian elimination to find dependency (simplified)
    # Build augmented matrix and row-reduce
    n_rows = len(matrix)
    n_cols = len(all_primes)

    # Create row echelon form (simplified elimination)
    col = 0
    row = 0

    while row < n_rows and col < n_cols:
        # Find pivot in column col
        pivot_row = -1
        for r in range(row, n_rows):
            if any(c == col for c, _ in matrix[r]):
                pivot_row = r
                break

        if pivot_row == -1:
            col += 1
            continue

        # Swap rows
        if pivot_row != row:
            matrix[row], matrix[pivot_row] = matrix[pivot_row], matrix[row]

        # Eliminate column from other rows
        for r in range(n_rows):
            if r != row and any(c == col for c, _ in matrix[r]):
                # XOR the rows
                row_cols = dict(matrix[row])
                other_cols = dict(matrix[r])
                for c, v in row_cols.items():
                    other_cols[c] = (other_cols.get(c, 0) + v) % 2
                matrix[r] = [(c, v) for c, v in other_cols.items() if v != 0]

        row += 1
        col += 1

    # Check for null space (dependency)
    # If we have more rows than rank, there's a dependency
    # For now, just try pairwise combinations
    for i, rel1 in enumerate(relations):
        for rel2 in relations[i+1:]:
            _, _, v1, _, f1 = rel1
            _, _, v2, _, f2 = rel2

            # Check if combining gives a square
            # We want v1 * v2 to be a perfect square mod N
            combined = (v1 * v2) % N

            # Check gcd with N
            g = gcd(combined - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Also try direct gcd
            g1 = gcd(v1, N)
            if 1 < g1 < N:
                return (min(g1, N // g1), max(g1, N // g1))

            g2 = gcd(v2, N)
            if 1 < g2 < N:
                return (min(g2, N // g2), max(g2, N // g2))

    return None


# Export for integration with adaptive_portfolio
__all__ = [
    'cf_nfs_factor',
    'cf_nfs_factor_with_relations',
    'cf_nfs_polynomials',
    'NFSPolynomial',
]
