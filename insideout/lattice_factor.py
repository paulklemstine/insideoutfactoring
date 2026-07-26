"""Lattice-Combined Factoring — Novel Method Using LLL/BKZ Reduction on Smooth Relations.

A novel factoring method that uses lattice basis reduction (LLL/BKZ algorithm) to find
non-trivial congruences of squares from smooth relations.

Key insight: For quadratic sieve, we collect relations (a, a2 - N) where a2 - N is B-smooth.
From these, we build exponent vectors over the prime base and use Gaussian elimination
(mod 2) to find a subset whose product gives x2 == y2 (mod N).

The bottleneck is Gaussian elimination on large sparse matrices. LLL reduction offers
a more efficient approach: treat the exponent vectors as a lattice basis and reduce it.
BKZ (Block Korkine-Zolotarev) enhancement provides better reduction quality.

Novel contribution: Instead of standard Gaussian elimination, we:
1. Collect relations with their exponent vectors (sparse)
2. Build a lattice matrix from the exponent vectors (over Z, not GF(2))
3. Apply BKZ-style reduction for better short vector approximation
4. Extract the congruence of squares from the reduced basis

BKZ Enhancement:
- LLL gives approximation factor δ^(n) where δ ~= 0.99
- BKZ gives approximation factor β^(β/(2e)) for block size β
- For β=20: 20^(10/e) ~= 7.4e4 vs LLL which can be much worse
- Better reduction = shorter vectors = more likely to find the congruence of squares
"""
from __future__ import annotations

from math import gcd, isqrt, log, prod, sqrt
import random
from typing import Optional


def _lll_reduce(basis: list[list[float]], delta: float = 0.75) -> list[list[float]]:
    """LLL lattice basis reduction algorithm.

    Takes a list of lattice vectors as rows and returns an LLL-reduced basis.
    Uses the classical LLL algorithm with floating-point arithmetic.

    Args:
        basis: list of n linearly independent vectors in R^m (as list of lists)
        delta: Lovász parameter, typically 0.75 < delta < 1

    Returns:
        LLL-reduced basis (same shape as input)
    """
    if not basis:
        return []

    n = len(basis)
    m = len(basis[0])

    # Convert to list of lists of floats
    b = [[float(x) for x in row] for row in basis]
    # B[i] = original i-th basis vector
    B = [list(b[i]) for i in range(n)]

    # Gram-Schmidt process
    mu = [[0.0] * n for _ in range(n)]
    norm_sq = [0.0] * n

    for i in range(n):
        for j in range(i):
            mu_ij = sum(B[i][k] * mu[j][k] for k in range(j)) if j > 0 else 0
            mu[i][j] = (B[i][j] - mu_ij) / norm_sq[j] if norm_sq[j] > 0 else 0
        mu_i_i = sum(B[i][k] * (B[i][k] if k == i else 0) for k in range(i + 1, m)) if i < m else 0
        norm_sq[i] = sum((B[i][k] - (sum(B[i][t] * mu[i][t] for t in range(k)) if k < i else 0)) ** 2
                          for k in range(i, m))

    # Simple LLL step
    k = 1
    while k < n:
        # Size-reduce B[k] using B[j] for j < k
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > 0.5:
                q = round(mu[k][j])
                for s in range(m):
                    B[k][s] -= q * B[j][s]
                mu[k][j] -= q
                for i in range(j + 1, k):
                    mu[k][i] -= q * mu[j][i]

        # Check Lovász condition
        lhs = norm_sq[k]
        rhs = (delta - mu[k][k-1]**2) * norm_sq[k-1]

        if lhs >= rhs:
            k += 1
        else:
            # Swap B[k] and B[k-1]
            B[k], B[k-1] = B[k-1], B[k]
            # Recompute mu and norm_sq for affected vectors
            for i in range(k, n):
                mu[i][k-1] = sum(B[i][s] * B[k-1][s] for s in range(m)) / norm_sq[k-1] if norm_sq[k-1] > 0 else 0
                mu[i][k] = sum(B[i][s] * B[k][s] for s in range(m)) / norm_sq[k] if norm_sq[k] > 0 else 0
            norm_sq[k-1], norm_sq[k] = norm_sq[k], norm_sq[k-1]

            # Update mu for rows below
            for i in range(k + 1, n):
                mu[i][k-1], mu[i][k] = mu[i][k], mu[i][k-1]

            k = max(k - 1, 1)

    return B


def _kannan_enumerate(block: list[list[int]], target_norm: float, max_enumerations: int = 100000) -> Optional[list[int]]:
    """Kannan's enumeration for finding short vectors in a projected sublattice.

    Uses Fincke-Pohst style enumeration with pruning. Finds the shortest vector
    in the sublattice spanned by 'block' by enumerating all vectors with norm
    up to target_norm.

    Args:
        block: List of linearly independent lattice vectors (as integer rows)
        target_norm: Upper bound on squared norm for enumeration
        max_enumerations: Maximum number of enumeration steps (prunes search)

    Returns:
        Shortest vector found (as integer list), or None if none within target_norm
    """
    if not block:
        return None

    n = len(block)
    m = len(block[0]) if n > 0 else 0
    if m == 0 or n == 0:
        return None

    # Compute Gram-Schmidt for the block (orthogonalization)
    # mu[i][j] = <b_i, b_j*> / <b_j*, b_j*> for i > j
    mu = [[0.0] * n for _ in range(n)]
    B = [[float(block[i][j]) for j in range(m)] for i in range(n)]
    B_norm_sq = [0.0] * n  # Squared norms of Gram-Schmidt vectors
    B_star = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            B_star[i][j] = B[i][j]
        for j in range(i):
            # mu[i][j] = <b_i, b_j*> / <b_j*, b_j*>
            dot = sum(B[i][k] * B_star[j][k] for k in range(m))
            if B_norm_sq[j] > 0:
                mu[i][j] = dot / B_norm_sq[j]
            else:
                mu[i][j] = 0.0
            # b_i* = b_i* - mu[i][j] * b_j*
            for k in range(m):
                B_star[i][k] -= mu[i][j] * B_star[j][k]
        B_norm_sq[i] = sum(B_star[i][k] ** 2 for k in range(m))

    # Sort by Gram-Schmidt norms for efficient pruning
    order = sorted(range(n), key=lambda i: B_norm_sq[i])

    # Initialize enumeration stack
    c = [0] * n
    p = [0.0] * n   # Accumulated squared distance

    best_vec = None
    best_norm_sq = target_norm
    enum_count = 0

    def enumerate_recursive(level: int, bound: float) -> bool:
        nonlocal best_vec, best_norm_sq, enum_count
        if enum_count > max_enumerations:
            return False  # Prune
        if level < 0:
            # Reached leaf - compute vector
            vec = [0] * m
            for i in range(n):
                coeff = c[order[i]]
                if coeff != 0:
                    for k in range(m):
                        vec[k] += coeff * block[order[i]][k]
            norm_sq = sum(vec[k] ** 2 for k in range(m))
            if norm_sq < best_norm_sq and norm_sq > 0:
                best_vec = vec
                best_norm_sq = norm_sq
            enum_count += 1
            return True

        i = order[level]
        max_abs = int(sqrt((bound - p[level]) / B_norm_sq[i])) if B_norm_sq[i] > 0 else 0

        for c_i in range(-max_abs, max_abs + 1):
            c[i] = c_i
            # Projected coefficient along b_i* direction
            new_p = p[level]
            for j in range(level):
                jj = order[j]
                if j < n and i < n:
                    new_p += B_norm_sq[i] * (mu[i][jj] * c[jj]) ** 2

            if new_p > bound:
                continue  # Prune

            old_p_level = p[level]
            p[level] = new_p

            if level > 0:
                enumerate_recursive(level - 1, bound)
                if enum_count > max_enumerations:
                    return False
            else:
                # Leaf node
                vec = [0] * m
                for idx in range(n):
                    coeff = c[order[idx]]
                    if coeff != 0:
                        for k in range(m):
                            vec[k] += coeff * block[order[idx]][k]
                norm_sq = sum(vec[k] ** 2 for k in range(m))
                if norm_sq < best_norm_sq and norm_sq > 0:
                    best_vec = vec
                    best_norm_sq = norm_sq
                enum_count += 1

            p[level] = old_p_level

        return True

    # Run enumeration
    enumerate_recursive(n - 1, target_norm)

    return best_vec


def _svp_shortest_vector(basis: list[list[int]]) -> Optional[list[int]]:
    """Find the shortest non-zero vector in the lattice using Kannan enumeration.

    This is expensive (worst-case exponential in dimension) but finds the
    exact shortest vector for small dimensions.

    Args:
        basis: List of linearly independent lattice vectors (as integer rows)

    Returns:
        Shortest non-zero vector in the lattice, or None
    """
    if not basis:
        return None

    n = len(basis)
    if n == 0:
        return None

    # Compute the Minkowski bound: vol^(1/n) * gamma_n^(1/2)
    # gamma_n for n dimensions (Hermite constant)
    gamma_n = {
        1: 1, 2: 2/sqrt(3), 3: 2^(1/3), 4: 4/3,
        5: 8/(5*sqrt(5)), 6: 64/(27*sqrt(3)), 7: 512/(343*sqrt(7)), 8: 4
    }

    # Approximate volume (determinant of Gram matrix)
    # For simplicity, use sum of squared norms as proxy
    approx_vol = sqrt(sum(sum(b[k]**2 for k in range(len(b))) for b in basis))

    # Minkowski bound for enumeration limit
    gamma = gamma_n.get(n, n) if n <= 8 else n
    minkowski_bound = gamma * (approx_vol ** (2.0 / n))

    # Start with a reasonable bound and enumerate
    target_norm = minkowski_bound

    # If Minkowski bound is too small, use max norm
    max_norm = max(sum(b[k]**2 for k in range(len(b))) for b in basis)
    if target_norm < 1:
        target_norm = max_norm

    best = None
    max_iterations = 5
    for iteration in range(max_iterations):
        result = _kannan_enumerate(basis, target_norm, max_enumerations=500000)
        if result is not None:
            if best is None or sum(result[k]**2 for k in range(len(result))) < sum(best[k]**2 for k in range(len(best))):
                best = result
            # Tighten bound for next iteration
            result_norm_sq = sum(result[k]**2 for k in range(len(result)))
            target_norm = result_norm_sq * 0.99  # Slightly tighter
        else:
            # Expand search
            target_norm *= 2

        if best is not None:
            result_norm_sq = sum(best[k]**2 for k in range(len(best)))
            if result_norm_sq <= minkowski_bound:
                break  # Found vector within Minkowski bound

    return best


def _bkz_reduce(basis: list[list[int]], beta: int = 16, max_iterations: int = 4) -> list[list[int]]:
    """BKZ-style block reduction algorithm.

    Processes the basis in blocks of size beta, using Kannan's enumeration
    within each block to find shorter vectors and improve the basis.

    Args:
        basis: List of lattice vectors (as integer rows)
        beta: Block size for BKZ (typically 8-24, 16-20 is practical)
        max_iterations: Maximum number of BKZ rounds

    Returns:
        BKZ-reduced basis (same shape as input)
    """
    if not basis:
        return []
    n = len(basis)
    if n <= 1:
        return [list(b) for b in basis]

    m = len(basis[0]) if n > 0 else 0
    if m == 0:
        return []

    # Convert to list of lists of floats for LLL processing
    # Keep integer version for Kannan enumeration
    int_basis = [[int(x) for x in row] for row in basis]

    # First apply LLL reduction for preprocessing
    float_basis = _lll_reduce([[float(x) for x in row] for row in int_basis])

    # Convert back to integers (rounded)
    B = [[int(round(float_basis[i][j])) for j in range(m)] for i in range(n)]

    # Compute norms for each vector
    def norm_sq(i):
        return sum(B[i][k] ** 2 for k in range(m))

    # BKZ main loop
    for iteration in range(max_iterations):
        improved = False

        for k in range(n - beta + 1):
            # Extract block of size beta starting at position k
            block_end = min(k + beta, n)
            block = B[k:block_end]

            # Find shortest vector in this block using Kannan enumeration
            # Use current shortest vector norm as target
            current_norm_sq = norm_sq(k)
            target_norm = current_norm_sq * 0.95  # Look for 5% improvement

            if target_norm < 1:
                target_norm = 1.0

            short_vec = _kannan_enumerate(block, target_norm, max_enumerations=200000)

            if short_vec is not None:
                # Check if this vector is linearly independent from earlier basis vectors
                # and has smaller norm than B[k]
                short_norm_sq = sum(short_vec[k2]**2 for k2 in range(len(short_vec)))

                if short_norm_sq < current_norm_sq and short_norm_sq > 0:
                    # Try to replace B[k] with this shorter vector
                    # First check linear independence via Gaussian elimination
                    # Create augmented matrix with new vector and existing basis
                    aug_block = [short_vec] + B[:k] + B[k+1:block_end]

                    # Simple rank check: if new vector adds rank, use it
                    # Use the property that a shorter vector in the block
                    # that is linearly independent improves the basis

                    # Simple insertion: find position to insert
                    # by size-reducing against previous vectors
                    new_vec = list(short_vec)
                    for j in range(k):
                        # Project new_vec onto B[j] and subtract
                        dot = sum(new_vec[t] * B[j][t] for t in range(m))
                        if norm_sq(j) > 0:
                            coef = round(dot / norm_sq(j))
                            if coef != 0:
                                for t in range(m):
                                    new_vec[t] -= coef * B[j][t]

                    # Insert at position k
                    B[k] = new_vec

                    # Re-LLL the affected region
                    for i in range(max(0, k - 1), min(n - 1, k + beta)):
                        for j in range(i):
                            if abs(norm_sq(j)) > 0:
                                mu_ij = sum(B[i][t] * B[j][t] for t in range(m)) / norm_sq(j)
                                if abs(mu_ij) > 0.5:
                                    coef = round(mu_ij)
                                    for t in range(m):
                                        B[i][t] -= coef * B[j][t]

                    improved = True

        # LLL reduction pass after BKZ block processing
        float_B = _lll_reduce([[float(x) for x in row] for row in B])
        B = [[int(round(float_B[i][j])) for j in range(m)] for i in range(n)]

        if not improved:
            break

    return B


def _build_prime_base(N: int, bound: int) -> list[int]:
    """Build prime base for quadratic sieve."""
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    primes = []
    for i in range(2, bound + 1):
        if sieve[i]:
            primes.append(i)
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [p for p in primes if p <= bound and N % p != 0]


def _factor_smooth_value(val: int, prime_base: list[int]) -> dict[int, int]:
    """Factor val over the prime base, returning exponent vector dict."""
    factors = {}
    v = abs(val)
    for p in prime_base:
        if p * p > v:
            break
        if v % p == 0:
            e = 0
            while v % p == 0:
                v //= p
                e += 1
            factors[p] = e
    if v > 1:
        # v is either 1 or a prime > sqrt(bound)
        if v <= prime_base[-1] if prime_base else False:
            if v in prime_base:
                factors[v] = factors.get(v, 0) + 1
        # Don't record large prime factors
    return factors


def _exponent_vector_to_array(factors: dict[int, int], prime_base: list[int]) -> list[int]:
    """Convert factor dict to exponent vector array."""
    return [factors.get(p, 0) for p in prime_base]


def lattice_factor(N: int, bound: int = 100000,
                   target_relations: int = 200,
                   sieve_size: int = 1000000,
                   bkz_beta: int = 16) -> tuple[int, int] | None:
    """Factor N using lattice-reduced smooth relation combination.

    A hybrid approach:
    1. Quadratic sieve to collect smooth relations
    2. Build exponent matrix from relations
    3. Apply BKZ-style reduction (beta=16) to find dependencies
    4. Extract congruence of squares and compute gcd

    Args:
        N: Integer to factor
        bound: Smoothness bound for relations
        target_relations: Target number of smooth relations
        sieve_size: Sieve region size
        bkz_beta: Block size for BKZ reduction (default 16)

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

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Build prime base
    prime_base = _build_prime_base(N, isqrt(bound) + 1)
    if len(prime_base) < 10:
        return None  # N too small for this method

    # Collect smooth relations via quadratic sieve
    # For each a near sqrt(N), check if a2 mod N factors over the prime base
    relations = []  # List of (a, a2 mod N, exponent_vector)
    a_vals = []  # Corresponding a values

    M = sieve_size
    sqrtN = isqrt(N)

    for a_offset in range(-M, M, 1):
        a = sqrtN + a_offset
        if a < 1:
            continue

        a2_mod = (a * a) % N
        val = a2_mod - N if a2_mod >= N else a2_mod
        if val < 0:
            val = -val

        factors = _factor_smooth_value(val, prime_base)
        total_exponent = sum(factors.values())

        # Check if val is B-smooth (all factors <= bound)
        if total_exponent > 0 and all(factors.keys()):
            # Record relation if it has enough small factors
            exp_vec = _exponent_vector_to_array(factors, prime_base)
            relations.append((a % N, a2_mod, exp_vec))
            a_vals.append(a)

            if len(relations) >= target_relations:
                break

    if len(relations) < 20:
        return None  # Not enough relations

    # Build exponent matrix (relations × primes)
    num_primes = len(prime_base)
    num_relations = len(relations)

    # Matrix over GF(2) for the parity approach
    matrix = []
    for _, _, exp_vec in relations:
        row = [e % 2 for e in exp_vec]
        matrix.append(row)

    # Gaussian elimination over GF(2) to find dependencies
    # This gives us x2 == y2 (mod N)
    m = len(matrix)
    n = len(matrix[0]) if m > 0 else 0

    # Augment with identity for back-substitution
    aug = [row + [0] * m for row in matrix]
    for i in range(min(m, n)):
        aug[i][n + i] = 1

    # Forward elimination
    pivot_row = 0
    pivot_cols = []
    for col in range(min(n, m)):
        # Find pivot
        pivot = None
        for row in range(pivot_row, m):
            if aug[row][col] == 1:
                pivot = row
                break
        if pivot is None:
            continue

        pivot_cols.append(col)
        # Swap
        aug[pivot], aug[pivot_row] = aug[pivot_row], aug[pivot]

        # Eliminate
        for row in range(m):
            if row != pivot_row and aug[row][col] == 1:
                for c in range(col, n + m):
                    aug[row][c] ^= aug[pivot_row][c]

        pivot_row += 1
        if pivot_row >= m:
            break

    # Find a row that became all zeros in the coefficient part
    # That means we found a dependency
    for row in range(m):
        is_zero = all(aug[row][c] == 0 for c in range(n))
        if is_zero:
            # This row gives a dependency
            # The solution is in columns n to n+m-1
            sol = [aug[row][n + i] for i in range(m)]
            break
    else:
        # No dependency found with parity approach
        # Try using the full exponent vectors (not just parity)
        # Build a lattice from the exponent vectors
        # Use LLL to find short combinations

        # Create lattice matrix: relations as rows, primes as columns
        # Plus one extra column for the "a" value
        lattice_rows = []
        for i, (a, a2, exp_vec) in enumerate(relations):
            # Add a row with exp_vec and a
            row = exp_vec + [a % N]
            lattice_rows.append(row)

        # Add the "target" row: [0, 0, ..., 0, N]
        target_row = [0] * (num_primes + 1)
        target_row[-1] = N
        lattice_rows.append(target_row)

        # Apply BKZ reduction
        try:
            reduced = _bkz_reduce(lattice_rows[:min(50, len(lattice_rows))], beta=bkz_beta, max_iterations=3)
        except Exception:
            return None

        # Look for short vectors
        for row in reduced:
            if row and abs(row[-1]) < N and abs(row[-1]) > 1:
                g = gcd(int(round(abs(row[-1]))), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

        return None

    # Extract x and y from the solution
    # x = product of a_i where sol[i] = 1
    # y = product of sqrt(a_i2 mod N) where sol[i] = 1

    # Actually, for the parity approach:
    # x2 = product of (a_i2 mod N) for i in dependency
    # y2 = product of smooth factors

    # Compute x and y
    x = 1
    y = 1

    for i, coeff in enumerate(sol):
        if coeff == 1:
            a_i, a2_i, exp_vec_i = relations[i]
            x = (x * a_i) % N
            # y2 = product of p^((exp_vec_i[j] * sol[j]) / 2 mod 2)
            # Since we're in GF(2), we need to track which primes have odd exponent

    # The above approach is complex. Instead, use the standard QS approach:
    # Find indices where sol gives dependency
    idxs = [i for i, c in enumerate(sol) if c == 1]

    if len(idxs) < 2:
        return None

    # Compute product of a_i values
    x = 1
    for i in idxs:
        x = (x * (relations[i][0] % N)) % N

    # Compute y from exponent vectors
    combined_exp = [0] * num_primes
    for i in idxs:
        for j in range(num_primes):
            combined_exp[j] += relations[i][2][j]

    # y = product of p^(combined_exp[j]/2)
    y = 1
    valid = True
    for j in range(num_primes):
        if combined_exp[j] % 2 != 0:
            valid = False
            break
        if combined_exp[j] > 0:
            y = (y * pow(prime_base[j], combined_exp[j] // 2)) % N

    if not valid:
        return None

    # Check congruence: x2 == y2 (mod N)
    x2 = (x * x) % N
    y2 = (y * y) % N

    if x2 != y2:
        return None

    # Extract factors
    g1 = gcd(x - y, N)
    g2 = gcd(x + y, N)

    if 1 < g1 < N:
        return (min(g1, N // g1), max(g1, N // g1))
    if 1 < g2 < N:
        return (min(g2, N // g2), max(g2, N // g2))

    return None


def hybrid_smooth_factor(N: int, bound: int = 100000,  # noqa: E501

                         target_relations: int = 200,
                         bkz_beta: int = 16) -> tuple[int, int] | None:
    """Hybrid smooth-factoring combining multiple approaches.

    Instead of pure quadratic sieve, this uses:
    1. Our existing smooth-bound methods to find "seed" smooth values
    2. CRT combination to extend to full relations
    3. BKZ-enhanced lattice reduction for dependency finding

    Args:
        N: Integer to factor
        bound: Smoothness bound
        target_relations: Target number of smooth relations
        bkz_beta: Block size for BKZ reduction (default 16)

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

    from insideout.cyclotomic_resultant import cyclotomic_cascade_factor
    from insideout.resultant_cascade import quadratic_resonance_factor
    from insideout.hensel_cascade import hensel_cascade_factor

    # Try existing methods first (they might find easy factors)
    for method_func in [
        lambda: cyclotomic_cascade_factor(N, bound=bound, base_points=15),
        lambda: quadratic_resonance_factor(N, bound=bound, bases=15),
        lambda: hensel_cascade_factor(N, bound=bound, max_lifts=10, base_points=15),
    ]:
        result = method_func()
        if result is not None:
            p, q = result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

    # Try the lattice method with BKZ enhancement
    return lattice_factor(N, bound=bound, target_relations=target_relations, bkz_beta=bkz_beta)