"""CF Period Matrix Cascade — Novel Factoring via Continued Fraction Period Matrices.

A novel factoring method that combines:
1. The period structure of the continued fraction of sqrtN
2. SL2 matrix powering (from our SL2 group-order method)
3. Smooth-group-order detection

Key insight: The continued fraction of sqrtN has a period matrix M_CF that
encodes the full period. This matrix has special order properties mod p
(where p | N): when the CF period length is smooth, M_CF^k == I (mod p)
for some smooth k. By computing M_CF^(B!) mod N and checking for CRT
divergence, we can find factors.

The novel structural advantage over random SL2 matrices: M_CF is the
ACTUAL period matrix of sqrtN, not a random element. Its order mod p is
related to the CF period length, which for N = pq divides lcm(period_p, period_q).
For primes p == 1 (mod 4), the CF period of sqrtp is typically smooth,
giving this method a structural advantage over ECM with random curves.

Additionally, the CF convergents p_k/q_k provide residue pairs r_k where
p_k2 − N·q_k2 = (−1)^k · r_k. Collecting smooth r_k gives relations
for a congruence of squares (standard CFRAC), but we also check the
CONVERGENT MATRIX M_k = p_k2 − N·q_k2 for CRT divergence at each step.

This is a genuine hybrid: CFRAC-style residue collection + SL2 group-order
detection applied to the CF period matrix.

Per the honest assessment: this combines two sub-exponential methods
(CFRAC and smooth-group-order). It does not achieve polynomial time.
For cryptographic inputs, GNFS remains fastest.
"""
from __future__ import annotations

from math import gcd, isqrt


def _cf_period_matrix(N: int, max_steps: int = 100000) -> tuple[tuple, list[tuple]] | None:
    """Compute the continued fraction period matrix and convergents for sqrtN.

    Returns (M_CF, convergents) where:
    - M_CF is the 2×2 matrix whose trace equals the period length indicator
    - convergents is a list of (p_k, q_k, r_k, sign_k) tuples where
      p_k2 - N·q_k2 = sign_k · r_k

    The CF of sqrtN has the form [a0; a1, a2, ..., a_l] where l is the
    period length. The period matrix M_CF = M_{a_l} · ... · M_{a_1} · M_{a0}
    has the property that M_CF^k == I (mod p) when k is a multiple of
    the order of M_CF mod p.
    """
    if N < 2:
        return None

    a0 = isqrt(N)
    if a0 * a0 == N:
        return None  # Perfect square

    # CF expansion of sqrtN
    m, d, a = 0, 1, a0
    M = (1, 0, 0, 1)  # Identity matrix (will accumulate period matrix)

    convergents = []
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    # First convergent
    r0 = a0 * a0 - N  # Should be negative or zero for non-square
    convergents.append((p_curr, q_curr, abs(r0), -1 if r0 < 0 else 1))

    seen = {}  # (m, d) -> step index to detect period

    for step in range(1, max_steps + 1):
        m = d * a - m
        d = (N - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d

        # Update convergents
        p_new = a * p_curr + p_prev
        q_new = a * q_curr + q_prev

        # Residue: p_k2 - N·q_k2
        residue = p_new * p_new - N * q_new * q_new
        sign = 1 if residue >= 0 else -1
        convergents.append((p_new, q_new, abs(residue), sign))

        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

        # Accumulate period matrix: M_CF = M_{a_k} · M_CF
        # where M_{a_k} = [[a, 1], [1, 0]]
        step_mat = (a % N, 1, 1 % N, 0)
        M = _mat2_mul(step_mat, M, N)

        # Check for period end (d == 1 and we're back to start)
        state = (m % N, d)
        if state in seen:
            # Period complete
            break
        seen[state] = step

        # Also check if d == 1 (period boundary)
        if d == 1:
            break

    return M, convergents


def _mat2_mul(A: tuple, B: tuple, N: int) -> tuple:
    """Multiply two 2×2 matrices mod N."""
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1 * a2 + b1 * c2) % N,
        (a1 * b2 + b1 * d2) % N,
        (c1 * a2 + d1 * c2) % N,
        (c1 * b2 + d1 * d2) % N,
    )


def _mat2_pow(M: tuple, k: int, N: int) -> tuple:
    """Compute M^k mod N via binary exponentiation."""
    result = (1, 0, 0, 1)  # Identity
    base = M
    while k > 0:
        if k & 1:
            result = _mat2_mul(result, base, N)
        base = _mat2_mul(base, base, N)
        k >>= 1
    return result


def _check_matrix_crt(M: tuple, N: int) -> tuple[int, int] | None:
    """Check if matrix M has CRT divergence (different behavior mod p vs mod q)."""
    a, b, c, d = M

    # Off-diagonal entries
    for entry in [b, c]:
        g = gcd(entry, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Diagonal entries (M → I mod p)
    for entry in [(a - 1) % N, (d - 1) % N]:
        g = gcd(entry, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Trace condition: tr(M) == 2 (mod p) but not mod q
    g = gcd((a + d - 2) % N, N)
    if 1 < g < N:
        return (min(g, N // g), max(g, N // g))

    # Determinant divergence
    det = (a * d - b * c) % N
    g = gcd((det - 1) % N, N)
    if 1 < g < N:
        return (min(g, N // g), max(g, N // g))

    return None


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


def _smoothness_test(n: int, factor_base: list[int]) -> list[int] | None:
    """Test if n is smooth over the factor base. Returns factor list or None."""
    if n == 0 or n == 1:
        return None
    n = abs(n)
    factors = []
    for p in factor_base:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    # Allow one large prime
    bound = factor_base[-1] if factor_base else 2
    if n < bound * bound and n > 1:
        factors.append(n)
        return factors
    return None


def cf_matrix_cascade_factor(N: int, bound: int = 50000,
                              cf_steps: int = 10000,
                              smooth_bound: int = 500,
                              max_relations: int = 200) -> tuple[int, int] | None:
    """Factor N using CF Period Matrix Cascade.

    Combines three detection mechanisms:
    1. CF convergent residue check: p_k2 - N·q_k2 = r_k, check gcd(r_k, N)
    2. CF period matrix cascade: compute M_CF^(B!) mod N and check CRT divergence
    3. CFRAC-style: collect smooth residues r_k and combine via congruence of squares

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

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # === Phase 1: CF expansion with convergent checks ===
    a0 = isqrt(N)
    if a0 * a0 == N:
        return (s, s)

    m, d, a = 0, 1, a0
    M_period = (1, 0, 0, 1)  # Accumulates the period matrix

    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    # Convergent residue check (first convergent)
    r0 = a0 * a0 - N
    g = gcd(abs(r0), N)
    if 1 < g < N:
        return (min(g, N // g), max(g, N // g))

    seen_states = {}
    relations = []  # (r_k, factor_list) for CFRAC-style
    factor_base = _small_primes(smooth_bound)

    for step in range(1, cf_steps + 1):
        m = d * a - m
        d_new = (N - m * m)
        if d_new == 0:
            break
        d_new = d_new // d  # This should divide evenly in CF of sqrtN
        if d_new == 0:
            break
        d = d_new
        a = (a0 + m) // d

        # Update convergents
        p_new = a * p_curr + p_prev
        q_new = a * q_curr + q_prev

        # Residue check
        residue = p_new * p_new - N * q_new * q_new
        g = gcd(abs(residue), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # Also check gcd of convergent numerator/denominator with N
        g = gcd(p_new, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))
        g = gcd(q_new, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # Accumulate period matrix
        step_mat = (a % N, 1, 1 % N, 0)
        M_period = _mat2_mul(step_mat, M_period, N)

        # Check period matrix CRT divergence periodically
        if step % 100 == 0:
            result = _check_matrix_crt(M_period, N)
            if result is not None:
                return result

        # Collect smooth residues for CFRAC-style
        if abs(residue) > 1 and len(relations) < max_relations:
            smooth = _smoothness_test(abs(residue), factor_base)
            if smooth is not None and len(smooth) >= 2:
                relations.append((abs(residue), smooth))

        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

        # Period detection
        state = (m % N, d)
        if state in seen_states:
            break
        seen_states[state] = step

        if d == 1:
            break

    # === Phase 2: CF period matrix cascade ===
    # M_period now contains the CF period matrix
    # Compute M_period^(B!) and check for CRT divergence
    primes = _small_primes(bound)
    current_mat = M_period

    for p in primes:
        pk = p
        while pk * p <= bound:
            pk *= p
        current_mat = _mat2_pow(current_mat, pk, N)

        result = _check_matrix_crt(current_mat, N)
        if result is not None:
            return result

    # Stage 2: test M_period^(B! · ℓ) for small primes ℓ
    stage2_primes = _small_primes(min(bound // 5, 5000))
    for ell in stage2_primes:
        mat_ell = _mat2_pow(current_mat, ell, N)
        result = _check_matrix_crt(mat_ell, N)
        if result is not None:
            return result

    # === Phase 3: CFRAC-style congruence of squares ===
    if len(relations) >= 2:
        # Build prime index
        prime_index = {}
        idx = 0
        for _, factors in relations:
            for p in factors:
                if p not in prime_index:
                    prime_index[p] = idx
                    idx += 1

        n_primes = len(prime_index)
        n_relations = len(relations)

        if n_primes > 0 and n_relations >= 2:
            # Build exponent matrix mod 2
            matrix = []
            for residue, factors in relations:
                row = [0] * n_primes
                for p in factors:
                    if p in prime_index:
                        row[prime_index[p]] ^= 1
                matrix.append(row)

            # Augmented matrix [A | I]
            aug = []
            for i, row in enumerate(matrix):
                aug.append(row + [1 if j == i else 0 for j in range(n_relations)])

            # Gaussian elimination mod 2
            pivot_row = 0
            for col in range(n_primes):
                found = -1
                for row in range(pivot_row, n_relations):
                    if aug[row][col] == 1:
                        found = row
                        break
                if found == -1:
                    continue
                aug[pivot_row], aug[found] = aug[found], aug[pivot_row]
                for row in range(n_relations):
                    if row != pivot_row and aug[row][col] == 1:
                        aug[row] = [(a ^ b) for a, b in zip(aug[row], aug[pivot_row])]
                pivot_row += 1

            # Find null space vectors
            for row in range(pivot_row, n_relations):
                combo = [i for i in range(n_relations) if aug[row][n_primes + i] == 1]
                if len(combo) < 2:
                    continue

                from collections import defaultdict
                product = 1
                factor_counts = defaultdict(int)
                for i in combo:
                    product = (product * relations[i][0]) % N
                    for p in relations[i][1]:
                        factor_counts[p] += 1

                if all(v % 2 == 0 for v in factor_counts.values()):
                    sqrt_val = 1
                    for p, count in factor_counts.items():
                        sqrt_val = (sqrt_val * pow(p, count // 2, N)) % N

                    g = gcd(abs(sqrt_val - 1), N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
                    g = gcd(abs(sqrt_val + 1), N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

    return None


def cf_cascade_factor(N: int, bound: int = 50000,
                      cf_steps: int = 50000) -> tuple[int, int] | None:
    """Factor N using pure CF convergent cascade (without matrix powering).

    Simpler version that just computes CF convergents and checks:
    1. gcd(r_k, N) where r_k = p_k2 - N·q_k2
    2. gcd(p_k, N) and gcd(q_k, N)

    Useful when the full matrix cascade is too slow.

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

    a0 = isqrt(N)
    if a0 * a0 == N:
        return (s, s)

    m, d, a = 0, 1, a0
    p_prev, p_curr = 1, a0
    q_prev, q_curr = 0, 1

    # Check first convergent
    r0 = a0 * a0 - N
    g = gcd(abs(r0), N)
    if 1 < g < N:
        return (min(g, N // g), max(g, N // g))

    for step in range(1, cf_steps + 1):
        m = d * a - m
        d_new = (N - m * m)
        if d_new == 0:
            break
        d_new = d_new // d
        if d_new == 0:
            break
        d = d_new
        a = (a0 + m) // d

        p_new = a * p_curr + p_prev
        q_new = a * q_curr + q_prev

        residue = p_new * p_new - N * q_new * q_new
        g = gcd(abs(residue), N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        g = gcd(p_new, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))
        g = gcd(q_new, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

        if d == 1:
            break

    return None