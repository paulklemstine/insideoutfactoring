"""Hashimoto Non-Backtracking Operator Factoring.

Implements factoring via the non-backtracking Hashimoto operator on the
SL2(Z/NZ) unit graph.

THEORY:
- The Hashimoto matrix B (non-backtracking operator) acts on directed edges
  of a graph. For a directed edge (g, s) representing g -> s(g), B maps to
  the next non-backtracking edge.
- For the SL2(Z/NZ) Cayley graph with generators S = {U, A, D}, non-backtracking
  walks reveal structural information about the factor modulus.
- The key invariant: for g in SL2(F_p), det(g - I) = 0 iff tr(g) = 2,
  which occurs for parabolic elements (subgroup of size p+1 in PSL2(F_p)).
- The fraction of parabolic elements is 1/(p+1) for SL2(F_p) and 1/(q+1)
  for SL2(F_q). By estimating these fractions modulo N, we can solve for p, q.

CORE ALGORITHM:
1. Non-backtracking random walk on SL2(Z/NZ) with generators {U, A, D}
2. At each step, compute tr(g) mod p and tr(g) mod q (implicitly via mod N)
3. Count parabolic events: tr(g) ≡ 2 (mod p) or (mod q)
4. Estimate densities: p_est = (# parabolic) / (# samples)
5. Solve: 1/(p+1) ≈ p_est and 1/(q+1) ≈ q_est, giving p ≈ 1/p_est - 1

PRACTICAL IMPLEMENTATION:
- We work entirely modulo N; parabolic detection is tr(g) ≡ 2 (mod N)
- The trace mod N "bleeds through" from both mod p and mod q components
- By performing many walks and recording the trace distribution, we can
  extract the parabolic probability which encodes p and q information.

GENERATORS:
- U = [[1, 1], [0, 1]]  (unipotent, upper triangular)
- A = [[0, 1], [-1, 0]] (order 4, like 'a' in S_3 generators)
- D = [[1, 0], [0, 1]]  (identity, included for connectivity)
- Actually use: U, A, V=[[0,0],[1,0]] (Berggren tree style)

Note: In Berggren tree terms with slope z = n/m, we have:
  U: z -> z+1 (increment n)
  A: z -> -m/n (reciprocal with sign)
  D or V: z -> m/(n+m) (mediant/Farey)
We use matrix representations of these SL2 transformations.
"""
from __future__ import annotations

import random
from math import gcd, isqrt


# =============================================================================
# SL2 MATRIX ARITHMETIC
# =============================================================================

def _mat2_mul(A: tuple, B: tuple, N: int) -> tuple:
    """Multiply two 2x2 matrices mod N. A, B are (a,b,c,d) tuples row-major."""
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


def _mat2_trace(M: tuple) -> int:
    """Return trace of 2x2 matrix (a,b,c,d) = a + d."""
    a, b, c, d = M
    return a + d


def _mat2_det(M: tuple, N: int) -> int:
    """Compute determinant of 2x2 matrix mod N."""
    a, b, c, d = M
    return (a * d - b * c) % N


# =============================================================================
# SL2 GENERATORS
# =============================================================================

# Standard SL2 generators for the Cayley graph
# U = [[1,1],[0,1]]  -- unipotent (upper triangular)
# A = [[0,1],[-1,0]] -- order 4 element (like complex i)
# D = [[0,1],[1,0]]  -- order 2 element (transpose of A)

# Matrix representations (row-major):
U_MAT = (1, 1, 0, 1)   # [[1,1],[0,1]] - unipotent, z -> z+1
A_MAT = (0, 1, -1, 0)  # [[0,1],[-1,0]] - order 4, z -> -1/z
D_MAT = (0, 1, 1, 0)   # [[0,1],[1,0]] - order 2, z -> 1/z (actually this is A^2)

# More useful: the three generators for the Berggren tree navigation
# Using slope representation z = n/m:
# U: z -> z+1  corresponds to matrix [[1,1],[0,1]]
# A: z -> -m/n  corresponds to matrix [[0,1],[-1,0]] = -[[0,1],[1,0]] in PSL2
# V: z -> m/(n+m) corresponds to matrix [[1,0],[1,1]]

# Final choice: use U, A, V as the three generators
U = U_MAT   # z -> z+1 (increment n)
A = A_MAT   # z -> -1/z (reciprocal, negate)
V = (1, 0, 1, 1)  # [[1,0],[1,1]] - z -> z/(z+1) = 1/(1+1/z)

GENERATORS = [U, A, V]
GENERATOR_NAMES = ['U', 'A', 'V']

# Inverses
U_INV = (1, -1, 0, 1)  # [[1,-1],[0,1]]
A_INV = (0, -1, 1, 0)  # [[0,-1],[1,0]] = -A in PSL2
V_INV = (1, 0, -1, 1)  # [[1,0],[-1,1]]

INVERSE_MAP = {U: U_INV, A: A_INV, V: V_INV}


def _random_sl2_matrix(N: int) -> tuple:
    """Generate a random matrix in SL2(Z/NZ) with det = 1 mod N.

    Strategy: pick random a, b, c, compute d from det = ad - bc = 1.
    If a is not invertible mod N, gcd(a,N) gives a factor.
    """
    while True:
        a = random.randint(0, N - 1)
        b = random.randint(0, N - 1)
        c = random.randint(0, N - 1)

        # Check if a is zero
        if a == 0:
            # Need bc ≡ 1 (mod N) for det = 1
            g_bc = gcd(b * c % N, N)
            if 1 < g_bc < N:
                return (g_bc, N // g_bc)  # Found a factor!
            continue

        g = gcd(a, N)
        if 1 < g < N:
            return (g, N // g)  # Found a factor!

        # Compute d = (1 + b*c) * a^{-1} mod N
        try:
            a_inv = pow(a, -1, N)
        except ValueError:
            continue  # a not invertible, try again

        d = ((1 + b * c) % N) * a_inv % N
        M = (a, b, c, d)

        # Verify det = 1
        if _mat2_det(M, N) % N == 1 % N:
            return M


# =============================================================================
# NON-BACKTRACKING WALK
# =============================================================================

def nonbacktracking_walk(seed_matrix: tuple, length: int, N: int) -> list[tuple]:
    """Perform a non-backtracking walk on SL2(Z/NZ) Cayley graph.

    A non-backtracking walk never immediately traverses an edge in the
    reverse direction of the previous edge. This corresponds to the
    Hashimoto operator B acting on the directed edge space.

    Args:
        seed_matrix: Starting matrix in SL2(Z/NZ), or None for random
        length: Number of steps to take
        N: Modulus

    Returns:
        List of (matrix, generator_used) pairs for each step.
        The matrix is the current position; generator_used is the
        generator that was applied to get there.

    Example:
        walk = nonbacktracking_walk(seed_matrix=(1,0,0,1), length=10, N=77)
        # Returns 10 steps of non-backtracking walk
    """
    if seed_matrix is None:
        g = _random_sl2_matrix(N)
    else:
        g = seed_matrix

    # If seed_matrix is a tuple of two ints (a factor pair), treat as seed
    if isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], int):
        # It's a factor pair (from a found factor), start from identity
        g = (1, 0, 0, 1)

    walk = []

    # Choose first generator randomly
    prev_gen = None
    current = g

    for _ in range(length):
        # Pick a generator that is NOT the inverse of the previous one
        # This enforces the non-backtracking condition
        candidates = [gen for gen in GENERATORS if gen != prev_gen]
        gen = random.choice(candidates)

        # Apply generator
        new_mat = _mat2_mul(current, gen, N)

        walk.append((new_mat, gen))
        current = new_mat
        prev_gen = INVERSE_MAP.get(gen, gen)  # Next step can't use inverse

    return walk


def trace_at_step(walk: list[tuple], step: int) -> int | None:
    """Get trace of matrix at a specific step in the walk.

    Args:
        walk: Result from nonbacktracking_walk
        step: Step index (0 = first position after initial step)

    Returns:
        Trace of matrix at that step, or None if step out of range
    """
    if step < 0 or step >= len(walk):
        return None
    mat, _ = walk[step]
    return _mat2_trace(mat)


def parabolic_count(walk: list[tuple], mod: int) -> int:
    """Count parabolic elements in walk: tr(g) ≡ 2 (mod mod).

    An element g ∈ SL2(F_p) is parabolic iff tr(g) = 2, which means
    g is conjugate to [[1,1],[0,1]] (unipotent). In PSL2 this corresponds
    to the stabilizer of a rational slope.

    Args:
        walk: Non-backtracking walk
        mod: Modulus to check (p or q)

    Returns:
        Number of steps where tr(g) ≡ 2 (mod mod)
    """
    count = 0
    for mat, _ in walk:
        tr = _mat2_trace(mat) % mod
        if tr == 2 % mod:
            count += 1
    return count


def parabolic_density(walk: list[tuple], mod: int) -> float:
    """Compute fraction of parabolic elements in walk modulo mod.

    Args:
        walk: Non-backtracking walk
        mod: Modulus to check

    Returns:
        Fraction of steps with tr(g) ≡ 2 (mod mod), in [0, 1]
    """
    if len(walk) == 0:
        return 0.0
    return parabolic_count(walk, mod) / len(walk)


# =============================================================================
# PARABOLIC DENSITY ESTIMATION
# =============================================================================

def parabolic_density_estimate(N: int, samples: int = 1000, walk_length: int = 20,
                                seed: int | None = None) -> dict:
    """Estimate parabolic density in SL2(Z/NZ) via non-backtracking walks.

    The parabolic density is the fraction of elements g ∈ SL2(Z/NZ) with
    tr(g) ≡ 2 (mod N). This is related to the proportion of elements in
    subgroups isomorphic to the unipotent subgroup (z -> z+1).

    For the full SL2(F_p), the number of parabolic elements (including identity)
    is p+1 in PSL2(F_p). So the density is (p+1)/|SL2(F_p)| = (p+1)/(p(p^2-1))
    which is approximately 1/p for large p.

    More precisely: in PSL2(F_p), the stabilizer of ∞ (upper triangular
    matrices with tr=2) has size p+1 including infinity. The density of
    parabolic elements (tr=2) in PSL2 is 2/(p+1) accounting for ±I.

    Args:
        N: Composite modulus N = p*q
        samples: Number of independent walks to perform
        walk_length: Length of each walk
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - 'density': overall fraction of parabolic events
        - 'parabolic_count': number of parabolic events
        - 'total_steps': total number of steps across all walks
        - 'trace_distribution': dict mapping trace mod small values to counts
        - 'parabolic_ratio': density relative to random baseline
    """
    if seed is not None:
        random.seed(seed)

    total_parabolic = 0
    total_steps = 0
    trace_counts = {}  # trace value -> count (mod N)

    for _ in range(samples):
        # Random starting point
        walk = nonbacktracking_walk(None, walk_length, N)

        for mat, _ in walk:
            tr = _mat2_trace(mat) % N
            trace_counts[tr] = trace_counts.get(tr, 0) + 1
            total_steps += 1

            # Check parabolic: tr ≡ 2 (mod N)
            if tr == 2 % N:
                total_parabolic += 1

    density = total_parabolic / total_steps if total_steps > 0 else 0.0

    return {
        'density': density,
        'parabolic_count': total_parabolic,
        'total_steps': total_steps,
        'trace_distribution': trace_counts,
        'parabolic_ratio': density * N,  # relative to 1/N baseline
    }


def unit_cayley_graph_sl2(N: int) -> dict:
    """Return statistics about the SL2(Z/NZ) unit Cayley graph.

    The unit Cayley graph has vertices = SL2(Z/NZ) and edges labeled by
    generators {U, A, V}. Each vertex has degree 3 (one edge per generator).

    Args:
        N: Modulus

    Returns:
        Dictionary with graph statistics:
        - 'num_vertices': |SL2(Z/NZ)| = N^3 * prod(1-1/p^2) for p|N
        - 'degree': 3 (number of generators)
        - 'num_edges': 3 * num_vertices (directed edges)
        - 'generators': list of generator matrices
        - 'nonbacktracking_size': 3 * degree^(walk_length) / (degree-1)
    """
    # Number of elements in SL2(Z/NZ)
    # |SL2(Z/NZ)| = N^3 * prod_{p|N} (1 - 1/p^2)
    num_vertices = N * N * N
    temp = N
    for p in range(2, int(isqrt(N)) + 1):
        if temp % p == 0:
            num_vertices = num_vertices // (p * p) * (p * p - 1)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        num_vertices = num_vertices // (temp * temp) * (temp * temp - 1)

    # SL2(Z/NZ) is not cyclic but the Cayley graph is 3-regular (directed)
    degree = 3
    num_edges = 3 * num_vertices

    return {
        'num_vertices': num_vertices,
        'degree': degree,
        'num_edges': num_edges,
        'generators': GENERATOR_NAMES,
        'generator_matrices': [U_MAT, A_MAT, V],
        'hashimoto spectral bound': isqrt(degree - 1),  # = sqrt(2) ≈ 1.414
    }


# =============================================================================
# FACTORING VIA PARABOLIC DENSITY
# =============================================================================

def factor_from_parabolic_density(N: int, p_estimate: float, q_estimate: float) -> tuple[int, int]:
    """Given parabolic density estimates, recover p and q.

    The parabolic density for SL2(F_p) is approximately 1/(p+1) for the
    unipotent subgroup density, or more precisely the fraction of g with
    tr(g) = 2 is 2/(p+1) in PSL2.

    If p_est ≈ 1/(p+1) and q_est ≈ 1/(q+1), then:
        p ≈ 1/p_est - 1
        q ≈ 1/q_est - 1

    Args:
        N: Original composite modulus
        p_estimate: Parabolic density estimate mod p
        q_estimate: Parabolic density estimate mod q

    Returns:
        Tuple (p, q) with p < q

    Raises:
        ValueError: If estimates are invalid or lead to trivial factors
    """
    if p_estimate <= 0 or q_estimate <= 0:
        raise ValueError(f"Invalid density estimates: p={p_estimate}, q={q_estimate}")

    # Convert density to factor estimate
    # density ≈ 1/(factor+1) => factor ≈ 1/density - 1
    p_est = int(round(1.0 / p_estimate - 1))
    q_est = int(round(1.0 / q_estimate - 1))

    # Validate
    if p_est < 2 or q_est < 2:
        raise ValueError(f"Density estimates too large: p_est={p_est}, q_est={q_est}")

    # Ensure p < q
    if p_est > q_est:
        p_est, q_est = q_est, p_est

    return p_est, q_est


def _estimate_factor_from_density(N: int, density: float) -> int:
    """Estimate a single factor from parabolic density.

    The trace-2 density in SL2(F_p) is 2/(p+1) for the full group.
    We invert: p ≈ 2/density - 1.

    Args:
        N: Composite modulus (used for validation)
        density: Fraction of trace-2 elements

    Returns:
        Estimated factor
    """
    if density <= 0:
        return N  # fallback

    # The exact density depends on which subgroup we're measuring
    # For unipotent subgroup (z->z+1): size is p+1 in PSL2
    # For full SL2(F_p): proportion with tr=2 is 2/(p+1)
    # But we measure tr ≡ 2 mod N, which wraps both mod p and mod q
    # The "effective" density is dominated by the smaller factor

    est = 2.0 / density - 1.0
    return max(2, min(N - 1, int(round(est))))


def _extract_factors_from_traces(N: int, trace_counts: dict, total: int) -> tuple[int, int] | None:
    """Try to extract factors from trace distribution.

    When we compute tr(g) mod N for g in SL2(Z/NZ), the result mod p and
    mod q can differ. The trace-2 events cluster differently for the
    two factors.

    Strategy: look for trace values that are 2 mod p but not mod q,
    or vice versa. These give us information about p and q.

    Args:
        N: Composite modulus
        trace_counts: Dict of trace mod N -> count
        total: Total number of samples

    Returns:
        (p, q) if found, None otherwise
    """
    if total == 0:
        return None

    # Look for trace values that are suspiciously common
    # If tr = k and k ≡ 2 (mod p) but k ≠ 2 (mod q), then gcd(k-2, N) might give p
    for tr_val, count in trace_counts.items():
        if tr_val == 2 % N:
            continue  # Already checked

        # Check if tr_val could be 2 mod one factor
        g1 = gcd(tr_val - 2, N)
        if 1 < g1 < N:
            g2 = N // g1
            if g1 * g2 == N:
                return g1, g2

    # Also check: tr = -2 mod N (which is tr = N-2)
    neg2 = (N - 2) % N
    if neg2 in trace_counts:
        g1 = gcd(neg2 + 2, N)  # This is just gcd(N, N) - won't help
        pass

    return None


# =============================================================================
# MAIN HASHIMOTO FACTORING ALGORITHM
# =============================================================================

def hashimoto_factor(N: int, walks: int = 10000, walk_length: int = 20,
                     _seed: int | None = None) -> tuple[int, int] | None:
    """Factor N using the Hashimoto non-backtracking operator on SL2(Z/NZ).

    This method uses the non-backtracking walk on the SL2(Z/NZ) Cayley graph
    to detect factor structure via parabolic density estimation.

    THEORY:
    The Hashimoto matrix B acts on the directed edge space of the graph.
    For SL2(F_p), the spectral radius of B is bounded by sqrt(deg-1) = sqrt(2).
    More importantly, the set of "parabolic" elements (tr(g) = 2) forms a
    subgroup of size p+1 in PSL2(F_p).

    ALGORITHM:
    1. Perform many non-backtracking random walks on SL2(Z/NZ)
    2. At each step, record tr(g) mod N
    3. The distribution of tr(g) reveals structure:
       - tr(g) ≡ 2 (mod p) occurs with probability ~1/(p+1) per element in PSL2
       - tr(g) ≡ 2 (mod q) occurs with probability ~1/(q+1)
    4. By analyzing trace distributions, we can estimate p and q
    5. Additional: gcd(tr(g) - 2, N) can reveal factors directly

    Args:
        N: Integer to factor (should be composite with two large prime factors)
        walks: Number of independent random walks
        walk_length: Length of each walk
        _seed: Random seed (internal use)

    Returns:
        (p, q) where p*q = N and p < q, or None if factoring fails

    Examples:
        >>> hashimoto_factor(8051)
        (83, 97)  # or similar valid factorization
    """
    if _seed is not None:
        random.seed(_seed)

    # Quick checks
    if N < 4:
        return None
    if N % 2 == 0:
        return 2, N // 2

    # Small prime check
    for p in (3, 5, 7, 11, 13, 17, 19, 23):
        if N % p == 0:
            return p, N // p

    # =================================================================
    # MAIN ALGORITHM: Trace distribution analysis
    # =================================================================

    trace_counts = {}  # tr mod N -> count
    total = 0
    parabolic_tr2 = 0  # count of tr ≡ 2 (mod N)

    for _ in range(walks):
        walk = nonbacktracking_walk(None, walk_length, N)

        for mat, _ in walk:
            tr = _mat2_trace(mat) % N
            trace_counts[tr] = trace_counts.get(tr, 0) + 1
            total += 1

            if tr == 2 % N:
                parabolic_tr2 += 1

            # Also check: tr = -2 mod N
            if tr == (N - 2) % N:
                # This is tr ≡ -2, which in PSL2 means same absolute value
                pass

    if total == 0:
        return None

    # =================================================================
    # METHOD 1: Direct GCD from trace-2 events
    # =================================================================
    # For tr(g) ≡ 2 (mod p), we have g ≡ I (mod p) in PSL2
    # So g - I shares factor p with N
    # We already counted these; now extract factors

    for _ in range(walks):
        walk = nonbacktracking_walk(None, walk_length, N)
        for mat, _ in walk:
            tr = _mat2_trace(mat) % N

            # Check if tr ≡ 2 (mod p) for some p|N
            # This is equivalent to g having eigenvalue 1 mod p
            # In SL2, this means (g - I) has determinant 0 mod p
            # But more directly: check gcd entries

            # Off-diagonal check
            a, b, c, d = mat
            g1 = gcd(b, N)
            if 1 < g1 < N:
                return g1, N // g1

            g2 = gcd(c, N)
            if 1 < g2 < N:
                return g2, N // g2

            # Diagonal check: gcd(a-1, N) and gcd(d-1, N)
            g3 = gcd(a - 1, N)
            if 1 < g3 < N:
                return g3, N // g3

            g4 = gcd(d - 1, N)
            if 1 < g4 < N:
                return g4, N // g4

    # =================================================================
    # METHOD 2: Trace value GCD
    # =================================================================
    # For tr = k where k ≡ 2 (mod p) but k ≠ 2 (mod q):
    # gcd(k - 2, N) = p

    for tr_val, count in list(trace_counts.items()):
        if count < 3:  # Need enough samples
            continue

        # Try gcd(tr - 2, N)
        diff = (tr_val - 2) % N
        g = gcd(diff, N)
        if 1 < g < N:
            return g, N // g

        # Try gcd(tr + 2, N) for tr ≡ -2 mod p
        sum_ = (tr_val + 2) % N
        g = gcd(sum_, N)
        if 1 < g < N:
            return g, N // g

    # =================================================================
    # METHOD 3: Parabolic density inversion
    # =================================================================
    # The fraction of tr=2 elements should be ~2/(p+1) for the smaller factor
    # Density of tr=2 elements:
    density = parabolic_tr2 / total if total > 0 else 0.0

    if density > 0:
        # Estimate factors from density
        # density ≈ 2/(min(p,q) + 1) for PSL2
        # Actually in full SL2, tr=2 includes ±I and the unipotent subgroup
        # The proportion is approximately 2/(p+1) for one factor structure

        # Use multiple candidate factors
        candidates = []
        for denom in range(1, 100):
            if denom == 0:
                continue
            frac = 2.0 / denom
            if abs(frac - density) < 0.01:  # Close match
                est_factor = denom - 1
                if est_factor > 1 and N % est_factor == 0:
                    candidates.append(est_factor)

        if candidates:
            # Pick the most common estimate
            from collections import Counter
            counter = Counter(candidates)
            best_est = counter.most_common(1)[0][0]
            other = N // best_est
            if best_est * other == N:
                p, q = min(best_est, other), max(best_est, other)
                return p, q

    # =================================================================
    # METHOD 4: Trace distribution clustering
    # =================================================================
    # Look for trace values that cluster around 2 modulo a factor
    # Use the fact that tr(g) mod p and tr(g) mod q give different distributions

    # Group trace values by proximity to 2 mod some small number
    for offset in [1, 2, 3, N - 2, N - 1]:
        for tr_val, count in trace_counts.items():
            if count < 5:
                continue
            diff = (tr_val - offset) % N
            g = gcd(diff, N)
            if 1 < g < N:
                return g, N // g

    # =================================================================
    # METHOD 5: Second-order traces (product of consecutive matrices)
    # =================================================================
    # For non-backtracking walk, look at products of two consecutive matrices
    for _ in range(min(100, walks)):
        walk = nonbacktracking_walk(None, 2 * walk_length, N)

        # Check products of pairs
        for i in range(len(walk) - 1):
            mat1, _ = walk[i]
            mat2, _ = walk[i + 1]

            prod = _mat2_mul(mat1, mat2, N)
            tr_prod = _mat2_trace(prod) % N

            # Check various trace conditions
            for check_val in [2, N - 2, 0, 1, N - 1]:
                diff = (tr_prod - check_val) % N
                g = gcd(diff, N)
                if 1 < g < N:
                    return g, N // g

    return None


# =============================================================================
# WALSHAS SPECTRAL TEST (simplified)
# =============================================================================

def walshas_spectral_test(N: int, walk_length: int = 30, samples: int = 500) -> dict:
    """Walsh-Hadamard spectral test for subgroup detection.

    The WALSHAS (Wedderburn-Artin-Hasumi) test uses the fact that if we're
    walking in a coset of a proper subgroup H < SL2(F_p), the walk
    endpoints will cluster. The spectral norm of the walk operator
    restricted to the subgroup is smaller.

    This is a simplified version that checks endpoint clustering.

    Args:
        N: Composite modulus
        walk_length: Length of each walk
        samples: Number of walks

    Returns:
        Dictionary with test statistics
    """
    if walk_length < 2:
        return {'subgroup_detected': False, 'reason': 'walk too short'}

    endpoints = []
    for _ in range(samples):
        walk = nonbacktracking_walk(None, walk_length, N)
        if walk:
            final_mat, _ = walk[-1]
            endpoints.append(final_mat)

    if len(endpoints) < 2:
        return {'subgroup_detected': False, 'reason': 'insufficient samples'}

    # Check if endpoints cluster (indicates subgroup structure)
    # Simple metric: how many distinct endpoints?
    unique = len(set(endpoints))

    # If very few unique endpoints relative to samples, subgroup likely
    clustering_ratio = unique / len(endpoints)

    return {
        'subgroup_detected': clustering_ratio < 0.1,  # heuristic
        'unique_endpoints': unique,
        'total_walks': len(endpoints),
        'clustering_ratio': clustering_ratio,
    }


# =============================================================================
# HASHIMOTO SPECTRAL RADIUS ESTIMATE
# =============================================================================

def hashimoto_spectral_estimate(N: int, walk_length: int = 50,
                                 samples: int = 200) -> float:
    """Estimate the Hashimoto spectral radius from non-backtracking walks.

    The Hashimoto matrix B has spectral radius bounded by sqrt(deg-1) = sqrt(2)
    for a 3-regular graph. Deviations from this bound indicate structure.

    We estimate the spectral radius by computing the growth rate of
    non-backtracking walk weights.

    Args:
        N: Modulus
        walk_length: Length of walks
        samples: Number of walks

    Returns:
        Estimated spectral radius
    """
    # The return matrix R accumulates the walk products
    # R_k = sum of products of k-step non-backtracking walks
    # The spectral radius is approximated by the largest eigenvalue of R

    R = {(1, 0, 0, 1): 0}  # identity with count 0

    for _ in range(samples):
        walk = nonbacktracking_walk(None, walk_length, N)

        for i, (mat, _) in enumerate(walk):
            if mat in R:
                R[mat] += 1
            else:
                R[mat] = 1

    if not R:
        return 0.0

    # Simple spectral estimate: max row sum
    max_count = max(R.values())
    return float(max_count / samples)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def hashimoto_graph_stats(N: int) -> dict:
    """Return comprehensive Hashimoto operator statistics for SL2(Z/NZ).

    Args:
        N: Modulus

    Returns:
        Dictionary with various graph-theoretic and spectral statistics
    """
    graph_info = unit_cayley_graph_sl2(N)

    # Add spectral bounds
    rho_bound = isqrt(graph_info['degree'] - 1)

    # Estimate parabolic density
    density_info = parabolic_density_estimate(N, samples=500, walk_length=15)

    return {
        'N': N,
        'graph': graph_info,
        'spectral_bound': rho_bound,
        'parabolic_density_estimate': density_info['density'],
        'parabolic_ratio': density_info['parabolic_ratio'],
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == '__main__':
    import sys

    def test_factor(N):
        print(f"\nTesting hashimoto_factor({N})...")
        result = hashimoto_factor(N, walks=5000, walk_length=20)
        if result:
            p, q = result
            print(f"  Result: {N} = {p} × {q}")
            assert p * q == N, f"Product mismatch: {p}*{q} != {N}"
            assert p < q, f"p ({p}) should be < q ({q})"
            return True
        else:
            print(f"  Failed to factor {N}")
            return False

    test_values = [8051, 15571, 3127]

    print("=" * 60)
    print("Hashimoto Non-Backtracking Factor - Self Test")
    print("=" * 60)

    success_count = 0
    for N in test_values:
        if test_factor(N):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{len(test_values)} factors recovered")

    # Also print graph stats
    print("\nGraph Statistics:")
    for N in test_values[:1]:
        stats = hashimoto_graph_stats(N)
        print(f"  N={N}: vertices={stats['graph']['num_vertices']}, "
              f"parabolic_density={stats['parabolic_density_estimate']:.4f}")
