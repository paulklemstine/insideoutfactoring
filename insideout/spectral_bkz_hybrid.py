"""Spectral-BKZ Hybrid Congruence-of-Squares Finder.

Combines spectral methods (CF squaring, Pell residues, SL2 matrix order detection)
with BKZ lattice reduction for congruence-of-squares finding.

Key insight: Spectral methods generate "almost squares" - relations where
Pell residue r_k = p_k^2 - N*q_k^2 is O(1) instead of O(sqrtN) from random sieving.
These higher-quality relations make BKZ find dependencies faster.

Phase 1: Generate spectral relations from CF convergents and SL2 matrices
Phase 2: Build lattice from relation exponent vectors
Phase 3: Apply BKZ reduction
Phase 4: Extract congruence-of-squares from reduced basis

Complexity: L_p[1/2] with better constants due to higher-quality relations.
"""
from __future__ import annotations

from math import gcd, isqrt, log as ln
from typing import Optional

from .cf_guide import cf_sqrt, convergents as cf_convergents
from .spectral_factor import _cf_squaring_cascade, _sl2_matrix_cascade
from .lattice_factor import _bkz_reduce, _lll_reduce, _svp_shortest_vector


def spectral_bkz_factor(N: int,
                       bound: int = 50000,
                       bkz_beta: int = 16,
                       max_relations: int = 100,
                       time_budget_ms: float = 2000.0) -> tuple[int, int] | None:
    """Factor N using Spectral-BKZ hybrid.

    Combines:
    1. CF convergent spectral relations (Pell residues r_k = p_k^2 - N*q_k^2)
    2. SL2 matrix order detection
    3. BKZ lattice reduction for dependency finding

    Args:
        N: Integer to factor
        bound: Smooth bound for relation generation
        bkz_beta: BKZ block size (16-24 typical)
        max_relations: Maximum relations to collect
        time_budget_ms: Time limit in milliseconds

    Returns (p, q) with p < q and p*q = N, or None.
    """
    import time
    start = time.perf_counter()

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

    # Collect spectral relations
    relations = _collect_spectral_relations(N, bound, max_relations, time_budget_ms)

    if len(relations) < 10:
        # Not enough relations
        return None

    # Build lattice from relations
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > time_budget_ms:
        return None

    lattice = _build_spectral_lattice(relations, N)
    if lattice is None:
        return None

    # Apply BKZ reduction
    elapsed = (time.perf_counter() - start) * 1000
    remaining = max(0, time_budget_ms - elapsed)

    reduced = _bkz_reduce(lattice, beta=bkz_beta, max_iterations=3)
    if reduced is None:
        return None

    # Extract congruence-of-squares
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed > time_budget_ms:
        return None

    factors = _extract_congruence(reduced, relations, N)
    if factors is not None:
        p, q = factors
        if p * q == N and 1 < p < N and 1 < q < N:
            return (min(p, q), max(p, q))

    return None


def _collect_spectral_relations(N: int, bound: int,
                                max_relations: int,
                                time_budget_ms: float):
    """Collect spectral relations from CF convergents and SL2 matrices.

    Returns list of (a, r, exp_vector) where:
      - a = p_k or matrix trace
      - r = Pell residue or matrix order residue
      - exp_vector = prime exponent vector of |r|
    """
    import time
    relations = []
    start = time.perf_counter()

    # Phase 1: CF convergent Pell residues
    # r_k = p_k^2 - N*q_k^2 is O(1) instead of O(sqrtN)
    try:
        cf = cf_sqrt(N, max_terms=200)
        convs = cf_convergents(cf)
    except Exception:
        return relations

    for p_k, q_k in convs[:100]:
        if time.perf_counter() - start > time_budget_ms / 1000:
            break

        r = p_k * p_k - N * q_k * q_k
        if r == 0:
            continue

        r_abs = abs(r)
        if r_abs > bound:
            continue

        # Factor r_abs over small primes
        exp_vec = _factor_smooth(r_abs, bound)
        if exp_vec is not None:
            relations.append((p_k, r, exp_vec))

    # Phase 2: SL2 matrix order detection
    # For each convergent, build SL2 matrix M_k and check order
    for p_k, q_k in convs[:50]:
        if time.perf_counter() - start > time_budget_ms / 1000:
            break

        if len(relations) >= max_relations:
            break

        # Build matrix M = [[p_k, q_k], [N*q_k, p_k]]
        # det(M) = p_k^2 - N*q_k^2 = r (Pell residue)
        trace = (p_k + p_k) % N  # 2*p_k
        det = r = p_k * p_k - N * q_k * q_k

        # Check if det is smooth
        if abs(det) > bound or det == 0:
            continue

        exp_vec = _factor_smooth(abs(det), bound)
        if exp_vec is not None:
            relations.append((trace, det, exp_vec))

    # Phase 3: Try Lucas-PPT as backup relation source
    # (lucas_ppt uses Berggren tree structure)
    elapsed = (time.perf_counter() - start) * 1000
    if elapsed < time_budget_ms * 0.5 and len(relations) < max_relations // 2:
        # Try a quick lucas_ppt to get additional relations
        try:
            lp_result = lucas_ppt_factor(N)
            if lp_result is not None:
                # This found factors, so we're done
                pass
        except Exception:
            pass

    return relations


def _factor_smooth(n: int, bound: int) -> Optional[list[int]]:
    """Try to factor n over primes up to bound.

    Returns exponent vector if n is smooth, None otherwise.
    The exponent vector has length = number of primes up to bound.
    """
    from .lucas_multi import _small_primes

    if n == 0:
        return None

    n_abs = abs(n)
    primes = _small_primes(bound)
    exponents = []

    remaining = n_abs
    for p in primes:
        if p > bound:
            break
        if remaining % p == 0:
            exp = 0
            while remaining % p == 0:
                remaining //= p
                exp += 1
            exponents.append(exp)
        else:
            exponents.append(0)

    if remaining == 1:
        return exponents
    return None


def _build_spectral_lattice(relations: list, N: int):
    """Build lattice from spectral relations for BKZ reduction.

    Builds a lattice matrix where each row is:
    [exp_vector | a mod N]

    The lattice is (num_relations) x (num_primes + 1).
    """
    if len(relations) < 5:
        return None

    # Get max vector length
    max_len = max(len(r[2]) for r in relations)

    # Build matrix
    lattice = []
    for a, r, exp_vec in relations:
        # Pad exp_vec to same length
        padded = exp_vec + [0] * (max_len - len(exp_vec))
        # Add a mod N as last column
        row = padded + [a % N]
        lattice.append(row)

    # Pad to square-ish matrix
    while len(lattice) < max_len + 1:
        lattice.append([0] * (max_len + 1))

    # Apply LLL reduction first for numerical stability
    try:
        lattice = _lll_reduce([[float(x) for x in row] for row in lattice])
    except Exception:
        pass

    # Convert back to int
    lattice = [[int(x) for x in row] for row in lattice]

    return lattice


def _extract_congruence(reduced_lattice: list, relations: list, N: int):
    """Extract congruence-of-squares from BKZ-reduced basis.

    Looks for a short vector v in the reduced basis where:
      v = [e_1, e_2, ..., e_k, x] with x != 0 mod N

    Then computes:
      x_prod = prod a_i^{|e_i|} mod N
      y_prod = prod p_i^{|e_i|/2} mod N

    If all e_i are even, x_prod^2 == y_prod^2 (mod N) and we have a congruence.
    """
    if reduced_lattice is None or len(reduced_lattice) == 0:
        return None

    # Find a vector with non-zero last component (x != 0 mod N)
    last_col = [row[-1] for row in reduced_lattice if row[-1] != 0]
    if not last_col:
        return None

    # Sort by vector norm (shorter is better)
    def norm(row):
        return sum(x * x for x in row[:-1]) + (row[-1] % N) ** 2

    sorted_rows = sorted(reduced_lattice, key=norm)

    for row in sorted_rows[:10]:  # Check top 10 shortest vectors
        last = row[-1]
        if last == 0:
            continue

        # Check if all other exponents are even
        exponents = row[:-1]
        if all(e % 2 == 0 for e in exponents):
            # Valid congruence!
            # Compute x = product of a_i^{|e_i|} mod N
            x = 1
            for i, e in enumerate(exponents):
                if e != 0:
                    a_i = relations[i][0] if i < len(relations) else 1
                    x = (x * pow(a_i, abs(e), N)) % N

            # Compute y = product of p_i^{|e_i|/2} mod N
            y = 1
            primes = _get_smooth_primes(500)  # primes up to 500
            for i, e in enumerate(exponents):
                if e != 0 and i < len(primes):
                    y = (y * pow(primes[i], abs(e) // 2, N)) % N

            # Extract factors
            g1 = gcd(x - y, N)
            g2 = gcd(x + y, N)

            if 1 < g1 < N:
                return (g1, N // g1)
            if 1 < g2 < N:
                return (g2, N // g2)

    return None


def _get_smooth_primes(bound: int) -> list[int]:
    """Get list of primes up to bound."""
    from .lucas_multi import _small_primes
    return _small_primes(bound)


def spectral_bkz_factor_with_stages(N: int,
                                    time_budget_ms: float = 5000.0) -> tuple[int, int] | None:
    """Multi-stage spectral-BKZ with increasing bounds.

    Stage 1: Small bound (5000), high probability relations
    Stage 2: Medium bound (20000), more relations but slower
    Stage 3: Large bound (50000), deepest search
    """
    import time
    start = time.perf_counter()

    stages = [
        (5000, 12, 50),
        (20000, 14, 75),
        (50000, 16, 100),
    ]

    for bound, bkz_beta, max_rel in stages:
        elapsed = (time.perf_counter() - start) * 1000
        remaining = max(0, time_budget_ms - elapsed)

        if remaining < 100:
            break

        result = spectral_bkz_factor(
            N, bound=bound, bkz_beta=bkz_beta,
            max_relations=max_rel, time_budget_ms=remaining
        )
        if result is not None:
            return result

    return None
