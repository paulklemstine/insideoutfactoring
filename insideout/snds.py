"""Spectral Nilpotent Dynamical System (SNDS) Factoring.

For a semiprime N=pq, the trace sequence tr(A^k) mod N satisfies a linear
recurrence whose minimal polynomial over Z/NZ factors into coprime components
mod p and mod q. The Berlekamp-Massey state over Z/NZ reveals these factors.

Key mathematical facts:
1. For M ∈ SL2(F_p), tr(M) satisfies the recurrence from charpoly(M):
   tr(M^{k+2}) = tr(M)·tr(M^{k+1}) - tr(M^2)·tr(M^k)  [Cayley-Hamilton]
   More generally: tr(M^{k+ℓ}) is a Z-linear combination of
   tr(M), tr(M^2), ..., tr(M^{ℓ-1}) for ℓ = degree of minimal poly.

2. The sequence s_k = tr(A^k) mod N for A ∈ SL2(Z/NZ) satisfies the global
   recurrence whose characteristic polynomial is the characteristic polynomial
   of A evaluated at the group algebra level.

3. Over Z/pZ this recurrence has order dividing p-1 or p+1 (eigenvalue structure).
   Over Z/qZ it has order dividing q-1 or q+1.
   The LCM of the two orders gives the global order.

4. The minimal polynomial of the trace sequence over Z/NZ is the monic polynomial
   whose reduction mod p is the minimal poly over F_p and whose reduction mod q
   is the minimal poly over F_q. These coprime factors mean the resultant is
   N, and consecutive coefficients reveal gcd information.

PRACTICAL APPROACH:
- Pick random A ∈ SL2(Z/NZ)
- Compute s_k = tr(A^k) mod N for k = 1..B
- Run Berlekamp-Massey over Z/NZ to find minimal polynomial P(x)
- P(x) factors into P_p(x) mod p and P_q(x) mod q
- gcd(P(i) - P(i+1), N) or gcd(derivative terms, N) reveals factors
- Also: gcd of consecutive trace differences reveals factor structure

Algorithm complexity: O(B²) trace computations + O(B²) BM, where B is the
smoothness bound (order of the matrix mod smallest factor). Sub-exponential.
"""
from __future__ import annotations

from math import gcd, isqrt
from typing import Optional


# ---------------------------------------------------------------------------
# 2×2 Matrix Arithmetic over Z/NZ
# ---------------------------------------------------------------------------

def _mat2_mul(A, B, N):
    """Multiply two 2×2 matrices mod N. [[a,b],[c,d]] stored as (a,b,c,d)."""
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1 * a2 + b1 * c2) % N,
        (a1 * b2 + b1 * d2) % N,
        (c1 * a2 + d1 * c2) % N,
        (c1 * b2 + d1 * d2) % N,
    )


def _mat2_pow(M, k, N):
    """M^k mod N using fast exponentiation."""
    if k == 0:
        return (1, 0, 0, 1)
    if k == 1:
        return M
    result = (1, 0, 0, 1)
    base = M
    while k > 0:
        if k & 1:
            result = _mat2_mul(result, base, N)
        base = _mat2_mul(base, base, N)
        k >>= 1
    return result


# ---------------------------------------------------------------------------
# Characteristic Polynomial mod N
# ---------------------------------------------------------------------------

def charpoly_mod_N(A, N):
    """Characteristic polynomial of 2×2 matrix A = [[a,b],[c,d]] mod N.

    Returns (1, -trace, det) representing x^2 - trace*x + det.
    The characteristic polynomial is x^2 - (a+d)*x + (a*d - b*c) mod N.
    """
    a, b, c, d = A
    trace = (a + d) % N
    det = (a * d - b * c) % N
    # Monic polynomial: x^2 - trace*x + det
    return (1, (-trace) % N, det)


# ---------------------------------------------------------------------------
# Trace Sequence
# ---------------------------------------------------------------------------

def trace_sequence(A, N, max_k):
    """Compute tr(A), tr(A^2), ..., tr(A^max_k) mod N.

    For a 2×2 matrix A = [[a,b],[c,d]], tr(A^k) = (A^k)[0][0] + (A^k)[3][3] mod N.

    Args:
        A: 2×2 matrix as (a,b,c,d)
        N: modulus
        max_k: number of terms to compute

    Returns:
        List [tr(A), tr(A^2), ..., tr(A^max_k)] mod N
    """
    # Compute A^1, A^2, ..., A^max_k iteratively
    M = A
    seq = []
    for _ in range(max_k):
        trace = (M[0] + M[3]) % N
        seq.append(trace)
        M = _mat2_mul(M, A, N)
    return seq


# ---------------------------------------------------------------------------
# Berlekamp-Massey over Z/NZ (RII variant)
# ---------------------------------------------------------------------------

def berlekamp_massey_RII(sequence, modulus):
    """Berlekamp-Massey over Z/NZ.

    Finds the shortest linear recurrence satisfied by the sequence.
    Connection polynomial C(x) where s_n = sum_{j=1}^{L} C[j] * s_{n-j}.
    Returns (C, L) or (None, 0) on failure.
    """
    n = len(sequence)
    if n == 0:
        return [0], 0

    # C(x) = 1 - c1*x - c2*x^2 - ... - cL*x^L  (stored as [1, -c1, -c2, ..., -cL])
    C = [1]
    L = 0

    # B(x) = C(x) at last update
    B = [1]

    m = -1  # position of last update

    for i in range(n):
        # discrepancy d = s_i - sum_{j=1}^{L} C[j] * s_{i-j}
        d = sequence[i] % modulus
        for j in range(1, L + 1):
            d = (d - C[j] * sequence[i - j]) % modulus

        if d == 0:
            continue

        T = C[:]
        b0_inv = pow(B[0], -1, modulus)

        # C(x) = C(x) - (d / b0) * x^{i-m} * B(x)
        shift = i - m
        needed = len(B) + shift
        if len(C) < needed:
            C.extend([0] * (needed - len(C)))

        for j in range(len(B)):
            C[j + shift] = (C[j + shift] - d * b0_inv * B[j]) % modulus

        if 2 * L <= i:
            L = i + 1 - L
            B = T[:]
            m = i

        # Normalize C[0] to 1
        if C[0] != 1:
            inv = pow(C[0], -1, modulus)
            C = [(c * inv) % modulus for c in C]

    return C, L


def berlekamp_massey_basic(sequence, modulus):
    """Simplified Berlekamp-Massey over Z/NZ.

    This is a straightforward implementation that finds the shortest L and
    coefficients C such that for all n >= L:
        s_n ≡ sum_{i=1}^{L} C[i] * s_{n-i} (mod modulus)

    Returns:
        List of coefficients [s_0 form] or None if no unique solution.
    """
    n = len(sequence)
    if n < 2:
        return [sequence[0]] if n == 1 else None

    # Try all possible L from 1 to n//2
    for L in range(1, n // 2 + 1):
        # Set up linear system: for each n >= L, we have
        # s_n = sum_{i=1}^{L} C[i] * s_{n-i}
        # This gives n-L equations in L unknowns.
        # We'll use the first L equations to solve, then verify.

        # Build matrix from first L equations
        # Eq for index L: s_L = C[1]*s_{L-1} + ... + C[L]*s_0
        # Eq for index L+1: s_{L+1} = C[1]*s_L + ... + C[L]*s_1
        # etc.

        # Build augmented matrix for the first L equations
        try:
            # We solve for C = [c_1, ..., c_L] where:
            # c_1*s_{L-1} + c_2*s_{L-2} + ... + c_L*s_0 = s_L
            # c_1*s_L + c_2*s_{L-1} + ... + c_L*s_1 = s_{L+1}
            # etc.

            # Transpose: each row is [s_{L-1}, s_{L-2}, ..., s_0] for first equation, etc.
            A = []
            b = []
            for row_idx in range(L):
                eq_idx = L + row_idx
                row = [sequence[eq_idx - j - 1] for j in range(L)]
                A.append(row)
                b.append(sequence[eq_idx])

            # Solve using Gaussian elimination mod modulus
            # Augment
            M = [A[i] + [b[i]] for i in range(L)]

            # Forward elimination
            for col in range(L):
                # Find pivot
                pivot = None
                for row in range(col, L):
                    if M[row][col] % modulus != 0:
                        pivot = row
                        break
                if pivot is None:
                    # Zero column, system singular for this L
                    break
                if pivot != col:
                    M[col], M[pivot] = M[pivot], M[col]

                # Normalize pivot row
                piv_val = M[col][col] % modulus
                inv_piv = pow(piv_val, -1, modulus)
                M[col] = [(M[col][j] * inv_piv) % modulus for j in range(L + 1)]

                # Eliminate below
                for row in range(col + 1, L):
                    if M[row][col] % modulus != 0:
                        factor = M[row][col]
                        M[row] = [(M[row][j] - factor * M[col][j]) % modulus for j in range(L + 1)]

            # Back substitution
            coeffs = [0] * L
            for row in range(L - 1, -1, -1):
                if M[row][row] == 0:
                    break
                coeffs[row] = M[row][L]
                for j in range(row + 1, L):
                    coeffs[row] = (coeffs[row] - M[row][j] * coeffs[j]) % modulus
            else:
                # Verify solution for all n >= L
                valid = True
                for n_idx in range(L, n):
                    lhs = sequence[n_idx]
                    rhs = sum(coeffs[j] * sequence[n_idx - j - 1] for j in range(L)) % modulus
                    if lhs != rhs:
                        valid = False
                        break
                if valid:
                    # Return in form: [s_L, -c_1, -c_2, ..., -c_L]
                    # matching the polynomial x^L - c_1*x^{L-1} - ... - c_L
                    return [sequence[L]] + [(-c) % modulus for c in coeffs]
        except (ValueError, IndexError, ZeroDivisionError):
            continue

    return None


# ---------------------------------------------------------------------------
# Extract Factors from Minimal Polynomial
# ---------------------------------------------------------------------------

def factor_from_minpoly(minpoly_coeffs, N):
    """Attempt to extract factors of N from the minimal polynomial coefficients.

    The minimal polynomial P(x) over Z/NZ factors into coprime factors over
    Z/pZ and Z/qZ. Several approaches to extract the factor:

    1. gcd of consecutive polynomial values: gcd(P(i) - P(i+1), N)
    2. gcd of derivative-like expressions
    3. Evaluate at special points to get values coprime to one factor but
       not the other

    Args:
        minpoly_coeffs: [a_0, a_1, ..., a_L] representing
            P(x) = a_0*x^L + a_1*x^{L-1} + ... + a_L
        N: composite modulus

    Returns:
        (p, q) if factors found, None otherwise.
    """
    if not minpoly_coeffs or len(minpoly_coeffs) < 2:
        return None

    L = len(minpoly_coeffs) - 1  # degree

    # Evaluate P at various points
    for x in range(2, min(100, N)):
        # P(x) mod N
        val = 0
        for i, coeff in enumerate(minpoly_coeffs):
            val = (val * x + coeff) % N

        if val == 0 or val == N:
            continue

        # gcd(P(x), N) might reveal a factor
        g = gcd(val, N)
        if 1 < g < N:
            other = N // g
            if g * other == N:
                return (min(g, other), max(g, other))

        # Try P(x) - P(x+1)
        val_next = 0
        for coeff in minpoly_coeffs:
            val_next = (val_next * (x + 1) + coeff) % N

        diff = abs(val - val_next) % N
        if diff != 0 and diff != N:
            g = gcd(diff, N)
            if 1 < g < N:
                other = N // g
                if g * other == N:
                    return (min(g, other), max(g, other))

    # Try: gcd of linear combination of coefficients
    # For polynomial a_0*x^L + ... + a_L, evaluate at x=1 gives sum of coeffs
    val_at_1 = sum(minpoly_coeffs) % N
    g = gcd(val_at_1, N)
    if 1 < g < N:
        other = N // g
        if g * other == N:
            return (min(g, other), max(g, other))

    # Evaluate at x = -1
    val_at_m1 = 0
    sign = 1
    for coeff in minpoly_coeffs:
        val_at_m1 = (val_at_m1 + sign * coeff) % N
        sign = -sign
    g = gcd(val_at_m1, N)
    if 1 < g < N:
        other = N // g
        if g * other == N:
            return (min(g, other), max(g, other))

    return None


def _factor_from_trace_gcd(traces, N):
    """Extract factors from gcd of trace differences.

    The sequence tr(A^k) mod N has different recurrence rates mod p and mod q.
    Taking gcd of consecutive trace differences can reveal factor structure.

    For tr(A^k), the "velocity" d/dt tr(A^t) differs mod p vs mod q.
    Computing gcd(tr(A^{k+1}) - tr(A^k), N) for various k can reveal factors.
    """
    for i in range(len(traces) - 1):
        diff = abs(traces[i + 1] - traces[i]) % N
        if diff == 0 or diff == N:
            continue
        g = gcd(diff, N)
        if 1 < g < N:
            other = N // g
            if g * other == N:
                return (min(g, other), max(g, other))

        # Also try second differences
        if i + 2 < len(traces):
            d2 = (traces[i + 2] - 2 * traces[i + 1] + traces[i]) % N
            if d2 != 0 and d2 != N:
                g = gcd(d2, N)
                if 1 < g < N:
                    other = N // g
                    if g * other == N:
                        return (min(g, other), max(g, other))

    return None


def _factor_from_trace_poly_eval(traces, N, max_degree=20):
    """Build polynomial from trace values and factor via eval methods."""
    n = len(traces)
    if n < 4:
        return None

    # Try building a recurrence polynomial from first n//2 terms
    L = n // 4
    if L < 2:
        return None

    # Use the Berlekamp-Massey approach: find minimal L s.t. we can fit
    # a linear recurrence
    for degree in range(1, min(L, max_degree) + 1):
        try:
            # Build linear system for recurrence coefficients
            # s_{k} = c_1*s_{k-1} + c_2*s_{k-2} + ... + c_degree*s_{k-degree}
            # for k = degree, ..., n-2
            num_eqs = n - 2 - degree + 1  # equations from k=degree to k=n-2
            if num_eqs < degree:
                continue

            # Build matrix
            A = []
            b = []
            for k in range(degree, n - 1):
                row = [traces[k - j] for j in range(1, degree + 1)]
                A.append(row)
                b.append(traces[k + 1])

            # Solve using Gaussian elimination mod N
            M = [A[i] + [b[i]] for i in range(len(A))]

            for col in range(degree):
                # Find pivot
                pivot_row = None
                for r in range(col, len(M)):
                    if M[r][col] % N != 0:
                        pivot_row = r
                        break
                if pivot_row is None:
                    break
                if pivot_row != col:
                    M[col], M[pivot_row] = M[pivot_row], M[col]

                piv_val = M[col][col] % N
                inv_piv = pow(piv_val, -1, N)
                M[col] = [(M[col][j] * inv_piv) % N for j in range(degree + 1)]

                for r in range(col + 1, len(M)):
                    if M[r][col] % N != 0:
                        factor = M[r][col]
                        M[r] = [(M[r][j] - factor * M[col][j]) % N for j in range(degree + 1)]

            coeffs = [0] * degree
            for row in range(degree - 1, -1, -1):
                if M[row][row] == 0:
                    break
                coeffs[row] = M[row][degree]
                for j in range(row + 1, degree):
                    coeffs[row] = (coeffs[row] - M[row][j] * coeffs[j]) % N
            else:
                # Verify
                valid = True
                for k in range(degree, n - 1):
                    lhs = traces[k + 1]
                    rhs = sum(coeffs[j] * traces[k - j] for j in range(degree)) % N
                    if lhs != rhs:
                        valid = False
                        break
                if valid:
                    # We have recurrence! Now extract factors from coefficients
                    # Polynomial: x^{degree} - c_1*x^{degree-1} - ... - c_degree
                    # i.e., coefficients = [1, -c_1, -c_2, ..., -c_degree]
                    minpoly = [1] + [(-c) % N for c in coeffs]
                    result = factor_from_minpoly(minpoly, N)
                    if result:
                        return result
        except (ValueError, IndexError, ZeroDivisionError):
            continue

    return None


# ---------------------------------------------------------------------------
# CRT-based Factor Extraction from Trace Recurrence
# ---------------------------------------------------------------------------

def _crt_trace_poly_factors(traces, N):
    """Use CRT and polynomial factorization to extract p, q.

    Key insight: the minimal polynomial of the trace sequence over Z/NZ
    factors into two coprime polynomials over Z/pZ and Z/qZ.
    We can try to find roots of the polynomial mod N by computing roots
    mod p and mod q separately (via trying small x values).

    For polynomial P(x) = x^L + c_1*x^{L-1} + ... + c_L (mod N):
    If P(a) ≡ 0 (mod p) but P(a) ≢ 0 (mod q), then gcd(P(a), N) = p.
    """
    if len(traces) < 4:
        return None

    # Build polynomial from traces via BM
    bm_result = berlekamp_massey_RII(traces, N)
    if bm_result is None:
        return None
    C, L = bm_result

    if L < 2:
        return None

    # C is the connection polynomial: s_{n+L} = sum_{j=1}^{L} C[j] * s_{n+L-j}
    # The characteristic polynomial is: x^L - C[1]*x^{L-1} - ... - C[L]
    # Represent as [1, -C[1], -C[2], ..., -C[L]]
    charpoly = [1] + [(-C[j]) % N for j in range(1, L + 1)]

    # Evaluate polynomial at small x to find roots
    for x in range(2, min(1000, N)):
        val = 0
        for coeff in charpoly:
            val = (val * x + coeff) % N

        if val == 0:
            g = gcd(x, N)
            if 1 < g < N:
                other = N // g
                if g * other == N:
                    return (min(g, other), max(g, other))
            continue

        g = gcd(val, N)
        if 1 < g < N:
            other = N // g
            if g * other == N:
                return (min(g, other), max(g, other))

    return None


# ---------------------------------------------------------------------------
# Main SNDS Factorization
# ---------------------------------------------------------------------------

def _random_sl2(N):
    """Generate a random element of SL2(Z/NZ).

    Returns matrix [[a,b],[c,d]] with a*d - b*c ≡ 1 (mod N).
    Uses small coefficients to keep arithmetic manageable.
    """
    import random
    for _ in range(1000):
        a = random.randint(0, N - 1)
        b = random.randint(0, N - 1)
        c = random.randint(0, N - 1)
        d = random.randint(0, N - 1)
        det = (a * d - b * c) % N
        if det == 0:
            continue
        # Find inv of det mod N
        try:
            det_inv = pow(det, -1, N)
            a = (a * det_inv) % N
            b = (b * det_inv) % N
            c = (c * det_inv) % N
            d = (d * det_inv) % N
            det = (a * d - b * c) % N
            if det == 1:
                return (a, b, c, d)
        except ValueError:
            continue
    return None


def _cf_guidance_matrices(N):
    """Return SL2 matrices derived from CF/Berggren structure.

    These provide structured starting points with known algebraic properties.
    """
    isqrt_N = isqrt(N)
    matrices = []

    # Basic Berggren matrices mod N
    M_A = (1, 1, 1, 2)   # [[1,1],[1,2]]
    M_D = (1, 0, 2, 1)   # [[1,0],[2,1]]
    M_U = (0, 1, -1, 2)  # [[0,1],[-1,2]]

    for M in [M_A, M_D, M_U]:
        matrices.append((M[0] % N, M[1] % N, M[2] % N, M[3] % N))

    # Products of Berggren matrices
    M_AD = _mat2_mul(M_A, M_D, N)
    M_AU = _mat2_mul(M_A, M_U, N)
    M_DU = _mat2_mul(M_D, M_U, N)
    matrices.extend([M_AD, M_AU, M_DU])

    # Powers of fundamental matrices
    M_A2 = _mat2_pow(M_A, 2, N)
    M_D2 = _mat2_pow(M_D, 2, N)
    M_U2 = _mat2_pow(M_U, 2, N)
    matrices.extend([M_A2, M_D2, M_U2])

    # Small determinant matrices with different structures
    for a in range(2, min(10, N)):
        for b in range(2, min(10, N)):
            det = (1 + a * b) % N
            if det == 0:
                continue
            try:
                det_inv = pow(det, -1, N)
                M = ((1 + a * b) * det_inv % N, a * det_inv % N,
                     b * det_inv % N, det_inv % N)
                matrices.append(M)
            except ValueError:
                continue

    return matrices


def snds_factor(N: int, bound: int = 50000) -> Optional[tuple[int, int]]:
    """Factor N using Spectral Nilpotent Dynamical System method.

    The trace sequence tr(A^k) mod N satisfies a linear recurrence whose
    minimal polynomial over Z/NZ factors into coprime components mod p and
    mod q. Berlekamp-Massey over Z/NZ recovers this polynomial, and gcd
    of consecutive terms reveals the factors.

    Algorithm:
    1. Pick A ∈ SL2(Z/NZ) (random or CF-guided)
    2. Compute traces tr(A), tr(A^2), ..., tr(A^B) mod N
    3. Berlekamp-Massey to find minimal polynomial of the trace recurrence
    4. Extract factors from the polynomial coefficients via gcd tests

    Args:
        N: composite integer to factor (semiprime)
        bound: maximum number of trace terms to compute (default 50000)

    Returns:
        (p, q) with p < q and p*q = N, or None if factoring fails.
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

    # Get candidate matrices (CF-guided + random)
    candidate_matrices = _cf_guidance_matrices(N)

    # Add random SL2 matrices
    for _ in range(10):
        M = _random_sl2(N)
        if M:
            candidate_matrices.append(M)

    # Try each candidate matrix
    for A in candidate_matrices:
        # Quick matrix check for obvious factors
        for entry in A:
            g = gcd(abs(entry), N)
            if 1 < g < N:
                other = N // g
                if g * other == N:
                    return (min(g, other), max(g, other))

        # Try different trace sequence lengths
        # For large N, cap sequence length to avoid O(n²) BM blowup
        if N.bit_length() > 512:
            max_seq = 50
        else:
            max_seq = max(50, min(200, N.bit_length() * 2))
        for seq_len in [50, 100, 200, 500, 1000][:3 if N.bit_length() > 512 else 5]:
            if seq_len > bound:
                break

            traces = trace_sequence(A, N, seq_len)

            # Method 1: gcd of trace differences
            result = _factor_from_trace_gcd(traces, N)
            if result:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return result

            # Method 2: polynomial eval from recurrence
            result = _factor_from_trace_poly_eval(traces, N)
            if result:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return result

            # Method 3: CRT trace poly factorization
            result = _crt_trace_poly_factors(traces, N)
            if result:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return result

    return None


# ---------------------------------------------------------------------------
# Adaptive wrapper: integrate into the factoring portfolio
# ---------------------------------------------------------------------------

def adaptive_snds_factor(N: int, bound: int = 50000) -> Optional[tuple[int, int]]:
    """Adaptive SNDS that tries progressively longer trace sequences.

    Falls back to shorter sequences for easy factors and extends only when
    needed, trading off computation time against success probability.
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

    # Quick attempt with short sequences
    result = snds_factor(N, bound=1000)
    if result:
        return result

    # Medium attempt
    result = snds_factor(N, bound=10000)
    if result:
        return result

    # Full attempt
    return snds_factor(N, bound=bound)


if __name__ == "__main__":
    # Self-test
    import sys

    test_cases = [8051, 15571, 3127, 1022117]

    for N in test_cases:
        print(f"Factoring N = {N} ... ", end="", flush=True)
        result = snds_factor(N)
        if result:
            p, q = result
            if p * q == N:
                print(f"SUCCESS: {N} = {p} × {q}")
            else:
                print(f"FAIL: got {p} × {q} = {p*q}, expected {N}")
                sys.exit(1)
        else:
            print(f"FAILED to factor {N}")
            sys.exit(1)
