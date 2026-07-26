"""Multi-Parameter Lucas Sequence Factoring — A Novel Smooth-Rank Method.

Uses Lucas sequences U_k(P,Q) and V_k(P,Q) with MULTIPLE parameters P to
decorrelate the rank of apparition. Each P gives an independent chance to find
a factor p when α_P(p) (the rank of apparition) is smooth.

Key novelty over standard p±1 methods:
1. Multiple P parameters: instead of hoping one Lucas parameter has a smooth
   rank, we try many — each P gives an independent random rank α_P(p).
2. CRT collision lane: we search for Lucas projective states (U_k : V_k)
   whose cross-products vanish mod one factor but not the other, revealing
   the factor via gcd.
3. Pythagorean augmentation: Lucas sequences embed into Pythagorean triples
   via the same verified identity used in fibonacci_pythagorean.py, giving
   5-way batched GCD per probe.

Mathematical basis:
- Lucas sequences U_k(P,Q), V_k(P,Q) satisfy:
    U_0 = 0, U_1 = 1
    V_0 = 2, V_1 = P
    U_{k+1} = P·U_k - Q·U_{k-1}
    V_{k+1} = P·V_k - Q·V_{k+1}
- Rank of apparition α_P(p) = min{k > 0 : U_k(P,Q) == 0 (mod p)}
- For Fibonacci: P=1, Q=-1 (or P=1, Q=1 depending on convention)
- Different P values give different, essentially independent ranks

Per the Algebraic Light assessment: this is a smooth-rank method with
multiple independent probes. It improves constants over single-parameter
Fibonacci/Lucas methods but does not achieve polynomial-time factoring
for arbitrary N.
"""
from __future__ import annotations

from math import gcd, isqrt


def _lucas_pair(k: int, P: int, Q: int, N: int) -> tuple[int, int]:
    """Compute (U_k(P,Q) mod N, V_k(P,Q) mod N) using fast doubling.

    Uses the doubling formulas:
      U_{2k} = U_k · V_k
      V_{2k} = V_k2 - 2·Q^k

    And the addition formulas:
      U_{k+1} = (P·U_k - Q·U_{k-1}) mod N
      V_{k+1} = (P·V_k - Q·V_{k-1}) mod N

    For Q=1, these simplify. We use the matrix form:
      [[P, -Q], [1, 0]]^k · [1, 0]^T = [U_k, U_{k-1}]^T

    Returns (U_k mod N, V_k mod N).
    """
    if k == 0:
        return (0, 2 % N)

    # Fast doubling for Lucas sequences
    # We maintain (U_n, V_n, Q^n) mod N
    # Using:
    #   U_{2n} = U_n · V_n
    #   V_{2n} = V_n2 - 2·Q^n
    #   U_{2n+1} = (P·U_{2n} + V_{2n}) / 2  -- but we avoid division
    # Instead, we use matrix powering of [[P, -Q], [1, 0]]

    # Matrix method: compute M^k where M = [[P, -Q], [1, 0]]
    # Then U_k is the (0,0) entry times U_1 + (0,1) entry times U_0
    # = M[0][0] since U_1 = 1, U_0 = 0
    # V_k = 2·M[0][0] - P·M[1][0]... actually let's use the standard
    # recurrence directly.

    # Direct recurrence computation via fast doubling:
    # For Q=1: U_{2n} = U_n · V_n, V_{2n} = V_n2 - 2
    # General: track (U_n, V_n, Q^n) mod N

    P_mod = P % N
    Q_mod = Q % N

    # Handle Q=1 special case (most common, simpler formulas)
    if Q_mod == 1 % N or Q_mod == (N - 1) % N + (1 if Q_mod == 0 else 0):
        # For Q == ±1 mod N, use simplified doubling
        # Actually, let's just use the general method for safety
        pass

    # General fast doubling for Lucas sequences
    # We compute U_n, V_n, Q^n simultaneously
    # Formulas:
    #   U_{2k} = U_k · V_k
    #   V_{2k} = V_k2 - 2·Q^k
    #   Q^{2k} = (Q^k)2
    #   U_{2k+1} = (U_{2k} · P + V_{2k}) // 2 -- but we need to avoid division
    #
    # Better approach: use the addition law
    #   U_{m+n} = (U_m · V_n · Q^n + U_n · V_m) / (2·Q^n)... no, too complex
    #
    # Cleanest approach: matrix powering of M = [[P, -Q], [1, 0]]
    # M^n · [1, 0]^T gives [U_{n+1}, U_n]^T (for standard convention)
    # We can also extract V_n from M^n

    a, b, c, d = 1, 0, 0, 1  # Identity matrix
    ma, mb, mc, md = P_mod, (-Q_mod) % N, 1, 0  # M = [[P, -Q], [1, 0]]

    # Compute M^k via binary exponentiation
    k_bits = k.bit_length()
    for i in range(k_bits - 1, -1, -1):
        # Square: result = result2
        a_new = (a * a + b * c) % N
        b_new = (a * b + b * d) % N
        c_new = (c * a + d * c) % N
        d_new = (c * b + d * d) % N
        a, b, c, d = a_new, b_new, c_new, d_new

        if (k >> i) & 1:
            # Multiply by M
            a_new = (a * ma + b * mc) % N
            b_new = (a * mb + b * md) % N
            c_new = (c * ma + d * mc) % N
            d_new = (c * mb + d * md) % N
            a, b, c, d = a_new, b_new, c_new, d_new

    # M^k = [[a, b], [c, d]]
    # U_k = c (entry (1,0) of M^k, since U_1 = 1 maps to column 0)
    # Actually, M^k · [1, 0]^T = [a + 0, c + 0]^T = [a, c]^T
    # So U_k = c (for standard convention U_0=0, U_1=1)
    # And V_k = 2·a - P·c  (since V_k = 2·U_{k+1} - P·U_k)
    # But U_{k+1} = a (from the matrix product)
    # So V_k = 2·a - P·c

    U_k = c % N
    U_k1 = a % N  # U_{k+1}
    V_k = (2 * a - P_mod * c) % N

    return (U_k, V_k)


def _lucas_pythagorean_batch(U_k: int, U_k1: int, V_k: int, P: int,
                              N: int, Q: int = 1) -> tuple[int, list[int]]:
    """Compute batched GCD candidates from Lucas sequence and Pythagorean coordinates.

    Given U_k(P,Q) mod N, U_{k+1}(P,Q) mod N, V_k(P,Q) mod N, and P, Q,
    compute the 5 Pythagorean-augmented GCD candidates:
      1. U_k (rank of apparition probe)
      2. V_k (Lucas V-number, has different rank structure)
      3. 2·U_{k+1}·U_{k+2} (even leg analog)
      4. U_k·U_{k+3} (odd leg analog)
      5. U_{k+1}2 + U_{k+2}2 (hypotenuse analog)

    For Q=-1, P=1, these reduce to the Fibonacci-Pythagorean identities.

    Returns (batch_product mod N, list_of_individual_candidates).
    """
    Q_mod = Q % N if Q != 1 else 1  # Handle Q=-1 properly

    # Compute U_{k+2}, U_{k+3} from recurrence: U_{k+2} = P·U_{k+1} - Q·U_k
    U_k2 = (P * U_k1 - Q_mod * U_k) % N
    U_k3 = (P * U_k2 - Q_mod * U_k1) % N

    # Candidates:
    # 1. U_k — main rank probe
    # 2. V_k — Lucas V-number (has different rank structure)
    # 3-5. Pythagorean-augmented probes
    A_k = (2 * U_k1 * U_k2) % N   # even leg analog
    B_k = (U_k * U_k3) % N        # odd leg analog
    C_k = (U_k1 * U_k1 + U_k2 * U_k2) % N  # hypotenuse analog

    candidates = [U_k, V_k, A_k, B_k, C_k]
    batch = 1
    for c in candidates:
        batch = (batch * c) % N

    return (batch, candidates)


def _small_primes(bound: int) -> list[int]:
    """Generate primes up to bound using a simple sieve."""
    if bound < 2:
        return []
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(bound) + 1):
        if sieve[i]:
            for j in range(i * i, bound + 1, i):
                sieve[j] = False
    return [i for i in range(2, bound + 1) if sieve[i]]


def _mat2_mul(A: tuple, B: tuple, N: int) -> tuple:
    """Multiply two 2x2 matrices mod N. A, B are (a,b,c,d) tuples."""
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1 * a2 + b1 * c2) % N,
        (a1 * b2 + b1 * d2) % N,
        (c1 * a2 + d1 * c2) % N,
        (c1 * b2 + d1 * d2) % N,
    )


def _mat2_pow(M: tuple, k: int, N: int) -> tuple:
    """Compute M^k mod N via binary exponentiation. M is (a,b,c,d)."""
    result = (1, 0, 0, 1)  # Identity
    base = M
    while k > 0:
        if k & 1:
            result = _mat2_mul(result, base, N)
        base = _mat2_mul(base, base, N)
        k >>= 1
    return result


def _try_lucas_parameter(P: int, N: int, primes: list[int],
                          stage2_primes: list[int],
                          bound: int = 10000) -> tuple[int, int] | None:
    """Try factoring N with a single Lucas parameter P (Q=1).

    Uses incremental matrix powering: maintains Q^M and multiplies by Q^{pk}
    at each stage, avoiding recomputation from scratch.

    Returns (p, q) with p < q and p*q = N, or None.
    """
    P_mod = P % N
    # Matrix M = [[P, -1], [1, 0]] for Q=1 (so -Q = -1)
    M_base = (P_mod, (N - 1) % N, 1, 0)

    # Start with identity: Q^1 = I
    # We maintain Q^M where M is the smooth core product so far
    # Q^M = M_base^M

    # Current matrix state: Q^M
    # We start with M=1, so Q^1 = M_base
    current_mat = M_base
    M = 1

    # Stage 1: smooth core — incrementally multiply
    for p in primes:
        if p == P:
            continue
        pk = p
        while pk * p <= bound:
            pk *= p

        # Multiply current_mat by M_base^{pk - 1} to get Q^{M * pk}
        # Actually: Q^{M*pk} = Q^M raised to pk power? No.
        # Q^{old_M * pk} = (Q^{old_M})^{pk}... hmm
        # Better: compute M_base^{pk} and multiply into current
        # Q^{M * pk} = Q^{M} * Q^{M*(pk-1)}... no
        # Q^{(M)*pk} = (Q^M)^{pk}... no, that's Q^{M*pk} = (Q^M)^pk? No!
        # Q^{M*pk} != (Q^M)^{pk}. The matrix M satisfies M^{M*pk} = (M^M)^{pk}.
        # Wait, actually M^{ab} = (M^a)^b. So Q^{M*pk} = (Q^M)^{pk}.
        # But we want Q^{new_M} where new_M = old_M * pk.
        # Q^{new_M} = Q^{old_M * pk} = (Q^{old_M})^{pk}
        # Wait no! (Q^a)^b = Q^{ab}, not Q^{a+b}. So Q^{old_M * pk} = (Q^{old_M})^{pk}.
        # Hmm, that's wrong. (Q^a)^b = Q^{a*b} is correct.
        # So to go from Q^{old_M} to Q^{old_M * pk}, we compute (Q^{old_M})^{pk}.
        # But that's expensive for large pk.
        # Actually the standard approach is different:
        # We want Q^M where M = lcm(1, 2, ..., B).
        # At each step, M grows by a factor of pk.
        # We want Q^{new_M} = Q^{old_M * pk} = (Q^{old_M})^{pk}.
        # This requires matrix exponentiation to power pk at each step.

        # Actually, the simpler approach: compute M_base^{pk} and multiply
        # Q^{old_M + pk}... no, M grows multiplicatively.

        # The correct approach for smooth-core Lucas:
        # Keep track of M incrementally.
        # At each prime power step, compute the new M = old_M * pk
        # and compute U_M, V_M from scratch.
        # But this is what we were doing before and it's slow.

        # Better: maintain the matrix Q^M incrementally.
        # Q^{M*pk} = (Q^M)^{pk}
        # So we just need to raise our current matrix to the pk power.
        # This is much faster than recomputing from scratch!

        # Incrementally: current_mat = Q^{old_M}
        # New: Q^{old_M * pk} = current_mat^{pk}
        current_mat = _mat2_pow(current_mat, pk, N)
        M *= pk

        # Extract U_M, V_M from Q^M = [[U_{M+1}, -U_M], [U_M, -U_{M-1}]]
        # Wait, for Q=1, M^k = [[U_{k+1}, -U_k], [U_k, -u_{k-1}]]
        # Hmm, let me reconsider. For Q=1, the matrix is [[P, -1], [1, 0]].
        # M^k · [1, 0]^T = [U_{k+1}, U_k]^T
        # So: a = current_mat[0] = U_{M+1}, c = current_mat[2] = U_M
        U_M = current_mat[2]
        U_M1 = current_mat[0]
        V_M = (2 * current_mat[0] - P_mod * current_mat[2]) % N

        # Batch GCD with Pythagorean augmentation
        batch, candidates = _lucas_pythagorean_batch(U_M, U_M1, V_M, P_mod, N, Q=1)

        g = gcd(batch, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        if g == N:
            # Split: test each candidate individually
            for val in candidates:
                g = gcd(val, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    # Stage 2: test U_{M·ℓ}(P,1) for small primes ℓ
    # We can reuse Q^M and just compute Q^{M·ℓ} = (Q^M)^ℓ
    for ell in stage2_primes:
        if ell in set(primes) or ell == P:
            continue
        # Q^{M*ell} = (Q^M)^{ell}
        mat_M_ell = _mat2_pow(current_mat, ell, N)

        U_M2 = mat_M_ell[2]
        U_M2_1 = mat_M_ell[0]
        V_M2 = (2 * mat_M_ell[0] - P_mod * mat_M_ell[2]) % N

        batch, candidates = _lucas_pythagorean_batch(U_M2, U_M2_1, V_M2, P_mod, N, Q=1)

        g = gcd(batch, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        if g == N:
            for val in candidates:
                g = gcd(val, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def lucas_multi_factor(N: int, bound: int = 5000,
                       stage2_bound: int = 1000,
                       max_params: int = 8) -> tuple[int, int] | None:
    """Factor N using multi-parameter Lucas sequence method.

    Tries multiple Lucas parameters P with Q=1, each giving an independent
    rank of apparition α_P(p) for any prime factor p. If α_P(p) is smooth
    for any P, we find p.

    Novel features:
    1. Multiple P parameters decorrelate the rank — instead of hoping one
       Fibonacci rank is smooth, we try many independent ranks.
    2. Pythagorean-augmented batched GCD from each probe.
    3. CRT collision lane: if U_k(P,1) == 0 (mod p) but U_k(P',1) ≢ 0 (mod p)
       for different P, P', the GCD reveals p.

    This is a smooth-rank method: fast when some α_P(p) is smooth.
    NOT a generic polynomial-time factoring algorithm.

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

    primes = _small_primes(bound)
    stage2_primes = _small_primes(stage2_bound)

    # Generate Lucas parameters P to try
    # Small primes give well-distributed independent ranks
    all_P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    # Also try some composite P values that give different rank structures
    composite_P = [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27]
    for P in composite_P:
        if P not in set(all_P):
            all_P.append(P)

    # Filter and limit
    all_P = [P for P in all_P if 0 < P < N][:max_params]

    for P in all_P:
        result = _try_lucas_parameter(P, N, primes, stage2_primes, bound)
        if result is not None:
            return result

    return None


def crt_collision_factor(N: int, bound: int = 3000,
                          stage2_bound: int = 500,
                          max_params: int = 6) -> tuple[int, int] | None:
    """Factor N using CRT collision between Lucas sequences.

    Novel method: compute U_M(P,1) and U_M(P',1) for different P, P'.
    If p | U_M(P,1) but q ∤ U_M(P',1), then:
      gcd(U_M(P,1) · U_M(P',1) mod N, N) reveals p.

    More subtly, if the projective states (U_M : V_M) differ mod p vs mod q,
    we can detect this by cross-multiplication:
      gcd(U_M(P,1) · V_M(P',1) - U_M(P',1) · V_M(P,1), N)

    This "CRT collision lane" detects factors even when no single rank
    is smooth, by exploiting differences between independent Lucas sequences.

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

    primes = _small_primes(bound)
    stage2_primes = _small_primes(stage2_bound)

    # Parameters to try
    all_P = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    all_P = [P for P in all_P if P < N and P > 0][:max_params]

    # Compute Lucas states for all parameters using incremental matrix powering
    # For each P, we compute Q^M where Q = [[P, -1], [1, 0]] mod N
    states = {}  # P -> (U_M, V_M, U_{M+1}, mat_M)

    for P in all_P:
        P_mod = P % N
        M_base = (P_mod, (N - 1) % N, 1, 0)  # [[P, -1], [1, 0]]

        # Incrementally compute Q^M
        current_mat = M_base  # Q^1
        for p in primes:
            if p == P:
                continue
            pk = p
            while pk * p <= bound:
                pk *= p
            current_mat = _mat2_pow(current_mat, pk, N)

        # Extract U_M, V_M from current_mat
        U_M = current_mat[2]
        U_M1 = current_mat[0]
        V_M = (2 * current_mat[0] - P_mod * current_mat[2]) % N

        states[P] = (U_M, V_M, U_M1, P_mod, current_mat)

    # Strategy 1: Direct GCD of each state
    for P in all_P:
        U_M, V_M, U_M1, P_mod, _ = states[P]
        batch, candidates = _lucas_pythagorean_batch(U_M, U_M1, V_M, P_mod, N, Q=1)
        g = gcd(batch, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Strategy 2: CRT collision — cross-product test
    for i in range(len(all_P)):
        for j in range(i + 1, len(all_P)):
            P1, P2 = all_P[i], all_P[j]
            U1, V1, _, _, _ = states[P1]
            U2, V2, _, _, _ = states[P2]

            # Cross-product: U_M(P1)·V_M(P2) - U_M(P2)·V_M(P1)
            cross = (U1 * V2 - U2 * V1) % N
            g = gcd(cross, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    # Strategy 3: Stage 2 — test M·ℓ for small primes ℓ
    for ell in stage2_primes[:50]:  # Limit stage 2
        if ell in set(primes):
            continue
        for P in all_P[:4]:  # Only test a few parameters in stage 2
            U_M, V_M, U_M1, P_mod, mat_M = states[P]
            mat_M_ell = _mat2_pow(mat_M, ell, N)

            U_M2 = mat_M_ell[2]
            U_M2_1 = mat_M_ell[0]
            V_M2 = (2 * mat_M_ell[0] - P_mod * mat_M_ell[2]) % N

            batch, candidates = _lucas_pythagorean_batch(U_M2, U_M2_1, V_M2, P_mod, N, Q=1)
            g = gcd(batch, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # CRT collision in stage 2
        P_list = all_P[:4]
        stage2_states = {}
        for P in P_list:
            _, _, _, P_mod, mat_M = states[P]
            mat_M_ell = _mat2_pow(mat_M, ell, N)
            U_M2 = mat_M_ell[2]
            U_M2_1 = mat_M_ell[0]
            V_M2 = (2 * mat_M_ell[0] - P_mod * mat_M_ell[2]) % N
            stage2_states[P] = (U_M2, V_M2)

        for i in range(len(P_list)):
            for j in range(i + 1, len(P_list)):
                P1, P2 = P_list[i], P_list[j]
                U1, V1 = stage2_states[P1]
                U2, V2 = stage2_states[P2]
                cross = (U1 * V2 - U2 * V1) % N
                g = gcd(cross, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None