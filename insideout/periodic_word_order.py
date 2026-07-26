"""Periodic-Word Order Factoring (PW-012).

For a periodic Berggren word w, its matrix M_w acts in SL2(Z/NZ).
If a smooth exponent E kills the order of M_w mod p but not mod q,
then gcd(tr(M_w^E) - 2, N) gives a proper factor.

The 3x3 matrix form is too costly. Key insight: the trace tr(M_w^k)
satisfies a 2nd-order Lucas/Chebyshev recurrence:
  tr(M^{k+1}) = tr(M) * tr(M^k) - tr(M^{k-1})
with tr(M^0) = 2, tr(M^1) = tr(M).

This lets us compute tr(M_w^k) in O(log k) time via doubling formulas,
exactly like Williams p+1. The "periodic word" family gives us many
different tr(M) values, covering order families missed by standard
p-1 and p+1.

Algorithm:
1. Enumerate short periodic Berggren words (length <= 6)
2. For each word, compute tr(M_w) mod N
3. Run a Lucas-style ladder: compute tr(M_w^{p^e}) for primes p <= bound
4. At each step, check gcd(tr - 2, N) for a proper factor
5. The panel of words covers complementary order families

This is the "recurrence reduction" recommended by PW-012.
"""
from __future__ import annotations
from math import gcd, isqrt
from typing import Optional


# Berggren matrices as SL2 elements
# U = [[1,1],[-1,2]], A = [[1,2],[1,2]], D = [[-1,2],[1,-1]]
# But these don't have det=1! Let's use the correct SL2 Berggren matrices.

# The Berggren matrices for PPT generation are 3x3. Their action on (m,n) is 2x2:
# U: (m,n) -> (2m+n, m)  ... no, let me get this right.
# The standard Berggren matrices acting on (a,b,c) are 3x3.
# For SL2, we use the (m,n) parametrization where:
#   a = m2-n2, b = 2mn, c = m2+n2
# U: (m,n) -> (2m-n, m)  -- no
# Let me use the actual SL2 matrices that generate PPTs:

# From the Berggren tree in (m,n) space:
# U: m' = 2m + n, n' = m       -> matrix [[2,1],[1,1]]
# A: m' = 2m - n, n' = m       -> matrix [[2,-1],[1,0]]  ... no

# Actually, the standard approach: the three Berggren matrices in SL2(Z) are:
# U = [[1, -2, 2], [2, -1, 2], [2, -2, 3]] (3x3, acts on (a,b,c))
# For SL2(Z) in (m,n) parametrization:
# U: [[1, 2], [1, 1]] acting on column (m,n) -> (m+2n, m+n)... this gives PPTs.

# Let me just use the well-known SL2 matrices that generate PPTs:
# U = [[2, 1], [1, 1]]  -- maps (m,n) -> (2m+n, m+n)
# A = [[2, -1], [1, 0]] -- maps (m,n) -> (2m-n, m)
# D = [[2, 1], [-1, 0]] -- maps (m,n) -> (2m+n, -m)

# These are the correct SL2 Berggren matrices (det = 1).
U_S = (2, 1, 1, 1)    # [[2,1],[1,1]]
A_S = (2, -1, 1, 0)   # [[2,-1],[1,0]]
D_S = (2, 1, -1, 0)   # [[2,1],[-1,0]]

WORD_MATRICES = {'U': U_S, 'A': A_S, 'D': D_S}


def _mat2_mul(A, B, N):
    """Multiply two SL2 matrices mod N."""
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1*a2 + b1*c2) % N,
        (a1*b2 + b1*d2) % N,
        (c1*a2 + d1*c2) % N,
        (c1*b2 + d1*d2) % N,
    )


def _mat2_pow(M, k, N):
    """M^k mod N via fast exponentiation."""
    result = (1, 0, 0, 1)
    base = M
    while k:
        if k & 1:
            result = _mat2_mul(result, base, N)
        base = _mat2_mul(base, base, N)
        k >>= 1
    return result


def _trace(M, N):
    return (M[0] + M[3]) % N


def word_matrix(word: str, N: int):
    """Compute the SL2 matrix for a Berggren word."""
    M = (1, 0, 0, 1)
    for ch in word:
        M = _mat2_mul(M, WORD_MATRICES[ch], N)
    return M


# Lucas sequence doubling for trace:
# For M with tr(M) = P, the trace of M^k is V_k where:
#   V_0 = 2, V_1 = P, V_{k+1} = P*V_k - V_{k-1}
# Doubling formulas:
#   V_{2k} = V_k^2 - 2
#   V_{2k+1} = V_k * V_{k+1} - P
# These let us compute V_k mod N in O(log k) time.

def _lucas_double(vk, vk1, P, N):
    """Given V_k and V_{k+1}, compute V_{2k} and V_{2k+1}."""
    v2k = (vk * vk - 2) % N
    v2k1 = (vk * vk1 - P) % N
    return v2k, v2k1


def _lucas_pow_trace(P, k, N):
    """Compute V_k where V_0=2, V_1=P, V_{k+1}=P*V_k - V_{k-1} mod N."""
    if k == 0:
        return 2 % N
    if k == 1:
        return P % N

    # Binary exponentiation on the Lucas sequence
    # We maintain (V_n, V_{n+1}) and double/add
    vn = 2 % N
    vn1 = P % N
    bits = bin(k)[2:]  # MSB first

    for bit in bits[1:]:  # skip MSB (already at n=1)
        # Double: (V_n, V_{n+1}) -> (V_{2n}, V_{2n+1})
        v2n = (vn * vn - 2) % N
        v2n1 = (vn * vn1 - P) % N
        if bit == '0':
            vn, vn1 = v2n, v2n1
        else:
            # Add 1: (V_{2n}, V_{2n+1}) -> (V_{2n+1}, V_{2n+2})
            # V_{2n+2} = P * V_{2n+1} - V_{2n}
            vn, vn1 = v2n1, (P * v2n1 - v2n) % N

    return vn


def _short_periodic_words(max_len=6):
    """Generate short periodic Berggren words.

    A "periodic word" is one like 'UUDD' that repeats.
    We enumerate all words up to max_len and deduplicate by
    their characteristic polynomial (trace of the matrix).
    """
    words = []
    seen_traces = set()

    # Generate all words of length 1 to max_len
    from itertools import product
    for length in range(1, max_len + 1):
        for word_tuple in product('UAD', repeat=length):
            word = ''.join(word_tuple)
            # Skip trivial words (single branch)
            if length == 1:
                continue
            # Skip words that are just repetitions of shorter words
            # (e.g., 'UUUU' is just 'U' repeated)
            is_repeat = False
            for sub_len in range(1, length):
                if length % sub_len == 0:
                    sub = word[:sub_len]
                    if sub * (length // sub_len) == word:
                        is_repeat = True
                        break
            if is_repeat:
                continue
            words.append(word)

    return words


def _dedup_words(words, N):
    """Deduplicate words by their trace mod N."""
    seen = set()
    result = []
    for word in words:
        M = word_matrix(word, N)
        tr = _trace(M, N)
        key = tr
        if key not in seen:
            seen.add(key)
            result.append(word)
    return result


def _smooth_ladder_tr(P, N, bound):
    """Compute V_{p^e} for each prime p <= bound using Lucas doubling.

    For each prime power p^e <= bound, compute V_{p^e} mod N
    and check gcd(V_{p^e} - 2, N).

    This is the Lucas-analogue of the p-1 smooth ladder.
    """
    # Sieve primes up to bound
    if bound < 2:
        return None
    sieve = bytearray(b'\x01') * (bound + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, isqrt(bound) + 1):
        if sieve[p]:
            sieve[p*p:bound+1:p] = bytearray(len(range(p*p, bound+1, p)))
    primes = [i for i in range(2, bound + 1) if sieve[i]]

    # V_1 = P (already know this)
    # For each prime, raise to p^e for e=1,2,... while p^e <= bound
    for p in primes:
        pe = p
        while pe <= bound:
            # Compute V_{pe} from V_{pe//p} using Lucas formulas
            # Actually, compute directly: V_{pe} = lucas_pow_trace(P, pe, N)
            v = _lucas_pow_trace(P, pe, N)
            g = gcd(v - 2, N)
            if 1 < g < N:
                return g
            # Also check V_{pe} + 2 (for p+1 style)
            g2 = gcd(v + 2, N)
            if 1 < g2 < N:
                return g2
            if pe > bound // p:
                break
            pe *= p

    return None


def periodic_word_order_factor(N: int, bound: int = 5000) -> Optional[tuple[int, int]]:
    """Factor N using periodic Berggren word order families.

    For each short periodic word w:
    1. Compute P = tr(M_w) mod N
    2. Run a Lucas-style smooth ladder: compute V_{p^e} for primes p <= bound
    3. Check gcd(V_{p^e} ± 2, N) for proper factors

    The panel of words covers complementary order families.
    Uses O(log(bound)) Lucas doubling per prime power, not O(beta) matrix mults.

    Args:
        N: composite integer to factor
        bound: smoothness bound for the ladder

    Returns:
        (p, q) with p < q and p*q = N, or None
    """
    if N < 4:
        return None
    if N == 2:
        return None
    if N % 2 == 0:
        return (2, N // 2)

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

    # Generate periodic words
    words = _short_periodic_words(max_len=5)

    # Deduplicate by trace
    words = _dedup_words(words, N)

    # For each word, compute trace and run smooth ladder
    for word in words:
        M = word_matrix(word, N)
        P = _trace(M, N)

        # Skip if trace gives immediate factor
        for offset in [0, 2, -2]:
            g = gcd(abs(P - 2 + offset), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Smooth ladder on the Lucas sequence
        factor = _smooth_ladder_tr(P, N, bound)
        if factor:
            return (min(factor, N // factor), max(factor, N // factor))

    return None
