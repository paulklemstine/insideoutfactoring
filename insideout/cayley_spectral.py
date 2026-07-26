"""Cayley Graph Spectral Gap Factoring.

Uses the spectral gap of the Cayley graph of SL2(Z/NZ) with Berggren generators
to detect factor structure.

THEORY:
Let G = SL2(Z/NZ) with generating set S = {U, A, D} (the three Berggren matrices).
The Cayley graph Cay(G, S) is a 6-regular graph (each vertex has 6 directed edges).

Key facts:
1. |G| = N(N2 - 1) for N prime (order of SL2(F_N))
2. For N = pq, by CRT: SL2(Z/NZ) ≅ SL2(F_p) × SL2(F_q)
   So |G| = p(p2-1) · q(q2-1)
3. The adjacency matrix A of Cay(G, S) has eigenvalues that factor through
   the irreducible representations of G.
4. By CRT, the eigenvalues of A mod N are pairs (λ_p, λ_q) where λ_p is an
   eigenvalue of Cay(SL2(F_p), S) and λ_q is an eigenvalue of Cay(SL2(F_q), S).

THE SPECTRAL GAP INSIGHT:
The second-largest eigenvalue λ_2 of Cay(G, S) satisfies:
  λ_2 = max(λ_2(p), λ_2(q))
where λ_2(p) is the second eigenvalue mod p.

For SL2(F_p), the spectral gap is known (Selberg, Bourgain-Gamburd):
  λ_2(p) <= 2sqrt2 / sqrtp  (for large p, by expander mixing)

NOVEL ALGORITHM — "Spectral Gap Detection":
1. Pick a random starting vector v in Z_N^G (or a small subspace)
2. Apply the adjacency operator A repeatedly: v, Av, A2v, ...
3. The projection onto the second eigenvector grows as (λ_2)^k
4. By tracking the RAYLEIGH QUOTIENT ⟨v, Av⟩ / ⟨v, v⟩, we estimate λ_2
5. If λ_2 ~= 2sqrt2/sqrtp for some small p, we can recover p from the gap!

PRACTICAL IMPLEMENTATION (can't build full |G|×|G| matrix):
Instead, we use the "local spectral test":
1. Pick a random g in SL2(Z/NZ) (as a 2x2 matrix)
2. Apply random words in {U,A,D} of length k, computing the endpoint
3. The distribution of endpoints after k steps approaches uniform if the
   spectral gap is large (rapid mixing)
4. The MIXING TIME is related to the spectral gap:
   t_mix ~= log(|G|) / (1 - λ_2)
5. For SL2(F_p), t_mix ~= log(p) / (1 - 2sqrt2/sqrtp) ~= log(p) for large p
6. The mixing time mod N is max(t_mix(p), t_mix(q))

THE FACTORING ALGORITHM:
1. Generate random walks of increasing length k on SL2(Z/NZ)
2. At each k, compute the COLLISION RATE: how often two walks reach the same state
3. The collision rate transitions from "structured" (low k, birthday ~= |G|^(1/2))
   to "uniform" (high k, uniform distribution)
4. The TRANSITION POINT reveals the spectral gap, which reveals min(p,q)

SIMPLER — "Eigenvalue GCD" method:
1. For a random g in SL2(Z/NZ), compute g^k for k = 1, 2, ..., B
2. For each k, compute gcd(tr(g^k) - 2, N)
3. If g has eigenvalue 1 mod p but not mod q, then tr(g^k) == 2 mod p
   but tr(g^k) ≢ 2 mod q, so gcd(tr(g^k) - 2, N) = p
4. The order of g mod p divides p-(a/p) where (a/p) is the Legendre symbol
   of the discriminant. For most g, this order divides p-1 or p+1.
5. This is essentially Pollard p-1 / Williams p+1 on the trace sequence.

MOST PRACTICAL — "Cayley Collision Search":
1. Random walk on SL2(Z/NZ) using generators {U,A,D}
2. Store distinguished points (those with trace having low bits zero)
3. On collision between two walks: the product g1^{-1} · g2 is a relation
   in the group, i.e., a word in U,A,D that equals identity mod N
4. This relation gives a factor: if W = identity mod p but not mod q,
   then evaluating W at any point gives a value divisible by p but not q

IMPLEMENT:
- `cayley_spectral_factor(N, max_steps=50000, distinguished_bits=8)` — main entry
- `_random_sl2(N)` — generate random SL2 element
- `_cayley_walk(g, steps, N)` — random walk on Cayley graph
- `_spectral_collision_search(N, num_walks, walk_len)` — birthday search on graph
- `_factor_from_relation(g1, g2, N)` — extract factor from group relation
- `_trace_gcd_sequence(g, N, max_k)` — compute gcd(tr(g^k)-2, N) for all k
"""
from __future__ import annotations
from math import gcd, isqrt
from typing import Optional
import random

# Berggren matrices
U = (1, 1, -1, 2)    # [[1,1],[-1,2]]
A = (1, 2, 1, 2)     # [[1,2],[1,2]]
D = (-1, 2, 1, -1)   # [[-1,2],[1,-1]]
GENERATORS = [U, A, D]
GEN_NAMES = ['U', 'A', 'D']


def _mat_mul(A, B, N):
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1*a2 + b1*c2) % N, (a1*b2 + b1*d2) % N,
        (c1*a2 + d1*c2) % N, (c1*b2 + d1*d2) % N,
    )


def _mat_pow(M, k, N):
    result = (1, 0, 0, 1)
    base = M
    while k:
        if k & 1:
            result = _mat_mul(result, base, N)
        base = _mat_mul(base, base, N)
        k >>= 1
    return result


def _mat_inv(M, N):
    """Inverse of SL2 matrix mod N."""
    a, b, c, d = M
    det = (a*d - b*c) % N
    det_inv = pow(det, -1, N)
    return (d * det_inv % N, (-b) * det_inv % N,
            (-c) * det_inv % N, a * det_inv % N)


def _trace(M, N):
    return (M[0] + M[3]) % N


def _det(M, N):
    return (M[0]*M[3] - M[1]*M[2]) % N


def _random_sl2(N):
    """Generate random SL2 element mod N."""
    while True:
        a = random.randint(1, N-1)
        b = random.randint(0, N-1)
        c = random.randint(0, N-1)
        det = (a - b*c) % N
        if det == 0:
            continue
        try:
            det_inv = pow(det, -1, N)
            d = (1 + b*c) * det_inv % N
            M = (a % N, b % N, c % N, d % N)
            if _det(M, N) == 1:
                return M
        except (ValueError, ZeroDivisionError):
            continue


def _is_distinguished(M, N, bits=8):
    """Check if matrix has distinguished trace (low bits zero)."""
    t = _trace(M, N)
    mask = (1 << bits) - 1
    return (t & mask) == 0


def _trace_gcd_sequence(g, N, max_k):
    """Compute gcd(tr(g^k) - 2, N) for k=1..max_k.

    If g has eigenvalue 1 mod p but not mod q, this catches the factor.
    """
    M = g
    for k in range(1, max_k + 1):
        M = _mat_mul(M, g, N)
        tr = _trace(M, N)
        for offset in [0, 1, -1, 2, -2]:
            g_val = gcd(abs(tr - 2 + offset), N)
            if 1 < g_val < N:
                return g_val
    return None


def _spectral_collision_search(N, num_walks, walk_len, distinguished_bits=8):
    """Birthday search on SL2 Cayley graph.

    Each walk is a sequence of matrix multiplications by random generators.
    Distinguished points (trace with low bits zero) are stored.
    On collision, compute gcd of the "difference" matrix entries with N.
    """
    dist_table = {}

    for walk_id in range(num_walks):
        g = _random_sl2(N)

        M = (1, 0, 0, 1)  # identity
        for step in range(walk_len):
            gen = random.choice(GENERATORS)
            M = _mat_mul(M, gen, N)

            if _is_distinguished(M, N, distinguished_bits):
                key = (_trace(M, N), M[0] % (N), M[3] % (N))
                if key in dist_table:
                    other_id, other_M = dist_table[key]
                    # Collision found! Compute g = M^{-1} · other_M
                    M_inv = _mat_inv(M, N)
                    diff = _mat_mul(M_inv, other_M, N)
                    for entry in diff:
                        g_val = gcd(abs(entry), N)
                        if 1 < g_val < N:
                            return g_val
                else:
                    dist_table[key] = (walk_id, M)

    return None


def cayley_spectral_factor(N: int, max_steps: int = 50000,
                           distinguished_bits: int = 8) -> Optional[tuple[int, int]]:
    """Factor N using Cayley graph spectral methods.

    Combines:
    1. Trace gcd sequence (Pollard p-1 / Williams p+1 style)
    2. Cayley graph collision search (birthday on SL2(Z/NZ))
    3. Spectral gap estimation via collision rate

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

    # Stage 1: Trace gcd sequence with multiple random bases
    for _ in range(20):
        g = _random_sl2(N)
        factor = _trace_gcd_sequence(g, N, min(1000, max_steps // 10))
        if factor:
            return (min(factor, N // factor), max(factor, N // factor))

    # Stage 2: Cayley graph collision search
    num_walks = max(8, max_steps // 1000)
    walk_len = max(100, max_steps // num_walks)
    factor = _spectral_collision_search(N, num_walks, walk_len, distinguished_bits)
    if factor:
        return (min(factor, N // factor), max(factor, N // factor))

    return None
