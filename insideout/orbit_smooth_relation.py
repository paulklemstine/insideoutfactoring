"""Orbit-to-Smooth-Relation NFS Lane.

Maps bounded-length Berggren branch words to integer norms.
Collects smooth relations and uses sparse linear algebra over GF(2)
to find a congruence of squares and extract a factor.

This is a research experiment: it may not beat established methods
for any N, but it is the only current path aimed at subexponential
asymptotic improvement for projective collision factoring.
"""
from __future__ import annotations
from math import gcd, isqrt
from typing import Optional
import random


# --------------------------------------------------------------------
# Branch matrices (same as projective_collision.py)
# --------------------------------------------------------------------

class Triple:
    """A projective triple (a, b, c) representing a point in P^2."""
    __slots__ = ('a', 'b', 'c')

    def __init__(self, a: int, b: int, c: int):
        self.a = a
        self.b = b
        self.c = c

    def __repr__(self):
        return f"Triple({self.a}, {self.b}, {self.c})"


def _apply_U(t: Triple, N: int) -> Triple:
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (a + 2*b - 2*c) % N,
        (-2*a - b + 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )


def _apply_A(t: Triple, N: int) -> Triple:
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (a + 2*b - 2*c) % N,
        (2*a + b - 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )


def _apply_D(t: Triple, N: int) -> Triple:
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (-a - 2*b + 2*c) % N,
        (2*a + b - 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )


def _apply_branch(t: Triple, N: int, branch: str) -> Triple:
    if branch == 'U':
        return _apply_U(t, N)
    elif branch == 'A':
        return _apply_A(t, N)
    else:
        return _apply_D(t, N)


# --------------------------------------------------------------------
# Norm computation
# --------------------------------------------------------------------

def norm_of_branch_word(word: str, seed: Triple, N: int) -> int:
    """Compute integer norm of a branch word applied to seed.

    The norm is ||M_word * v||^2 where v = (a, b, c) is the seed triple.
    This is always an integer because Berggren matrices are integer matrices.
    """
    t = Triple(seed.a % N, seed.b % N, seed.c % N)
    for branch in word:
        t = _apply_branch(t, N, branch)
    return t.a * t.a + t.b * t.b + t.c * t.c


# --------------------------------------------------------------------
# Smoothness detection
# --------------------------------------------------------------------

def _primes_upto(n: int) -> list[int]:
    """Simple sieve for primes up to n.  Hard cap at 5 million."""
    n = min(n, 5_000_000)
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p:n+1:p] = b'\x00' * ((n - p*p) // p + 1)
    return [i for i in range(2, n + 1) if sieve[i]]


def is_smooth(n: int, bound: int) -> bool:
    """Return True if all prime factors of n are ≤ bound.

    Returns True for n < 2.
    """
    if n < 2:
        return True
    d = 2
    while d * d <= n:
        if n % d == 0:
            if d > bound:
                return False
            while n % d == 0:
                n //= d
        d += 1 if d == 2 else 2
    return n == 1 or n <= bound


def smoothness_bound_for_N(N: int) -> int:
    """Suggest a smoothness bound based on N size.

    Capped at 2^22 (~4 million) to keep factor base manageable.
    Uses integer sqrt to avoid float overflow for large N.
    """
    # N ** 0.25 = sqrt(sqrt(N)) using integer arithmetic
    n_bits = N.bit_length()
    # Approximate N^0.25 as 2^(n_bits/4)
    approx = 1 << max(0, n_bits // 4)
    return min(max(2**18, approx), 2**22)


def factorize_small(n: int, bound: int) -> dict[int, int] | None:
    """Return exponent vector dict {prime: exponent} for n's prime factors ≤ bound.

    Skips numbers with any prime factor > bound.  Returns None if n has
    a large prime factor, otherwise returns the exponent map.
    """
    result: dict[int, int] = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            if d > bound:
                return None
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            result[d] = cnt
        d += 1 if d == 2 else 2
    if n > 1 and n <= bound:
        result[n] = 1
    elif n > 1:
        return None
    return result


# --------------------------------------------------------------------
# Relation collection
# --------------------------------------------------------------------

def collect_smooth_relations(N: int,
                            word_length: int = 20,
                            max_words: int = 50000,
                            smooth_bound: int | None = None,
                            seed_words: int = 5
                            ) -> tuple[list[dict[int, int]], list[int]]:
    """Collect smooth relations from bounded-length branch words.

    Args:
        N: Integer to factor
        word_length: Maximum branch word length (sliding window)
        max_words: Maximum number of words to try
        smooth_bound: Largest prime allowed in smooth relation
        seed_words: Number of different starting seeds to use

    Returns:
        (list of exponent-vector relations, list of corresponding norms)
        Only norms that are fully smooth are returned.
    """
    if smooth_bound is None:
        smooth_bound = smoothness_bound_for_N(N)

    # Factor base
    fb = _primes_upto(smooth_bound)
    if len(fb) > 150:
        fb = fb[:150]
    if not fb:
        return [], []

    thin_seed = Triple(3 % N, 4 % N, 5 % N)
    branches = ['U', 'A', 'D']
    relations: list[dict[int, int]] = []
    rel_norms: list[int] = []

    for seed_idx in range(seed_words):
        salt = (seed_idx * 7919) % (N - 1) + 1
        seed = Triple(
            (thin_seed.a * salt) % N,
            (thin_seed.b * salt) % N,
            (thin_seed.c * salt) % N,
        )

        random.seed(seed_idx * 12345)
        word: list[str] = []
        limit = max_words // seed_words
        for _ in range(limit):
            word.append(branches[random.randrange(3)])
            if len(word) > word_length:
                word.pop(0)
            if len(word) < 3:
                continue

            norm = norm_of_branch_word(''.join(word), seed, N)
            if norm <= 1:
                continue

            exp_vec = factorize_small(norm, fb[-1])
            if exp_vec is not None and len(exp_vec) > 0:
                relations.append(exp_vec)
                rel_norms.append(norm)

            if len(relations) >= len(fb) + 25:
                break

        if len(relations) >= len(fb) + 25:
            break

    return relations, rel_norms


# --------------------------------------------------------------------
# Sparse linear algebra over GF(2)
# --------------------------------------------------------------------

class GF2SparseMatrix:
    """Sparse matrix over GF(2), stored as list of sets per row."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data: list[set[int]] = [set() for _ in range(rows)]

    def set(self, r: int, c: int) -> None:
        """Set matrix[r][c] = 1 (xor toggle)."""
        if c in self.data[r]:
            self.data[r].remove(c)
        else:
            self.data[r].add(c)

    def gaussian_elimination(self) -> list[int]:
        """Row-reduce over GF(2).  Returns one nullspace vector as column list."""
        m, n = self.rows, self.cols
        if m == 0 or n == 0:
            return []

        mat = [set(r) for r in self.data]
        r = 0
        for c in range(n):
            pr = -1
            for i in range(r, m):
                if c in mat[i]:
                    pr = i
                    break
            if pr == -1:
                continue
            mat[r], mat[pr] = mat[pr], mat[r]
            for i in range(m):
                if i != r and c in mat[i]:
                    mat[i] ^= mat[r]
            r += 1
            if r >= m:
                break

        # Find a row with no pivot (columns not in any pivot position)
        pivot_cols = set()
        rr = 0
        for c in range(n):
            pr = -1
            for i in range(rr, m):
                if c in mat[i]:
                    pr = i
                    break
            if pr != -1:
                pivot_cols.add(c)
                rr += 1

        for i in range(m):
            non_pivot = mat[i] - pivot_cols
            if non_pivot:
                return list(non_pivot)
        return []

    def __repr__(self):
        return f"GF2SparseMatrix({self.rows}x{self.cols})"


def build_relation_matrix(relations: list[dict[int, int]],
                          fb: list[int]) -> GF2SparseMatrix:
    """Build GF(2) matrix from exponent vectors.

    Args:
        relations: list of {prime: exponent}
        fb: factor base primes (column index = position in fb)

    Returns:
        GF2SparseMatrix with rows=len(relations), cols=len(fb)
    """
    prime_to_col = {p: i for i, p in enumerate(fb)}
    M = GF2SparseMatrix(len(relations), len(fb))
    for ri, row_exp in enumerate(relations):
        for prime, exp in row_exp.items():
            if prime in prime_to_col and exp % 2 == 1:
                M.set(ri, prime_to_col[prime])
    return M


def solve_nullspace(M: GF2SparseMatrix) -> list[int] | None:
    """Compute a nullspace vector over GF(2)."""
    result = M.gaussian_elimination()
    return result if result else None


# --------------------------------------------------------------------
# Congruence extraction
# --------------------------------------------------------------------

def extract_congruence(null_vec: list[int],
                       fb: list[int],
                       norms: list[int],
                       relations: list[dict[int, int]],
                       N: int) -> tuple[int, int] | None:
    """Build x^2 ≡ y^2 (mod N) from nullvector and extract a factor.

    The nullvector (over GF(2)) encodes which rows to xor.
    The xor of those rows sums to zero → product of their norms
    is a perfect square modulo N.
    """
    if not null_vec or not norms or not relations:
        return None

    x_sq = 1
    for ri in null_vec:
        if ri < len(norms):
            x_sq = (x_sq * norms[ri]) % N

    y_sq = 1
    for c in null_vec:
        if c < len(fb):
            y_sq = (y_sq * fb[c]) % N

    diff = x_sq - y_sq
    g = gcd(abs(diff), N)
    if 1 < g < N:
        return (g, N // g)
    return None


# --------------------------------------------------------------------
# Main factoring entry point
# --------------------------------------------------------------------

def orbit_smooth_relation_factor(N: int,
                                 bound: int = 50000,
                                 norm_bound: int | None = None,
                                 fb_size: int = 100,
                                 word_length: int = 20
                                 ) -> tuple[int, int] | None:
    """Factor N via orbit-to-smooth-relation NFS lane.

    This is a research experiment.  Not guaranteed to succeed.

    Algorithm:
    1. Map branch words to integer norms
    2. Sieve for smooth norms
    3. Build GF(2) relation matrix
    4. Compute nullspace → congruence of squares
    5. Extract factor via gcd

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

    if norm_bound is None:
        norm_bound = smoothness_bound_for_N(N)

    relations, rel_norms = collect_smooth_relations(
        N, word_length=word_length, max_words=bound,
        smooth_bound=norm_bound, seed_words=5
    )

    if len(relations) < fb_size + 5:
        return None

    fb = _primes_upto(norm_bound)[:fb_size]
    if not fb:
        return None

    # Pad relations to at least fb_size + 25 rows
    while len(relations) < fb_size + 25:
        relations.append({})
        rel_norms.append(1)

    M = build_relation_matrix(relations[:fb_size + 25], fb)
    null_vec = solve_nullspace(M)
    if null_vec is None:
        return None

    return extract_congruence(null_vec, fb, rel_norms, relations[:fb_size + 25], N)
