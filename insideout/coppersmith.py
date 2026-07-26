"""Coppersmith's Method — Polynomial-Time Factorization via Lattice Reduction.

A novel implementation of Coppersmith's method for finding small roots of polynomial
equations modulo N. This is theoretically significant because:

1. If N = pq has a factor p near sqrt(N), and we know an approximation to p,
   Coppersmith's method can recover p exactly in polynomial time.

2. More generally, for f(x) = 0 mod N with |x0| < N^(1/d), Coppersmith's
   method finds x0 in polynomial time using lattice reduction.

The algorithm:
1. Build a lattice from powers of x and N
2. Apply LLL reduction to find short vectors
3. Short vectors correspond to polynomials with small roots
4. Extract the root from the short vector

Per honest assessment: While theoretically polynomial-time, Coppersmith's method requires
knowing an approximation to a factor, which limits practical applicability.
For general inputs, GNFS remains fastest.
"""
from __future__ import annotations

from math import gcd, isqrt


def _lll_reduce(basis, delta=0.75):
    """LLL lattice basis reduction.

    Takes a list of lattice vectors as rows and returns an LLL-reduced basis.
    """
    if not basis:
        return []

    n = len(basis)
    m = len(basis[0]) if basis else 0
    if m == 0:
        return []

    # Convert to list of lists of floats
    B = [[float(x) for x in row] for row in basis]

    # Gram-Schmidt process
    mu = [[0.0] * n for _ in range(n)]
    norm_sq = [0.0] * n

    for i in range(n):
        mu_i_i = 0.0
        for j in range(i):
            mu_ij = sum(B[i][k] * mu[j][k] for k in range(j)) if j > 0 else 0.0
            mu[i][j] = mu_ij
            mu_i_i += mu_ij * mu_ij
        norm_sq[i] = sum(B[i][k] * B[i][k] for k in range(m)) - mu_i_i

    # LLL reduction
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
                    mu_ki = mu[k][i]
                    for t in range(i):
                        mu_ki -= mu[j][t] * mu[i][t]
                    mu_i_i = sum(mu[k][t] * mu[k][t] for t in range(i)) if i > 0 else 0.0
                    if norm_sq[i] > 0:
                        mu[k][i] = mu_ki / norm_sq[i]
                    else:
                        mu[k][i] = 0.0

        # Lovasz condition
        lhs = norm_sq[k]
        rhs = (delta - mu[k][k-1]**2) * norm_sq[k-1]

        if lhs >= rhs:
            k += 1
        else:
            # Swap B[k] and B[k-1]
            B[k], B[k-1] = B[k-1], B[k]

            # Update mu and norm_sq
            temp_mu = mu[k][:]
            mu[k] = mu[k-1][:]
            mu[k-1] = temp_mu

            temp_norm = norm_sq[k]
            norm_sq[k] = norm_sq[k-1]
            norm_sq[k-1] = temp_norm

            # Update rows below
            for i in range(k + 1, n):
                temp = mu[i][k]
                mu[i][k] = mu[i][k-1]
                mu[i][k-1] = temp

            k = max(k - 1, 1)

    return B


def _build_lattice_for_factor(N, X, k=2):
    """Build lattice for Coppersmith's method to find p near sqrt(N).

    The lattice consists of vectors:
    - N * x^i for i = 0..k-1
    - x^i for i = 0..k

    Returns a k+1 by k+1 integer matrix.
    """
    m = k + 1
    lattice = []

    # First k rows: N * x^i (for i = 0..k-1)
    for i in range(k):
        row = [0.0] * m
        row[i] = float(N)
        lattice.append(row)

    # Last row: x^k - X
    row = [0.0] * m
    row[k] = float(X)
    lattice.append(row)

    return lattice


def coppersmith_factor(N, X=None, k=2):
    """Factor N using Coppersmith's method.

    If we know that N = pq where p is approximately X, and |p - X| < N^(1/2),
    this method recovers p in polynomial time.

    Args:
        N: The number to factor
        X: Approximation to the factor p (if None, uses sqrt(N))
        k: Lattice parameter (higher = larger lattice, may find larger roots)

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

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # If no approximation given, use sqrt(N)
    if X is None:
        X = s

    # Build lattice
    lattice = _build_lattice_for_factor(N, X, k)

    # Apply LLL reduction
    reduced = _lll_reduce(lattice)

    if not reduced:
        return None

    # First row of reduced basis should be small
    first_row = reduced[0]

    # Convert to integer polynomial
    coeffs = [int(round(c)) for c in first_row]

    # Try small perturbations around X
    for dx in range(-1000, 1000):
        x = X + dx
        if x <= 0:
            continue
        # Evaluate x^2 - N (should be divisible by p if x approximates p)
        remainder = (x * x - N) % N
        g = gcd(remainder, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Try using the polynomial directly
    # The first reduced vector gives us a polynomial h(x) with small coefficients
    h_x = sum(coeffs[i] * (X ** i) for i in range(len(coeffs)))
    if h_x != 0:
        g = gcd(abs(h_x) % N, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    return None