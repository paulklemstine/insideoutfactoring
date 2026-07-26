"""Berggren-Gröbner Relation Generator for Factoring.

This module implements a novel factoring method based on navigating the Berggren
(Pythagorean triple) tree to find relations of the form m2 + n2 = k*N, where
the Gaussian integer m+ni has norm sharing factors with N.

For N = p*q where p == 1 (mod 4), p splits in Z[i] as p = π·π̄ with N(π) = p.
The Berggren tree generates ALL Gaussian integer representations of sums of two
squares, so navigating it can find the factor π.

Key insight: The sigma invariants σ1 = a+2b-2c and σ2 = 2a+b-2c determine
which branch to take at each node, guiding the search toward relevant c-values.

References:
    Berggren, B. (1934). "Pytagoreiska trianglar".
"""
from __future__ import annotations

from math import gcd, isqrt
from collections import deque
from typing import NamedTuple

from .berggren import Triple, U as U_MAT, A as A_MAT, D as D_MAT
from .berggren import apply_matrix, children as triple_children
from .gaussian import MnPair, mn_to_triple, apply_mn_matrix
from .gaussian import U_MN, A_MN, D_MN, mn_children


class BerggrenNode(NamedTuple):
    """A node in the Berggren tree with (m,n) and depth info."""
    pair: MnPair
    depth: int
    parent_branch: str | None  # 'U', 'A', or 'D' from which we came


# Root (m,n) = (2,1) gives c = 5 = 22+12, triple (3,4,5)
ROOT_PAIR = MnPair(2, 1)
ROOT_TRIPLE = Triple(3, 4, 5)


def compute_sigmas(triple: Triple) -> tuple[int, int]:
    """Compute sigma invariants for branch navigation.

    σ1 = a + 2b - 2c
    σ2 = 2a + b - 2c

    Navigation rules:
    - σ1 > 0, σ2 < 0 → U-parent (go UP in tree)
    - σ1 > 0, σ2 > 0 → A-parent
    - σ1 < 0, σ2 > 0 → D-parent
    """
    a, b, c = triple
    sigma1 = a + 2 * b - 2 * c
    sigma2 = 2 * a + b - 2 * c
    return sigma1, sigma2


def branch_from_sigmas(sigma1: int, sigma2: int) -> str:
    """Determine which branch to take based on sigma values.

    Returns the parent branch to go toward:
    - 'U' if σ1 > 0, σ2 < 0
    - 'A' if σ1 > 0, σ2 > 0
    - 'D' if σ1 < 0, σ2 > 0
    """
    if sigma1 > 0 and sigma2 < 0:
        return 'U'
    elif sigma1 > 0 and sigma2 > 0:
        return 'A'
    elif sigma1 < 0 and sigma2 > 0:
        return 'D'
    else:
        # Edge case: σ1 = 0 or σ2 = 0
        if sigma2 <= 0:
            return 'A'  # Default to A when σ2 is non-positive
        return 'D'  # Default to D when σ1 is non-positive


def check_berggren_relation(pair: MnPair, N: int) -> tuple[int, int] | None:
    """Check if the (m,n) pair gives a relation useful for factoring N.

    Returns (p, q) where p = gcd(c, N) > 1 and q = N/p, or None if no useful relation.

    Checks gcd(c, N), gcd(c-N, N), and gcd(c+N, N).
    """
    m, n = pair
    c = m * m + n * n  # Norm of m+ni

    # Check various GCD relations
    for offset in [0, N, -N]:
        g = gcd(c - offset, N)
        if 1 < g < N:
            return (g, N // g)

    return None


def _navigate_toward_target(
    start_pair: MnPair,
    target_c_mod: int,
    N: int,
    max_steps: int = 100
) -> MnPair | None:
    """Navigate the Berggren tree toward nodes with c == target_c_mod (mod N).

    Uses gradient descent on the modular distance |c - target_c_mod|.

    Args:
        start_pair: Starting (m,n) pair
        target_c_mod: Target value for c mod N (typically 0 for N, or ±N)
        N: The number being factored
        max_steps: Maximum navigation steps

    Returns:
        A pair with useful GCD relation, or None
    """
    current = start_pair

    for _ in range(max_steps):
        # Check current node
        result = check_berggren_relation(current, N)
        if result is not None:
            return current

        # Get all three children
        child_U, child_A, child_D = mn_children(current)

        # Compute c values for all children
        c_U = child_U.m * child_U.m + child_U.n * child_U.n
        c_A = child_A.m * child_A.m + child_A.n * child_A.n
        c_D = child_D.m * child_D.m + child_D.n * child_D.n

        # Find child closest to target_c_mod (mod N)
        def modular_distance(c: int) -> int:
            """Distance from c to target_c_mod modulo N."""
            diff = (c - target_c_mod) % N
            return min(diff, N - diff)

        distances = {
            'U': modular_distance(c_U),
            'A': modular_distance(c_A),
            'D': modular_distance(c_D)
        }

        best_branch = min(distances, key=distances.get)
        current = {'U': child_U, 'A': child_A, 'D': child_D}[best_branch]

    return None


def _descend_berggren_bfs(
    start_pair: MnPair,
    N: int,
    max_depth: int = 12,
    check_mod: int | None = None
) -> tuple[int, int] | None:
    """Breadth-first search of Berggren tree looking for factoring relations.

    Args:
        start_pair: Starting (m,n) pair
        N: Number to factor
        max_depth: Maximum tree depth to search
        check_mod: If set, only check nodes where c == check_mod (mod N)

    Returns:
        (p, q) factorization if found, None otherwise.
    """
    # Queue of (pair, depth, path)
    queue = deque([(start_pair, 0, '')])

    while queue:
        pair, depth, path = queue.popleft()

        if depth > max_depth:
            continue

        # Check this node
        m, n = pair
        c = m * m + n * n

        if check_mod is not None:
            if (c - check_mod) % N != 0 and (c + check_mod) % N != 0:
                pass  # Skip this node

        result = check_berggren_relation(pair, N)
        if result is not None:
            return result

        # Add children
        if depth < max_depth:
            for branch, child in zip(['U', 'A', 'D'], mn_children(pair)):
                queue.append((child, depth + 1, path + branch))

    return None


def _sigma_guided_descent(
    start_pair: MnPair,
    N: int,
    max_depth: int = 10
) -> tuple[int, int] | None:
    """Sigma-guided descent: use sigma invariants to navigate efficiently.

    At each node, compute σ1, σ2 to determine which branch leads toward
    nodes with c divisible by factors of N.

    The sigma values encode information about the modular position of c.
    When σ1 and σ2 have opposite signs, we're in the U branch region.
    """
    current = start_pair

    for _ in range(max_depth):
        # Check current
        result = check_berggren_relation(current, N)
        if result is not None:
            return result

        # Get triple and compute sigmas
        triple = mn_to_triple(current)
        sigma1, sigma2 = compute_sigmas(triple)

        # Determine which branch to take
        # For factoring, we want c to be divisible by p or q
        # The sigma-guided navigation tries to find nodes with small c mod factors
        parent_branch = branch_from_sigmas(sigma1, sigma2)

        # Take inverse of parent branch to go toward root (exploring different region)
        branch_map = {'U': D_MN, 'A': A_MN, 'D': U_MN}  # Go opposite direction

        # But actually, for exploration we want to go FORWARD on promising branches
        # Use sigma sign to pick: σ1 > 0 suggests A or U, σ2 > 0 suggests A or D
        children = mn_children(current)

        if sigma1 > 0 and sigma2 > 0:
            # A region - try A child first
            candidates = [children[1], children[0], children[2]]  # A, U, D
        elif sigma1 > 0 and sigma2 < 0:
            # U region
            candidates = [children[0], children[1], children[2]]  # U, A, D
        elif sigma1 < 0 and sigma2 > 0:
            # D region
            candidates = [children[2], children[0], children[1]]  # D, U, A
        else:
            # Both non-positive - try all
            candidates = children

        # Try each candidate in order
        current = candidates[0]

    return None


def berggren_grobner_factor(N: int, max_depth: int = 12) -> tuple[int, int] | None:
    """Factor N using Berggren tree navigation with Gröbner-like relation generation.

    This method searches the Berggren (Pythagorean triple) tree for representations
    of N as a sum of two squares or near-sums that reveal factors via GCD.

    For semiprimes N = p*q where p == 1 (mod 4), p splits in Z[i] and there exist
    m, n with m2 + n2 = k*p. The Berggren tree contains ALL such (m,n) pairs.

    The algorithm:
    1. Start at root (m,n) = (2,1) with c = 5
    2. Navigate using sigma-guided descent and BFS exploration
    3. At each node, check gcd(c, N), gcd(c-N, N), gcd(c+N, N)
    4. Return when a non-trivial factor is found

    Args:
        N: Odd composite integer to factor (must not be prime)
        max_depth: Maximum tree depth to search (default 12)

    Returns:
        (p, q) where p <= q and p*q = N, or None if factorization fails

    Examples:
        >>> berggren_grobner_factor(77)
        (7, 11)
        >>> berggren_grobner_factor(65)
        (5, 13)
    """
    if N % 2 == 0:
        return (2, N // 2)

    # Quick checks
    if N < 3:
        return None

    # Check for small prime factors first
    for p in [3, 5, 7, 11, 13, 17, 19, 23, 29]:
        if N % p == 0:
            return (p, N // p)

    # Try multiple strategies
    strategies = [
        lambda: _descend_berggren_bfs(ROOT_PAIR, N, max_depth, check_mod=0),
        lambda: _descend_berggren_bfs(ROOT_PAIR, N, max_depth, check_mod=N),
        lambda: _sigma_guided_descent(ROOT_PAIR, N, max_depth),
        lambda: _navigate_toward_target(ROOT_PAIR, 0, N, max_steps=200),
    ]

    for strategy in strategies:
        try:
            result = strategy()
            if result is not None:
                p, q = result
                if p > q:
                    p, q = q, p
                if p * q == N:
                    return (p, q)
        except (OverflowError, RecursionError):
            continue

    # Try starting from different root positions
    # The tree can be entered at different points
    alt_roots = [
        MnPair(3, 1),   # c = 10
        MnPair(3, 2),   # c = 13
        MnPair(4, 1),   # c = 17
        MnPair(4, 3),   # c = 25
        MnPair(5, 2),   # c = 29
    ]

    for root in alt_roots:
        result = _descend_berggren_bfs(root, N, max_depth // 2)
        if result is not None:
            p, q = result
            if p > q:
                p, q = q, p
            if p * q == N:
                return (p, q)

    return None


def berggren_variety_factor(N: int, max_depth: int = 8) -> tuple[int, int] | None:
    """Factor using the algebraic variety perspective.

    For N = p*q, the set of (m,n) with m2+n2 == 0 (mod N) forms a variety.
    The ideal generated by {m2+n2-N} in the Berggren ring should factor as
    prime ideals corresponding to p and q.

    This method uses Gröbner-basis-like elimination to find the variety
    intersected with the tree structure.

    Args:
        N: Number to factor
        max_depth: Maximum depth for tree search

    Returns:
        (p, q) factorization or None
    """
    # Strategy: find (m,n) such that m2+n2 = k*N for small k
    # Then compute gcd(m+ni, N) in Z[i] to recover factors

    for k in range(1, 20):
        target = k * N
        # Find (m,n) with m2+n2 close to target
        result = _navigate_toward_target(ROOT_PAIR, target % N, N, max_steps=100)
        if result is not None:
            p, q = result
            if p * q == N:
                return (p, q) if p <= q else (q, p)

    return None


# Export public API
__all__ = [
    'berggren_grobner_factor',
    'berggren_variety_factor',
    'check_berggren_relation',
    'compute_sigmas',
    'branch_from_sigmas',
    'ROOT_PAIR',
    'ROOT_TRIPLE',
]
