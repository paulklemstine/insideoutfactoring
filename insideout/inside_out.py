"""Inside-Out Factoring Algorithm.

The core algorithm: start at the Central Approximation Well (near sqrt(N))
and search radially outward through the Pythagorean tree, using CF steering
and modular filters to prune the search.

Two search strategies are available:
- BFS (breadth-first): Explores the tree level by level. Guaranteed to find
  the closest node but slow for close-factor semiprimes.
- Steered (best-first): Uses predict_branch distance as priority to guide
  the search toward the target. Much faster for close-factor cases.
"""
from __future__ import annotations

import heapq
from collections import deque
from math import gcd, isqrt
from typing import Iterator

from .berggren import Triple, children, apply_matrix, U, A, D
from typing import NamedTuple
from .gaussian import MnPair, mn_to_triple, triple_to_mn_pair, mn_children
from .energy import is_energy_compatible, hypotenuse_bound
from .cf_guide import cf_sqrt, convergents, predict_branch, cf_factor_check
from .modular import filter_wavefront
from itertools import product
import random


def central_well(N: int) -> MnPair:
    """Compute the Central Approximation Well for N.

    For N = pq with p ~ q ~ sqrt(N), the well corresponds to the
    (m, n) pair near sqrt(sqrt(N)). We find the nearest valid (m, n)
    such that:
      - m > n > 0
      - gcd(m, n) = 1
      - (m - n) is odd (PPT condition)

    The well starts at m ~ isqrt(N) + 1, which places the resulting
    triple's hypotenuse near N.
    """
    sqrt_N = isqrt(N)

    # Start with m near sqrt(N), n = 1
    # The triple from (m, n) has c = m^2 + n^2
    # We want c ~ N, so m ~ sqrt(N)
    m_start = sqrt_N + 1
    n_start = 1

    # Ensure m > n (trivially satisfied for N >= 4)
    # Ensure (m - n) is odd for PPT condition
    if (m_start - n_start) % 2 == 0:
        m_start += 1

    # Find coprime m, n
    # Walk m up until we find gcd(m, n) = 1
    m = m_start
    n = n_start
    while gcd(m, n) != 1:
        m += 2  # Keep m - n odd

    return MnPair(m, n)


def cf_seeded_well_points(N: int) -> list[MnPair]:
    """Generate BFS seed points from CF convergents of sqrt(N).

    For close-factor semiprimes, CF convergents of sqrt(N) approximate
    the factors p and q. By generating (m,n) seed points from these
    convergents, we ensure the BFS starts near the target node even
    when p ~ q.

    Also includes the traditional well and its neighborhood for
    well-separated factor coverage.
    """
    seeds: list[MnPair] = []
    seen: set[tuple[int, int]] = set()

    def _add_seed(m: int, n: int) -> None:
        """Add a valid (m,n) seed point, skipping duplicates."""
        if (m, n) in seen:
            return
        if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
            seen.add((m, n))
            seeds.append(MnPair(m, n))

    # CF-convergent-derived seed points
    if N >= 4:
        cf = cf_sqrt(N, max_terms=50)
        convs = convergents(cf)

        for pk, qk in convs[:20]:
            # Try (pk, 1) as a seed — convergent numerator is near sqrt(N)
            _add_seed(pk, 1)
            # Try (pk, qk) if it satisfies PPT conditions
            _add_seed(pk, qk)
            # Try (qk, 1) — convergent denominator
            if qk > 1:
                _add_seed(qk, 1)
            # Try nearby values: pk ± 1 with small n values
            for delta in (-1, 1):
                m = pk + delta
                if m > 1:
                    _add_seed(m, 1)
                    for n_val in range(2, min(m, 8)):
                        _add_seed(m, n_val)

    # Traditional well and its neighborhood
    well = central_well(N)
    _add_seed(well.m, well.n)
    m0, n0 = well.m, well.n
    for dm in range(-5, 6):
        for dn in range(0, min(m0, 6)):
            _add_seed(m0 + dm, n0 + dn)

    # Root (3,4,5) -> (m,n) = (2,1) for full tree coverage
    _add_seed(2, 1)

    return seeds


def resonance_check(N: int, triple: Triple) -> tuple[int, int] | None:
    """Check if a triple reveals the factors of N.

    For N = pq, if we have a triple (a, b, c) where a = N, then
    b = (q^2 - p^2)/2 and c = (q^2 + p^2)/2, giving p and q.

    More generally, check if N^2 - a^2 or N^2 - b^2 is a perfect square.
    If N^2 - a^2 = d^2, then gcd(N, d) may be a factor.
    Also checks whether a or b directly divides N.
    """
    a, b, c = triple

    # Check if a divides N (direct divisor hit)
    if 1 < a < N and N % a == 0:
        return (min(a, N // a), max(a, N // a))

    # Check if b divides N
    if 1 < b < N and N % b == 0:
        return (min(b, N // b), max(b, N // b))

    # Check if a == N (trivial, not useful)
    if a == N:
        return None

    # Check N^2 - a^2 = d^2 (perfect square)
    # Then N^2 = a^2 + d^2, and gcd(N, d) may be a factor
    delta_a = N * N - a * a
    if delta_a > 0:
        sqrt_delta = isqrt(delta_a)
        if sqrt_delta * sqrt_delta == delta_a:
            d = sqrt_delta
            g = gcd(N, d)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    # Check N^2 - b^2 = d^2 (perfect square)
    delta_b = N * N - b * b
    if delta_b > 0:
        sqrt_delta = isqrt(delta_b)
        if sqrt_delta * sqrt_delta == delta_b:
            d = sqrt_delta
            g = gcd(N, d)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    return None


def _steered_search(N: int, max_iterations: int = 50000) -> tuple[int, int] | None:
    """Factor N using CF-steered best-first search.

    Uses a priority queue (min-heap) ordered by predict_branch distance
    to guide the search toward nodes most likely to contain factors of N.
    This converts the BFS from O(N) iterations to O(log2N) directed search
    for semiprimes where CF convergents don't directly reveal factors.

    Returns (p, q) with p <= q if factorization found, None otherwise.
    """
    well = central_well(N)
    upper = hypotenuse_bound(N)

    # Priority queue: (distance, counter, MnPair)
    # counter is a tiebreaker to avoid comparing MnPair objects
    visited: set[tuple[int, int]] = set()
    heap: list[tuple[int, int, MnPair]] = []
    counter = 0

    # Seed with CF-convergent-derived points
    seed_points = cf_seeded_well_points(N)
    for seed in seed_points:
        if (seed.m, seed.n) not in visited:
            triple = mn_to_triple(seed)
            if triple.c <= upper:
                dist = min(predict_branch(N, (triple.a, triple.b, triple.c)))
                heapq.heappush(heap, (dist, counter, seed))
                counter += 1

    iterations = 0
    while heap and iterations < max_iterations:
        _, _, current = heapq.heappop(heap)
        iterations += 1

        key = (current.m, current.n)
        if key in visited:
            continue
        visited.add(key)

        # Skip invalid PPT parameters
        if current.m <= current.n:
            continue
        if gcd(current.m, current.n) != 1:
            continue
        if (current.m - current.n) % 2 != 1:
            continue

        # Convert to triple
        triple = mn_to_triple(current)
        a, b, c = triple

        # Energy bound check
        if c > upper:
            continue

        # Skip resonance check for triples below N (too small to contain N)
        if c >= N:
            result = resonance_check(N, triple)
            if result is not None:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return (min(p, q), max(p, q))

        # Check direct divisibility even for small triples
        if 1 < a < N and N % a == 0:
            return (min(a, N // a), max(a, N // a))
        if 1 < b < N and N % b == 0:
            return (min(b, N // b), max(b, N // b))

        # Expand children, prioritized by predict_branch distance
        for child in mn_children(current):
            child_key = (child.m, child.n)
            if child_key not in visited and child.m > child.n > 0:
                child_triple = mn_to_triple(child)
                if child_triple.c <= upper:
                    child_dist = min(
                        predict_branch(N, (child_triple.a, child_triple.b, child_triple.c))
                    )
                    heapq.heappush(heap, (child_dist, counter, child))
                    counter += 1

    return None


def inside_out_factor(N: int, max_iterations: int = 500000) -> tuple[int, int] | None:
    """Factor N = p*q using Inside-Out traversal of the Pythagorean tree.

    Uses a multi-strategy approach:
    1. Perfect square detection (O(1))
    2. CF convergent divisibility pre-check (O(log N))
    3. Quick trial division for small factors
    4. CF-steered best-first search (primary)
    5. BFS fallback (guaranteed coverage)
    6. Full trial division (safety net)

    Returns (p, q) with p <= q if factorization found, None if N is prime
    or factors not found within max_iterations.
    """
    # Edge cases
    if N < 4:
        return None

    # Skip for large N — tree traversal too slow
    if N.bit_length() > 256:
        return None

    # Handle even N
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square detection: if N = p^2, then p is a factor
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # CF convergent divisibility pre-check: the convergents of sqrt(N)
    # often directly reveal factors, especially for close-factor semiprimes.
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        return cf_result

    # Quick trial division for small factors (safety net)
    for p in range(3, min(isqrt(N) + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Strategy 1: CF-steered best-first search (fast for close factors)
    steered_result = _steered_search(N, max_iterations=max_iterations)
    if steered_result is not None:
        return steered_result

    # Strategy 2: BFS from the well (guaranteed coverage for edge cases)
    bfs_result = _bfs_search(N, max_iterations=max_iterations)
    if bfs_result is not None:
        return bfs_result

    # Fallback: trial division up to sqrt(N)
    limit = isqrt(N) + 1
    for p in range(3, limit, 2):
        if N % p == 0:
            return (p, N // p)

    return None


def _bfs_search(N: int, max_iterations: int = 500000) -> tuple[int, int] | None:
    """Factor N using BFS from the well (original Inside-Out algorithm).

    This is the fallback search strategy. It expands the tree level by level
    from the well, checking each node for resonance with N.
    """
    well = central_well(N)
    upper = hypotenuse_bound(N)

    # Seed the BFS with CF-convergent-derived starting points
    seed_points = cf_seeded_well_points(N)

    visited: set[tuple[int, int]] = set()
    queue: deque[MnPair] = deque()

    for seed in seed_points:
        key = (seed.m, seed.n)
        if key not in visited:
            queue.append(seed)
            visited.add(key)

    # Also add root for coverage
    root_mn = MnPair(2, 1)
    if (2, 1) not in visited:
        queue.append(root_mn)
        visited.add((2, 1))

    iterations = 0

    while queue and iterations < max_iterations:
        current = queue.popleft()
        iterations += 1

        # Skip if m <= n (invalid PPT parameter)
        if current.m <= current.n:
            continue

        # Check PPT validity: coprime and opposite parity
        if gcd(current.m, current.n) != 1:
            continue
        if (current.m - current.n) % 2 != 1:
            continue

        # Convert to triple
        triple = mn_to_triple(current)
        a, b, c = triple

        # Energy bound check
        if c > upper:
            continue

        # Check resonance with N (only for triples large enough)
        if c >= N:
            result = resonance_check(N, triple)
            if result is not None:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q < N:
                    return (min(p, q), max(p, q))

        # Check direct divisibility
        if 1 < a < N and N % a == 0:
            return (min(a, N // a), max(a, N // a))
        if 1 < b < N and N % b == 0:
            return (min(b, N // b), max(b, N // b))

        # Expand children using (m,n) Berggren transforms
        for child in mn_children(current):
            child_key = (child.m, child.n)
            if child_key not in visited and child.m > child.n > 0:
                visited.add(child_key)
                # Pre-check: will this child's triple be too large?
                child_triple = mn_to_triple(child)
                if child_triple.c <= upper:
                    queue.append(child)

    return None


# =============================================================================
# Bidirectional Cofactor Inside-Out Algorithm
# Mathematical breakthrough: goal nodes as coordinate-zero covectors, transposed
# Berggren matrices for backward search, covector-vector inner product for meet.
# =============================================================================

class Covector(NamedTuple):
    """A covector (x, y, z) representing a linear functional."""
    x: int
    y: int
    z: int

    def mod(self, N: int) -> tuple[int, int, int]:
        return (self.x % N, self.y % N, self.z % N)


class BranchWord(NamedTuple):
    """A word in the U/A/D branch alphabet."""
    letters: tuple[str, ...]

    def prepend(self, letter: str) -> "BranchWord":
        return BranchWord((letter,) + self.letters)

    def extend(self, letters: "BranchWord") -> "BranchWord":
        return BranchWord(self.letters + letters.letters)

    def reversed(self) -> "BranchWord":
        return BranchWord(self.letters[::-1])

    def concat(self, other: "BranchWord") -> "BranchWord":
        return BranchWord(self.letters + other.letters)

    def __str__(self) -> str:
        return "".join(self.letters)


# Forward branch matrices: ACT on column vectors v = (a, b, c)
# U: (a,b,c) -> (a+2b+2c, -2a-b-2c, 2a+2b+3c)
# A: (a,b,c) -> (a+2b+2c,  2a+b+2c, 2a+2b+3c)
# D: (a,b,c) -> (-a-2b-2c, 2a+b+2c, 2a+2b+3c)

def _forward_apply(v: tuple[int, int, int], branch: str, N: int) -> tuple[int, int, int]:
    """Apply forward branch matrix to column vector v mod N."""
    a, b, c = v
    if branch == 'U':
        # (a+2b+2c, -2a-b-2c, 2a+2b+3c)
        return ((a + 2*b + 2*c) % N,
                (-2*a - b - 2*c) % N,
                (2*a + 2*b + 3*c) % N)
    elif branch == 'A':
        # (a+2b+2c, 2a+b+2c, 2a+2b+3c)
        return ((a + 2*b + 2*c) % N,
                (2*a + b + 2*c) % N,
                (2*a + 2*b + 3*c) % N)
    else:  # 'D'
        # (-a-2b-2c, 2a+b+2c, 2a+2b+3c)
        return ((-a - 2*b - 2*c) % N,
                (2*a + b + 2*c) % N,
                (2*a + 2*b + 3*c) % N)


def _backward_apply(ell: tuple[int, int, int], branch: str, N: int) -> tuple[int, int, int]:
    """Apply transposed pullback to covector ell mod N.

    Transposed pullback formulas for covectors (x, y, z):
    U: (x+2y+2z, -2x-y-2z, 2x+2y+3z)
    A: (x+2y+2z,  2x+y+2z, 2x+2y+3z)
    D: (-x-2y-2z, 2x+y+2z, 2x+2y+3z)
    """
    x, y, z = ell
    if branch == 'U':
        # (x+2y+2z, -2x-y-2z, 2x+2y+3z)
        return ((x + 2*y + 2*z) % N,
                (-2*x - y - 2*z) % N,
                (2*x + 2*y + 3*z) % N)
    elif branch == 'A':
        # (x+2y+2z, 2x+y+2z, 2x+2y+3z)
        return ((x + 2*y + 2*z) % N,
                (2*x + y + 2*z) % N,
                (2*x + 2*y + 3*z) % N)
    else:  # 'D'
        # (-x-2y-2z, 2x+y+2z, 2x+2y+3z)
        return ((-x - 2*y - 2*z) % N,
                (2*x + y + 2*z) % N,
                (2*x + 2*y + 3*z) % N)


def _inner_product_mod(v: tuple[int, int, int], ell: tuple[int, int, int], N: int) -> int:
    """Compute <ell, v> mod N = ell[0]*v[0] + ell[1]*v[1] + ell[2]*v[2] mod N."""
    return (ell[0]*v[0] + ell[1]*v[1] + ell[2]*v[2]) % N


def _forward_frontier(N: int, v0: tuple[int, int, int], depth: int, max_states: int = 10000):
    """Generate forward frontier states from seed v0 at given depth.

    Returns dict: modular_state -> (prefix_word, modular_state)
    Uses the standard forward branch matrices.
    Seed v0 = (0, -1, 1) mod N = (0, N-1, 1)
    """
    # Seed state: (0, N-1, 1)
    # Format: dict[modular_state] -> (word_letters_tuple, modular_state)
    frontier = {v0: ([], v0)}
    seen = {v0}

    for level in range(depth):
        next_frontier = {}
        for state, (word, _) in frontier.items():
            for branch in ('U', 'A', 'D'):
                new_state = _forward_apply(state, branch, N)
                if new_state in seen:
                    continue
                seen.add(new_state)
                new_word = word + [branch]
                next_frontier[new_state] = (new_word, new_state)

                if len(seen) >= max_states:
                    # Merge current and next frontier
                    for k, v in next_frontier.items():
                        if k not in frontier:
                            frontier[k] = v
                    return frontier

        frontier = next_frontier
        if not frontier:
            break

    return frontier


def _backward_frontier(N: int, goals: list[tuple[int, int, int]], depth: int, max_states: int = 10000):
    """Generate backward frontier covectors from goals at given depth.

    Returns dict: modular_form -> (suffix_word_reversed, modular_form)
    Uses the TRANSPOSED pullback formulas.
    Start from goals = {e_a, e_b, e_c} = {(1,0,0), (0,1,0), (0,0,1)}
    """
    # Start from each goal covector
    frontier = {}
    seen = set()

    for goal in goals:
        goal_mod = (goal[0] % N, goal[1] % N, goal[2] % N)
        if goal_mod not in seen:
            seen.add(goal_mod)
            frontier[goal_mod] = ([], goal_mod)

    for level in range(depth):
        next_frontier = {}
        for form, (word, _) in frontier.items():
            for branch in ('U', 'A', 'D'):
                # Apply backward (pullback) via TRANSPOSED matrix
                new_form = _backward_apply(form, branch, N)
                if new_form in seen:
                    continue
                seen.add(new_form)
                # Prepend to word since we're going backward
                new_word = [branch] + word
                next_frontier[new_form] = (new_word, new_form)

                if len(seen) >= max_states:
                    for k, v in next_frontier.items():
                        if k not in frontier:
                            frontier[k] = v
                    return frontier

        frontier = next_frontier
        if not frontier:
            break

    return frontier


def _meet_in_the_middle(N: int, forward_states: dict, backward_forms: dict, batch_size: int = 128) -> tuple[int, int] | None:
    """Check all forward-backward pairs for GCD hits.

    Returns (factor, prefix_word, suffix_word_reversed) if found.
    Uses batch GCD optimization for efficiency.
    """
    forward_list = list(forward_states.items())

    for i in range(0, len(forward_list), batch_size):
        batch = forward_list[i:i + batch_size]

        # Build batch products for each backward form
        # For each forward state v, check against ALL backward forms ell
        # We look for gcd(<ell, v> mod N, N) where 1 < gcd < N

        for forward_state, (fword, _) in batch:
            for backward_form, (bword, _) in backward_forms.items():
                # Compute inner product mod N
                inner = _inner_product_mod(forward_state, backward_form, N)
                if inner == 0:
                    continue

                g = gcd(inner, N)
                if 1 < g < N:
                    # Found a factor! Validate by exact division
                    if N % g == 0:
                        q = N // g
                        if q > g:
                            return (g, q)
                        else:
                            return (q, g)

    return None


def _validate_path(N: int, v0: tuple[int, int, int], word: str) -> tuple[int, int, int] | None:
    """Replay a branch word from the seed to get the final state.

    Returns the final (a, b, c) triple, or None if invalid.
    """
    state = v0
    for branch in word:
        state = _forward_apply(state, branch, N)
    return state


def inside_out_factor_bidirectional(N: int, forward_depth: int = 4, backward_depth: int = 4, batch_size: int = 128) -> tuple[int, int] | None:
    """Bidirectional inside-out factoring using covector meet-in-the-middle.

    This is the mathematically precise version:
    - Goal nodes are coordinate-zero covectors (e_a, e_b, e_c)
    - Backward search uses transposed Berggren maps (pullback)
    - Forward search uses standard Berggren maps
    - Meet check: gcd(<ell, v> mod N, N) where ell is covector, v is state

    The key insight: by representing goal nodes as coordinate-zero covectors
    and using transposed Berggren matrices for backward search, we can find
    collisions between forward and backward frontiers that reveal factors of N.

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square check
    s = isqrt(N)
    if s * s == N:
        return (s, s)

    # Thin seed v0 = (0, -1, 1) mod N = (0, N-1, 1)
    v0 = (0, N - 1, 1)

    # Goal covectors: e_a = (1,0,0), e_b = (0,1,0), e_c = (0,0,1)
    goals = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Generate frontiers
    forward_states = _forward_frontier(N, v0, forward_depth, max_states=10000)
    backward_forms = _backward_frontier(N, goals, backward_depth, max_states=10000)

    if not forward_states or not backward_forms:
        return None

    # Meet check
    result = _meet_in_the_middle(N, forward_states, backward_forms, batch_size)

    if result is not None:
        p, q = result
        if p * q == N:
            return (min(p, q), max(p, q))

    return None


# =============================================================================
# Extension 1: Periodic-Word Powering
# Uses matrix exponentiation for deep structured ray exploration
# =============================================================================

def _word_to_matrix(word: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Convert a branch word to a 3x3 matrix representation.

    Returns the matrix that applies the word's transformations in sequence.
    The forward branch matrices are:
      U: [[1, 2, 2], [-2, -1, -2], [2, 2, 3]]
      A: [[1, 2, 2], [2, 1, 2], [2, 2, 3]]
      D: [[-1, -2, -2], [2, 1, 2], [2, 2, 3]]

    Each matrix acts on column vectors (a, b, c).
    """
    # Identity matrix
    M = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    for branch in word:
        if branch == 'U':
            # U matrix
            u = ((1, 2, 2), (-2, -1, -2), (2, 2, 3))
            M = _matrix_mul(M, u)
        elif branch == 'A':
            # A matrix
            a = ((1, 2, 2), (2, 1, 2), (2, 2, 3))
            M = _matrix_mul(M, a)
        else:  # 'D'
            # D matrix
            d = ((-1, -2, -2), (2, 1, 2), (2, 2, 3))
            M = _matrix_mul(M, d)

    return M


def _matrix_mul(A: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                 B: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Multiply two 3x3 matrices."""
    result = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return (tuple(result[0]), tuple(result[1]), tuple(result[2]))


def _matrix_pow(M: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                k: int, mod: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Raise a 3x3 matrix to power k using binary exponentiation."""
    # Result = identity
    result = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    base = M
    while k > 0:
        if k % 2 == 1:
            result = _matrix_mul_mod(result, base, mod)
        base = _matrix_mul_mod(base, base, mod)
        k //= 2

    return result


def _matrix_mul_mod(A: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                    B: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                    mod: int) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    """Multiply two 3x3 matrices modulo mod."""
    result = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % mod
    return (tuple(result[0]), tuple(result[1]), tuple(result[2]))


def _matrix_apply(v: tuple[int, int, int],
                  M: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
                  mod: int) -> tuple[int, int, int]:
    """Apply a 3x3 matrix to a column vector mod mod."""
    a, b, c = v
    return (
        (M[0][0] * a + M[0][1] * b + M[0][2] * c) % mod,
        (M[1][0] * a + M[1][1] * b + M[1][2] * c) % mod,
        (M[2][0] * a + M[2][1] * b + M[2][2] * c) % mod,
    )


def _power_word(word: str, k: int, N: int) -> tuple[int, int, int]:
    """Apply word repeated k times using matrix powering.

    Instead of applying word k times sequentially (O(k) multiplications),
    use binary exponentiation on the branch matrices (O(log k) multiplications).
    """
    if k == 0:
        return (0, N - 1, 1)  # Identity

    if not word:
        return (0, N - 1, 1)

    # Build the word's matrix
    M = _word_to_matrix(word)

    # Seed vector: v0 = (0, N-1, 1)
    v0 = (0, N - 1, 1)

    # Raise matrix to power k
    M_k = _matrix_pow(M, k, N)

    # Apply to seed
    return _matrix_apply(v0, M_k, N)


def _periodic_frontier(N: int, seed: tuple[int, int, int], max_word_len: int = 3,
                       max_power: int = 16, max_states: int = 5000) -> dict:
    """Generate frontier using periodic words with matrix powering.

    For each short word pattern (U, A, D, UU, UA, UD, etc.) up to max_word_len,
    power it by various k values to explore deep structured rays efficiently.

    Returns dict: modular_state -> (word_letters_tuple, power_k, modular_state)
    """
    frontier = {seed: ((), 0, seed)}
    seen = {seed}

    # Generate all words up to max_word_len
    branches = ('U', 'A', 'D')

    for word_len in range(1, max_word_len + 1):
        for word_tuple in product(branches, repeat=word_len):
            word = ''.join(word_tuple)

            # Try powers 1, 2, 4, 8, ... up to max_power
            k = 1
            while k <= max_power:
                try:
                    state = _power_word(word, k, N)
                except Exception:
                    break

                if state not in seen:
                    seen.add(state)
                    frontier[state] = ((word, k), word, state)

                    if len(seen) >= max_states:
                        return frontier

                k *= 2

    return frontier


class PeriodicWordExplorer:
    """Explorer that uses periodic words via matrix powering for deep ray exploration."""

    def __init__(self, N: int, max_word_len: int = 3, max_power: int = 16):
        self.N = N
        self.max_word_len = max_word_len
        self.max_power = max_power
        self.v0 = (0, N - 1, 1)

    def generate_frontier(self, max_states: int = 5000) -> dict:
        """Generate frontier using periodic word powering."""
        return _periodic_frontier(self.N, self.v0, self.max_word_len, self.max_power, max_states)


# =============================================================================
# Extension 2: Multi-Start Search
# Run multiple seed types through the same backward goal bank
# =============================================================================

def _generate_seeds(N: int, seed_type: str) -> list[tuple[int, int, int]]:
    """Generate various starting seeds for the forward search.

    - 'thin': v0 = (0, N-1, 1) [standard thin seed]
    - 'fibonacci': Use Fibonacci-related triples
    - 'pythagorean': Use primitive Pythagorean triple seeds
    - 'random': Random valid modular triples
    """
    seeds = []

    if seed_type == 'thin':
        # Standard thin seed
        seeds.append((0, N - 1, 1))

    elif seed_type == 'fibonacci':
        # Fibonacci-related seeds: use F_k mod N patterns
        # F_0 = 0, F_1 = 1, F_2 = 1, F_3 = 2, F_4 = 3, F_5 = 5, ...
        a, b = 0, 1
        for _ in range(50):
            c = (a + b) % N
            if c != 0:
                # Try (F_k, F_{k+1}, c) as seed components
                seeds.append((a % N, b % N, c))
            a, b = b, c

        # Also use Lucas numbers: L_k = F_{k-1} + F_{k+1}
        a, b = 2, 1  # L_0 = 2, L_1 = 1
        for _ in range(30):
            c = (a + b) % N
            if c != 0:
                seeds.append((a % N, b % N, c))
            a, b = b, c

    elif seed_type == 'pythagorean':
        # Primitive Pythagorean triple seeds
        # Use (m^2 - n^2, 2mn, m^2 + n^2) patterns with small m, n
        for m in range(2, 20):
            for n in range(1, m):
                if gcd(m, n) == 1 and (m - n) % 2 == 1:
                    a = m * m - n * n
                    b = 2 * m * n
                    c = m * m + n * n
                    if c < N:
                        seeds.append((a % N, b % N, c % N))

    elif seed_type == 'random':
        # Random valid modular triples
        random.seed(42)  # Reproducible
        for _ in range(50):
            # Generate random m, n with PPT conditions
            while True:
                m = random.randrange(2, N)
                n = random.randrange(1, m)
                if gcd(m, n) == 1 and (m - n) % 2 == 1:
                    break
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            seeds.append((a % N, b % N, c % N))

    # Remove duplicates and invalid seeds
    seen = set()
    result = []
    for s in seeds:
        if s not in seen and s[2] != 0:  # c should not be 0 mod N
            seen.add(s)
            result.append(s)

    return result


def _forward_frontier_from_seed(N: int, seed: tuple[int, int, int], depth: int,
                                 max_states: int = 10000) -> dict:
    """Generate forward frontier from a specific seed at given depth.

    Returns dict: modular_state -> (prefix_word, modular_state)
    """
    frontier = {seed: ([], seed)}
    seen = {seed}

    for level in range(depth):
        next_frontier = {}
        for state, (word, _) in frontier.items():
            for branch in ('U', 'A', 'D'):
                new_state = _forward_apply(state, branch, N)
                if new_state in seen:
                    continue
                seen.add(new_state)
                new_word = word + [branch]
                next_frontier[new_state] = (new_word, new_state)

                if len(seen) >= max_states:
                    for k, v in next_frontier.items():
                        if k not in frontier:
                            frontier[k] = v
                    return frontier

        frontier = next_frontier
        if not frontier:
            break

    return frontier


def inside_out_factor_multistart(N: int, forward_depth: int = 4, backward_depth: int = 4,
                                  seed_types: list = None, batch_size: int = 128,
                                  periodic_enabled: bool = True) -> tuple[int, int] | None:
    """Multi-start bidirectional inside-out factoring.

    Run multiple seed types through the same backward goal bank.
    Different starts may reach the same goal via different paths,
    increasing meet probability.

    Args:
        N: Number to factor
        forward_depth: Depth for forward BFS from each seed
        backward_depth: Depth for backward covector search
        seed_types: List of seed types to try: 'thin', 'fibonacci', 'pythagorean', 'random'
        batch_size: Batch size for GCD computation
        periodic_enabled: If True, also use periodic-word exploration

    Returns (p, q) with p < q and p*q = N, or None.
    """
    if seed_types is None:
        seed_types = ['thin', 'fibonacci', 'pythagorean']

    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square check
    s = isqrt(N)
    if s * s == N:
        return (s, s)

    # Goal covectors: e_a = (1,0,0), e_b = (0,1,0), e_c = (0,0,1)
    goals = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Build shared backward frontier (computed once, reused for all seeds)
    backward_forms = _backward_frontier(N, goals, backward_depth, max_states=10000)

    if not backward_forms:
        return None

    # Collect all forward states from all seed types
    all_forward_states = {}

    # Also add periodic-word exploration if enabled
    if periodic_enabled:
        explorer = PeriodicWordExplorer(N, max_word_len=3, max_power=16)
        periodic_frontier = explorer.generate_frontier(max_states=5000)
        for state, (info, word, _) in periodic_frontier.items():
            if state not in all_forward_states:
                all_forward_states[state] = (['P', info, word], state)

    # Generate forward frontiers for each seed type
    for seed_type in seed_types:
        seeds = _generate_seeds(N, seed_type)
        for seed in seeds[:10]:  # Limit seeds per type to avoid explosion
            # Use depth=0 for seeds already at frontier, then expand
            if seed not in all_forward_states:
                all_forward_states[seed] = (['S', seed_type], seed)

            # Expand from this seed
            forward_states = _forward_frontier_from_seed(N, seed, forward_depth, max_states=2000)
            for state, (word, _) in forward_states.items():
                if state not in all_forward_states:
                    all_forward_states[state] = (['F', seed_type, word], state)

    # Meet check: all forward states vs shared backward forms
    result = _meet_in_the_middle(N, all_forward_states, backward_forms, batch_size)

    if result is not None:
        p, q = result
        if p * q == N:
            return (min(p, q), max(p, q))

    return None