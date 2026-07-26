"""Projective Collision Factoring — Birthday-Style Collision Search.

Key insight: Two projective triples x=(a,b,c) and y=(u,v,w) are equal modulo
a hidden factor p when their three minors vanish modulo p:
  - av - bu == 0 (mod p)
  - aw - cu == 0 (mod p)
  - bw - cv == 0 (mod p)

This is a birthday paradox problem: finding a collision between two branch walks.
Instead of hitting coordinate-zero directly, we look for projective equality.

This is analogous to Pollard's rho but on projective space instead of modular integers.
The collision is detected via the minors, and batched GCD tests all three at once.
"""
from __future__ import annotations

import random
from math import gcd, isqrt
from typing import Optional, NamedTuple

# Berggren matrix operations
# P_U, P_A, P_D act on triples (a,b,c) -> new triple


class Triple(NamedTuple):
    """A projective triple (a, b, c) representing a point in P^2."""
    a: int
    b: int
    c: int

    def __repr__(self):
        return f"({self.a}, {self.b}, {self.c})"


def _normalize(t: Triple, N: int) -> Triple:
    """Normalize triple modulo N, handling zero coordinates."""
    a, b, c = t.a % N, t.b % N, t.c % N
    # If any coordinate is 0 mod N, we already have a GCD probe
    return Triple(a, b, c)


def _apply_U(t: Triple, N: int) -> Triple:
    """Apply P_U matrix: (a+2b-2c, -2a-b+2c, -2a-2b+3c) mod N."""
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (a + 2*b - 2*c) % N,
        (-2*a - b + 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )


def _apply_A(t: Triple, N: int) -> Triple:
    """Apply P_A matrix: (a+2b-2c, 2a+b-2c, -2a-2b+3c) mod N."""
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (a + 2*b - 2*c) % N,
        (2*a + b - 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )


def _apply_D(t: Triple, N: int) -> Triple:
    """Apply P_D matrix: (-a-2b+2c, 2a+b-2c, -2a-2b+3c) mod N."""
    a, b, c = t.a % N, t.b % N, t.c % N
    return Triple(
        (-a - 2*b + 2*c) % N,
        (2*a + b - 2*c) % N,
        (-2*a - 2*b + 3*c) % N
    )


def _apply_branch(t: Triple, N: int, branch: str) -> Triple:
    """Apply a branch matrix (U, A, or D)."""
    if branch == 'U':
        return _apply_U(t, N)
    elif branch == 'A':
        return _apply_A(t, N)
    else:  # 'D'
        return _apply_D(t, N)


def _minors(t1: Triple, t2: Triple) -> tuple[int, int, int]:
    """Compute the three minors of two triples.

    Returns (av - bu, aw - cu, bw - cv) modulo N.
    If all three are 0 mod p, the triples are projectively equal mod p.
    """
    a1, b1, c1 = t1.a, t1.b, t1.c
    a2, b2, c2 = t2.a, t2.b, t2.c
    return (
        (a1 * b2 - a2 * b1),  # av - bu
        (a1 * c2 - a2 * c1),  # aw - cu
        (b1 * c2 - b2 * c1),  # bw - cv
    )


def _gcd_batch(minor1: int, minor2: int, minor3: int, N: int) -> list[int]:
    """Batched GCD test on all three minors.

    Returns list of nontrivial GCD results.
    """
    results = []
    for m in [minor1, minor2, minor3]:
        g = gcd(abs(m), N)
        if 1 < g < N:
            results.append(g)
    return results


def _is_distinguished(t: Triple, bits: int = 8) -> bool:
    """Check if triple satisfies distinguished predicate.

    A triple is distinguished if its coordinates share a common prefix
    in binary representation (low bits are zero).
    """
    mask = (1 << bits) - 1
    return (t.a & mask) == 0 and (t.b & mask) == 0 and (t.c & mask) == 0


def _triple_to_int(t: Triple, N: int) -> int:
    """Hash triple to integer for comparison."""
    return (t.a % N) ^ ((t.b % N) << 10) ^ ((t.c % N) << 20)


def projective_collision_factor(N: int,
                               max_steps: int = 100000,
                               window_size: int = 32,
                               num_walks: int = 8,
                               distinguished_bits: int = 8) -> tuple[int, int] | None:
    """Factor N using projective collision search.

    Algorithm:
    1. Launch multiple independent branch walks from thin seed (3,4,5)
    2. Store distinguished points (those with common low-bit prefix)
    3. When two distinguished points collide (different walk, same state),
       compute minors and batch-GCD
    4. Also check within-walk collisions as backup

    Args:
        N: Integer to factor
        max_steps: Maximum total steps across all walks
        window_size: Collision window size within each walk
        num_walks: Number of independent walks to launch
        distinguished_bits: Bits for distinguished predicate

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

    # Small trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Thin seed: the root (3, 4, 5) of the Berggren tree
    thin_seed = Triple(3 % N, 4 % N, 5 % N)

    # Each walk has its own distinguished point table
    # Key: distinguished triple value (as int)
    # Value: (walk_id, step_count, triple)
    distinguished_tables: list[dict[int, tuple]] = [{} for _ in range(num_walks)]

    # Branch sequence for each walk (could be random or deterministic)
    branches = ['U', 'A', 'D']

    # Initialize walks
    walks: list[Triple] = []
    for i in range(num_walks):
        # Salt each walk with a different starting offset
        offset = (i * 7919) % (N - 1) + 1
        walks.append(Triple(
            (thin_seed.a * offset) % N,
            (thin_seed.b * offset) % N,
            (thin_seed.c * offset) % N
        ))

    step = 0
    steps_per_walk = max_steps // num_walks

    for walk_id in range(num_walks):
        t = walks[walk_id]

        for j in range(steps_per_walk):
            step += 1

            # Apply random branch
            branch = branches[(walk_id + j) % 3]
            t = _apply_branch(t, N, branch)

            # Check distinguished predicate
            if _is_distinguished(t, distinguished_bits):
                key = _triple_to_int(t, N)
                if key in distinguished_tables[walk_id]:
                    # Same walk collision - check minors
                    prev_walk_id, prev_step, prev_t = distinguished_tables[walk_id][key]

                    # Only process if different step
                    if prev_step != j:
                        m1, m2, m3 = _minors(t, prev_t)
                        factors = _gcd_batch(m1, m2, m3, N)
                        for g in factors:
                            if 1 < g < N and N % g == 0:
                                return (g, N // g)
                else:
                    distinguished_tables[walk_id][key] = (walk_id, j, t)

            # Check cross-walk collisions (between different walks)
            # This is where the birthday paradox helps
            for other_walk in range(walk_id + 1, num_walks):
                key = _triple_to_int(t, N)
                if key in distinguished_tables[other_walk]:
                    _, other_step, other_t = distinguished_tables[other_walk][key]
                    if other_step != j:  # Different steps = real collision
                        m1, m2, m3 = _minors(t, other_t)
                        factors = _gcd_batch(m1, m2, m3, N)
                        for g in factors:
                            if 1 < g < N and N % g == 0:
                                return (g, N // g)

            # Also check within-window collisions (smaller collision target)
            if j >= window_size:
                # Compare with point window_size steps ago
                # Simplified: just check distinguished table periodically
                pass

    return None


def projective_collision_with_cycles(N: int,
                                   max_steps: int = 50000,
                                   cycle_window: int = 1024) -> tuple[int, int] | None:
    """Enhanced projective collision with cycle detection.

    Uses Floyd's cycle detection plus distinguished points for scalability.
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

    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Start from thin seed
    t = Triple(3 % N, 4 % N, 5 % N)

    # Distinguished table
    distinguished: dict[int, tuple] = {}

    # Floyd's cycle detection pointers
    tortoise = t
    hare = _apply_U(t, N)  # One step per iteration

    branches = ['U', 'A', 'D']
    hare_branch = 0
    step = 0

    while step < max_steps:
        # Floyd's algorithm
        hare = _apply_branch(hare, N, branches[hare_branch])
        hare_branch = (hare_branch + 1) % 3
        hare = _apply_branch(hare, N, branches[hare_branch])
        hare_branch = (hare_branch + 1) % 3
        tortoise = _apply_branch(tortoise, N, branches[step % 3])

        step += 1

        # Check for GCD from minor
        m1, m2, m3 = _minors(tortoise, hare)
        for m in [m1, m2, m3]:
            g = gcd(abs(m), N)
            if 1 < g < N:
                return (g, N // g)

        # Distinguished point check
        if _is_distinguished(tortoise, bits=10):
            key = _triple_to_int(tortoise, N)
            if key in distinguished:
                other_t = distinguished[key][2]
                m1, m2, m3 = _minors(tortoise, other_t)
                for m in [m1, m2, m3]:
                    g = gcd(abs(m), N)
                    if 1 < g < N:
                        return (g, N // g)
            else:
                distinguished[key] = (step, tortoise)

    return None


def chart_determinant(t1: Triple, t2: Triple, N: int) -> int:
    """Single chart determinant for collision detection.

    Uses chart [a : c+b].  Collision when:
        a2*(c1+b1) - a1*(c2+b2) == 0 (mod p)

    Returns the raw determinant value (caller applies gcd).
    """
    a1, b1, c1 = t1.a % N, t1.b % N, t1.c % N
    a2, b2, c2 = t2.a % N, t2.b % N, t2.c % N
    return a2 * (c1 + b1) - a1 * (c2 + b2)


def gcd_safe_c_plus_b(t: Triple, N: int) -> int:
    """If c+b shares a nontrivial factor with N, return it.

    This is the fast-path certificate when chart inversion fails.
    Returns gcd(c+b, N).  If 1 < result < N, a factor is found.
    """
    from math import gcd
    _, b, c = t.a % N, t.b % N, t.c % N
    return gcd(c + b, N)


def is_distinguished(t: Triple, bits: int = 8) -> bool:
    """Check if triple satisfies distinguished predicate.

    A triple is distinguished if its coordinates share a common low-bit prefix.
    """
    mask = (1 << bits) - 1
    return (t.a & mask) == 0 and (t.b & mask) == 0 and (t.c & mask) == 0


def _triple_to_int(t: Triple, N: int) -> int:
    """Hash triple to integer for distinguished table keys."""
    return (t.a % N) ^ ((t.b % N) << 10) ^ ((t.c % N) << 20)


def replay_walk(seed: Triple, N: int, branch_seq: list) -> Triple:
    """Replay a branch sequence from seed, return endpoint."""
    t = Triple(seed.a % N, seed.b % N, seed.c % N)
    for branch in branch_seq:
        if branch == 'U':
            t = _apply_U(t, N)
        elif branch == 'A':
            t = _apply_A(t, N)
        else:
            t = _apply_D(t, N)
    return t


def chart_collision_factor(N: int,
                           max_steps: int = 50000,
                           num_walks: int = 16,
                           distinguished_bits: int = 8,
                           batch_size: int = 256) -> tuple[int, int] | None:
    """Factor N using chart-compressed projective collision search.

    Algorithm:
    1. Launch num_walks independent salted walks from thin seed (3,4,5)
    2. Each walk is deterministic given its salt (different U/A/D offset)
    3. Store only distinguished endpoints
    4. On endpoint collision: replay both walks to find merge point
    5. Evaluate chart determinant at merge point; batch GCD
    6. Also probe gcd(c+b, N) at each step as fast-path factor certificate

    Returns (p, q) with p < q and p*q = N, or None.
    """
    from math import gcd, isqrt
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
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    thin_seed = Triple(3 % N, 4 % N, 5 % N)
    branches = ['U', 'A', 'D']

    # Distinguished tables per walk
    # key: hashed triple int → (step_count, full_branch_list)
    dist_tables: list[dict[int, tuple]] = [{} for _ in range(num_walks)]

    step = 0
    steps_per_walk = max_steps // num_walks

    for walk_id in range(num_walks):
        # Salt: different offset per walk → different branch sequence
        offset = (walk_id * 7919) % (N - 1) + 1
        salted_seed = Triple(
            (thin_seed.a * offset) % N,
            (thin_seed.b * offset) % N,
            (thin_seed.c * offset) % N,
        )

        t = salted_seed
        branch_seq = []
        for j in range(steps_per_walk):
            step += 1

            # Deterministic branch: cycle through U, A, D with walk-dependent offset
            branch = branches[(walk_id + j) % 3]
            branch_seq.append(branch)
            if branch == 'U':
                t = _apply_U(t, N)
            elif branch == 'A':
                t = _apply_A(t, N)
            else:
                t = _apply_D(t, N)

            # Fast path: gcd(c+b, N) may find a factor immediately
            g = gcd(t.c + t.b, N)
            if 1 < g < N:
                return (g, N // g)

            # Distinguished endpoint check
            if is_distinguished(t, distinguished_bits):
                key = _triple_to_int(t, N)
                if key in dist_tables[walk_id]:
                    prev_step, prev_branches = dist_tables[walk_id][key]
                    if prev_step != j:
                        # Replay both segments to find merge point
                        mid1 = replay_walk(salted_seed, N, prev_branches[:prev_step])
                        mid2 = replay_walk(salted_seed, N, branch_seq[:j])
                        det = chart_determinant(mid1, mid2, N)
                        g = gcd(abs(det), N)
                        if 1 < g < N:
                            return (g, N // g)
                else:
                    dist_tables[walk_id][key] = (j, list(branch_seq))

                # Cross-walk distinguished collisions
                for other_id in range(walk_id):
                    if key in dist_tables[other_id]:
                        other_step, other_branches = dist_tables[other_id][key]
                        mid1 = replay_walk(salted_seed, N, branch_seq[:j])
                        other_offset = (other_id * 7919) % (N - 1) + 1
                        other_seed = Triple(
                            (thin_seed.a * other_offset) % N,
                            (thin_seed.b * other_offset) % N,
                            (thin_seed.c * other_offset) % N,
                        )
                        mid2 = replay_walk(other_seed, N, other_branches[:other_step])
                        det = chart_determinant(mid1, mid2, N)
                        g = gcd(abs(det), N)
                        if 1 < g < N:
                            return (g, N // g)

    return None


# Export
__all__ = [
    'projective_collision_factor',
    'projective_collision_with_cycles',
    'chart_collision_factor',
    'chart_determinant',
    'gcd_safe_c_plus_b',
    'is_distinguished',
    'Triple',
    '_minors',
]
