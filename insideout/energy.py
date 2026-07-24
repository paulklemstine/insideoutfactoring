"""Energy spectrum of Pythagorean tree nodes.

The energy of a node v = (a, b, c) is E(v) = ln(c). Since ln is
monotonic, we use c directly for comparisons — no floating point needed.

Energy is the key metric for the Inside-Out algorithm: prime factors
manifest as resonances in the energy spectrum, and we use energy bounds
to prune the search tree.
"""
from __future__ import annotations

from .berggren import Triple


def energy(triple: Triple) -> int:
    """Return the 'energy' of a triple, which is its hypotenuse c.

    Since E(v) = ln(c) and ln is monotonic, comparing energies is
    equivalent to comparing hypotenuses. We avoid computing ln() entirely.
    """
    return triple.c


def energy_gap(parent: Triple, child: Triple) -> int:
    """Compute the energy gap between parent and child.

    Returns c_child - c_parent (always positive for forward traversal).
    """
    return child.c - parent.c


def hypotenuse_bound(N: int) -> int:
    """Compute an upper bound on the hypotenuse of a triple containing N.

    For N = pq with p < q, the target triple has c = (q^2 + p^2)/2.
    Since p >= 1 and q <= N, we get c <= (N^2 + 1)/2.

    Returns a practical upper bound: if c > bound, the triple cannot
    factor N and can be pruned.
    """
    return (N * N + 1) // 2


def is_energy_compatible(N: int, triple: Triple) -> bool:
    """Check whether a triple's energy is compatible with factoring N.

    A triple (a, b, c) can only reveal factors of N if:
    1. c >= N (the hypotenuse must be at least as large as N)
    2. c <= (N^2 + 1)/2 (upper bound from the factorization)

    For energy pruning, we check both bounds.
    """
    c = triple.c
    lower = N  # c must be >= N
    upper = (N * N + 1) // 2  # c must be <= (N^2+1)/2
    return lower <= c <= upper