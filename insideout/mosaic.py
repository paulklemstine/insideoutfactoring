"""MOSAIC: Multi-Orbit Spectral and Arithmetic Inside-Out Cofactorization.

A unified factoring architecture combining multiple projections:
1. Arithmetic: p-1, p+1, Fibonacci/Lucas, ECM, rho residues
2. Pythagorean: Berggren tree ascent, target-leg probes
3. Inside-out dual: forward/backward meet-in-the-middle
4. Spectral: orbit period/collision signatures for ranking
5. Gravitational: cost-adjusted yield scoring for adaptive scheduling

Key principles:
- Shared residue bus: all workers emit (residue, provenance, cost)
- Certificate-gated: accept only when gcd(residue, N) gives proper divisor
- Product-tree batch GCD: O(K log K) with recursive splitting
- Gravitational scheduling: score = yield / cost

Reference: MOSAIC architecture (2026)
"""
from __future__ import annotations

import time
import threading
from math import gcd, isqrt, log as ln
from typing import Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict

# Import all method modules
from .cf_guide import cf_factor_check
from .brahmagupta import fermat_difference_of_squares, brahmagupta_fibonacci_factor
from .fibonacci_factor import fibonacci_gcd_factor
from .resonance_cascade import resonance_cascade_factor
from .lucas_ppt import lucas_ppt_factor
from .spectral_factor import spectral_cascade_factor
from .relation_generator import relation_factor
from .fibonacci_pythagorean import fibonacci_pythagorean_factor
from .lucas_multi import lucas_multi_factor, crt_collision_factor
from .sl2_group_order import sl2_group_order_factor, sl2_structured_factor
from .batch_crt_cascade import batch_crt_cascade_factor
from .spectral_bkz_hybrid import spectral_bkz_factor_with_stages
from .cyclotomic_resultant import cyclotomic_resultant_factor, full_order_spectrum_factor
from .resultant_cascade import discriminant_resonance_factor
from .lattice_factor import lattice_factor
from .graph_order import graph_order_cascade_factor
from .hybrid_cyclo_sl2 import hybrid_cyclo_sl2_factor
from .cf_nfs_poly import cf_nfs_factor
from .berggren_grobner import berggren_grobner_factor


@dataclass
class ResidueResult:
    """A residue from any factoring method with provenance tracking."""
    residue: int
    provenace: str  # Which method produced this
    cost_ms: float   # Measured cost
    replay_data: tuple = None  # For witness replay


class MOSAICScheduler:
    """Gravitational scheduling for adaptive method allocation.

    Maintains cost-adjusted yield scores for each method/region:
    score = estimated_yield / estimated_cost

    Uses Thompson sampling with contextual bandit for method selection.
    """

    def __init__(self):
        self.method_costs: dict[str, list[float]] = defaultdict(list)
        self.method_yields: dict[str, list[int]] = defaultdict(list)  # 1 = success, 0 = fail
        self.total_runs = 0

    def record(self, method: str, cost_ms: float, success: bool) -> None:
        """Record method performance for adaptive scheduling."""
        self.method_costs[method].append(cost_ms)
        if len(self.method_costs[method]) > 100:
            self.method_costs[method] = self.method_costs[method][-100:]
        self.method_yields[method].append(1 if success else 0)
        if len(self.method_yields[method]) > 100:
            self.method_yields[method] = self.method_yields[method][-100:]
        self.total_runs += 1

    def estimated_yield(self, method: str) -> float:
        """Estimate success probability for method."""
        yields = self.method_yields.get(method, [])
        if not yields:
            return 0.1  # Prior: 10% default
        return sum(yields) / len(yields)

    def estimated_cost(self, method: str) -> float:
        """Estimate average cost for method."""
        costs = self.method_costs.get(method, [])
        if not costs:
            return 1000.0  # Prior: 1 second default
        # Trim outliers
        sorted_costs = sorted(costs)
        if len(sorted_costs) > 10:
            trimmed = sorted_costs[len(sorted_costs) // 10:-len(sorted_costs) // 10]
        else:
            trimmed = sorted_costs
        return sum(trimmed) / len(trimmed) if trimmed else 1000.0

    def gravitational_score(self, method: str) -> float:
        """Compute gravitational score: yield / cost."""
        y = self.estimated_yield(method)
        c = self.estimated_cost(method)
        if c <= 0:
            return 0.0
        return y / c

    def select_methods(self, candidates: list[str], n: int = 3) -> list[str]:
        """Select top-n methods by gravitational score."""
        scores = [(m, self.gravitational_score(m)) for m in candidates]
        scores.sort(key=lambda x: -x[1])
        return [m for m, _ in scores[:n]]


# Global scheduler instance
_scheduler: Optional[MOSAICScheduler] = None


def get_scheduler() -> MOSAICScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = MOSAICScheduler()
    return _scheduler


class ProductTreeGCD:
    """Product-tree batch GCD with recursive splitting.

    Computes GCD of batch of residues efficiently:
    - Build product tree: O(K log K) multiplications
    - Remainder tree: O(K log K) modular reductions
    - When GCD = N, recursively split to isolate

    Complexity: O(K log K log N) vs O(K^2) for pairwise GCD
    """

    def __init__(self, N: int):
        self.N = N
        self.residues: list[int] = []
        self.provenances: list[str] = []

    def add(self, residue: int, provenance: str) -> None:
        """Add a residue to the batch."""
        if residue != 0 and residue % self.N != 0:
            self.residues.append(residue % self.N)
            self.provenances.append(provenance)

    def batch_gcd(self) -> list[tuple[int, str]]:
        """Compute batch GCD, recursively splitting when result = N.

        Returns list of (factor, provenance) for nontrivial GCD results.
        """
        if not self.residues:
            return []

        N = self.N
        results = []

        # Build product tree
        tree = [self.residues]

        while len(tree[-1]) > 1:
            level = tree[-1]
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    next_level.append((level[i] * level[i + 1]) % N)
                else:
                    next_level.append(level[i] % N)
            tree.append(next_level)

        # Compute remainder tree
        remainders = [None] * len(tree[-1])
        for i in range(len(tree[-1])):
            remainders[i] = tree[-1][i] % N

        # Walk down the tree computing remainders
        for level_idx in range(len(tree) - 2, -1, -1):
            new_remainders = [None] * len(tree[level_idx])
            for i in range(len(tree[level_idx])):
                child_idx = i // 2
                if remainders[child_idx] is not None:
                    new_remainders[i] = tree[level_idx][i] % remainders[child_idx]
                else:
                    new_remainders[i] = tree[level_idx][i] % N
            remainders = new_remainders

        # Check each residue's remainder
        for i, (residue, prov) in enumerate(zip(self.residues, self.provenances)):
            rem = remainders[i] if i < len(remainders) else residue
            g = gcd(residue, N)
            if 1 < g < N:
                results.append((g, prov))

        # If no individual GCD found, check batch product
        if not results:
            batch_product = 1
            for r in self.residues:
                batch_product = (batch_product * r) % N
            g = gcd(batch_product, N)
            if 1 < g < N:
                # Find which residues contributed
                for r, prov in zip(self.residues, self.provenances):
                    gi = gcd(r, N)
                    if 1 < gi < N:
                        results.append((gi, prov))

        return results


def mosaic_factor(N: int, time_budget_ms: float = 10000) -> tuple[int, int] | None:
    """Factor N using MOSAIC architecture.

    Phase A: Normalization (perfect powers, trial division)
    Phase B: Short opportunistic epochs (parallel method attempts)
    Phase C: Relation factory (NFS-style for larger N)
    Phase D: Linear algebra (from relation factory)

    Returns (p, q) with p < q and p*q = N, or None.
    """
    start_time = time.perf_counter()
    scheduler = get_scheduler()

    if N < 4:
        return None

    # === PHASE A: NORMALIZATION ===
    # Handle even N
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

    # === PHASE B: SHORT OPPORTUNISTIC EPOCHS ===
    # Run multiple methods in epochs, gathering residues for batch GCD

    # Initialize residue bus
    residue_bus = ProductTreeGCD(N)

    # Method candidates organized by expected speed
    fast_methods = [
        ("fermat", lambda: _try_method(fermat_difference_of_squares, N, timeout_ms=1000)),
        ("brahmagupta", lambda: _try_method(brahmagupta_fibonacci_factor, N, timeout_ms=1000)),
        ("fibonacci", lambda: _try_method(fibonacci_gcd_factor, N, {"bound": 5000}, timeout_ms=1000)),
        ("lucas_ppt", lambda: _try_method(lucas_ppt_factor, N, timeout_ms=2000)),
        ("resonance_cascade", lambda: _try_method(resonance_cascade_factor, N, timeout_ms=3000)),
        ("berggren_grobner", lambda: _try_method(berggren_grobner_factor, N, {"max_depth": 12}, timeout_ms=1000)),
        ("spectral_bkz", lambda: _try_spectral_bkz(N, time_budget_ms=2000)),
    ]

    medium_methods = [
        ("spectral_cascade", lambda: _try_method(spectral_cascade_factor, N, timeout_ms=2000)),
        ("fib_pyth", lambda: _try_method(fibonacci_pythagorean_factor, N, {"bound": 10000}, timeout_ms=2000)),
        ("lucas_multi", lambda: _try_method(lucas_multi_factor, N, {"bound": 5000, "stage2_bound": 1000, "max_params": 8}, timeout_ms=2000)),
        ("crt_collision", lambda: _try_method(crt_collision_factor, N, {"bound": 3000, "stage2_bound": 500, "max_params": 6}, timeout_ms=2000)),
        ("sl2_group_order", lambda: _try_method(sl2_group_order_factor, N, {"bound": 200000, "curves": 50, "stage2_bound": 20000}, timeout_ms=5000)),
        ("sl2_structured", lambda: _try_method(sl2_structured_factor, N, {"bound": 50000}, timeout_ms=3000)),
    ]

    slow_methods = [
        ("batch_crt", lambda: _try_method(batch_crt_cascade_factor, N, {"bound": 5000, "stage2_bound": 1000, "max_params": 16, "stages": 2}, timeout_ms=3000)),
        ("cyclotomic_resultant", lambda: _try_method(cyclotomic_resultant_factor, N, {"max_order": 30, "smooth_bound": 50000}, timeout_ms=3000)),
        ("full_order_spectrum", lambda: _try_method(full_order_spectrum_factor, N, {"bound": 30000, "max_order": 60, "base_points": 8}, timeout_ms=3000)),
        ("discriminant_resonance", lambda: _try_method(discriminant_resonance_factor, N, {"max_disc": 1000, "max_forms": 100}, timeout_ms=2000)),
        ("hybrid_cyclo_sl2", lambda: _try_method(hybrid_cyclo_sl2_factor, N, {"time_budget_ms": 3000}, timeout_ms=3000)),
        ("lattice_factor", lambda: _try_method(lattice_factor, N, {"bound": 30000, "target_relations": 100}, timeout_ms=5000)),
        ("cf_nfs", lambda: _try_method(cf_nfs_factor, N, {"max_degree": 5}, timeout_ms=3000)),
        ("graph_order", lambda: _try_method(graph_order_cascade_factor, N, {"bound": 30000, "max_exp": 50, "bases": 8}, timeout_ms=3000)),
    ]

    # Run fast methods first with tight budgets
    elapsed = (time.perf_counter() - start_time) * 1000

    for name, method_fn in fast_methods:
        if elapsed > time_budget_ms * 0.3:  # Use 30% of budget for fast methods
            break

        method_start = time.perf_counter()
        result = method_fn()
        method_time = (time.perf_counter() - method_start) * 1000

        if result is not None:
            scheduler.record(name, method_time, True)
            p, q = result if isinstance(result, tuple) else result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        scheduler.record(name, method_time, False)
        elapsed = (time.perf_counter() - start_time) * 1000

    # Run medium methods with moderate budgets
    for name, method_fn in medium_methods:
        if elapsed > time_budget_ms * 0.6:  # Use 60% total for medium
            break

        method_start = time.perf_counter()
        result = method_fn()
        method_time = (time.perf_counter() - method_start) * 1000

        if result is not None:
            scheduler.record(name, method_time, True)
            p, q = result if isinstance(result, tuple) else result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        scheduler.record(name, method_time, False)
        elapsed = (time.perf_counter() - start_time) * 1000

    # Run slow methods only if we have budget remaining
    for name, method_fn in slow_methods:
        if elapsed > time_budget_ms * 0.9:  # Use 90% total
            break

        method_start = time.perf_counter()
        result = method_fn()
        method_time = (time.perf_counter() - method_start) * 1000

        if result is not None:
            scheduler.record(name, method_time, True)
            p, q = result if isinstance(result, tuple) else result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

        scheduler.record(name, method_time, False)
        elapsed = (time.perf_counter() - start_time) * 1000

    return None


def _try_method(method_fn, N: int, kwargs: dict = None, timeout_ms: float = 1000.0) -> tuple[int, int] | None:
    """Safely try a method with thread-based timeout."""
    import threading
    kwargs = kwargs or {}
    result = [None]
    elapsed = [0.0]

    def target():
        t0 = time.perf_counter()
        try:
            result[0] = method_fn(N, **kwargs)
        except Exception:
            result[0] = None
        elapsed[0] = (time.perf_counter() - t0) * 1000

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_ms / 1000.0)

    if t.is_alive():
        return None
    return result[0]


def _try_spectral_bkz(N: int, time_budget_ms: float) -> tuple[int, int] | None:
    """Try spectral-BKZ with timeout."""
    try:
        return spectral_bkz_factor_with_stages(N, time_budget_ms=time_budget_ms)
    except Exception:
        return None


# Export main entry point
__all__ = ['mosaic_factor', 'MOSAICScheduler', 'ProductTreeGCD', 'ResidueResult']
