"""Self-Tuning Adaptive Portfolio — Parameter Optimization Based on Input Characteristics.

A self-tuning version of the adaptive portfolio that:
1. Estimates input difficulty based on bit-length
2. Selects optimal method parameters for each strategy
3. Uses early termination when any method succeeds

The key insight: different input sizes require different parameters.
- Small inputs (<=40 bits): low bounds, few curves
- Medium inputs (40-72 bits): moderate bounds, more curves
- Large inputs (72-128 bits): high bounds, many curves
"""
from __future__ import annotations

import time
from math import isqrt, gcd
from typing import Optional

from insideout.cyclotomic_resultant import cyclotomic_cascade_factor, cyclotomic_resultant_factor
from insideout.sl2_group_order import sl2_group_order_factor, sl2_structured_factor
from insideout.lucas_multi import lucas_multi_factor, crt_collision_factor
from insideout.resultant_cascade import discriminant_resonance_factor, quadratic_resonance_factor
from insideout.hensel_cascade import hensel_cascade_factor, crt_lattice_factor
from insideout.graph_order import graph_order_cascade_factor, order_spectrum_factor
from insideout.factor import factor_with_method
from insideout.resonance_cascade import resonance_cascade_factor
from insideout.lucas_ppt import lucas_ppt_factor
from insideout.spectral_factor import spectral_cascade_factor
from insideout.fibonacci_pythagorean import fibonacci_pythagorean_factor


def _estimate_difficulty(N: int) -> str:
    """Estimate difficulty based on bit-length."""
    bits = N.bit_length()
    if bits <= 40:
        return 'easy'
    elif bits <= 56:
        return 'medium'
    elif bits <= 72:
        return 'hard'
    else:
        return 'very_hard'


# Parameter grids for each difficulty level
_METHOD_PARAMS = {
    'cyclotomic_cascade': {
        'easy': dict(bound=10000, base_points=5),
        'medium': dict(bound=50000, base_points=10),
        'hard': dict(bound=100000, base_points=15),
        'very_hard': dict(bound=500000, base_points=20),
    },
    'sl2_group_order': {
        'easy': dict(bound=100000, curves=20),
        'medium': dict(bound=1000000, curves=100),
        'hard': dict(bound=5000000, curves=300),
        'very_hard': dict(bound=20000000, curves=1000),
    },
    'lucas_multi': {
        'easy': dict(bound=10000, max_params=10),
        'medium': dict(bound=50000, max_params=20),
        'hard': dict(bound=200000, max_params=50),
        'very_hard': dict(bound=500000, max_params=100),
    },
    'quadratic_resonance': {
        'easy': dict(bound=10000, bases=5),
        'medium': dict(bound=50000, bases=10),
        'hard': dict(bound=100000, bases=15),
        'very_hard': dict(bound=500000, bases=20),
    },
    'hensel_cascade': {
        'easy': dict(bound=10000, max_lifts=5, base_points=5),
        'medium': dict(bound=50000, max_lifts=10, base_points=10),
        'hard': dict(bound=100000, max_lifts=15, base_points=15),
        'very_hard': dict(bound=500000, max_lifts=20, base_points=20),
    },
    'graph_order': {
        'easy': dict(bound=10000, max_exp=20, bases=5),
        'medium': dict(bound=50000, max_exp=50, bases=8),
        'hard': dict(bound=100000, max_exp=50, bases=10),
        'very_hard': dict(bound=500000, max_exp=100, bases=15),
    },
    'spectral_cascade': {
        'easy': dict(bound=50000, max_iterations=1000),
        'medium': dict(bound=200000, max_iterations=5000),
        'hard': dict(bound=500000, max_iterations=10000),
        'very_hard': dict(bound=2000000, max_iterations=20000),
    },
}

# Method priority order for each difficulty
_METHOD_ORDER = {
    'easy': ['cyclotomic_cascade', 'lucas_multi', 'quadratic_resonance', 'sl2_group_order'],
    'medium': ['cyclotomic_cascade', 'sl2_group_order', 'lucas_multi', 'quadratic_resonance', 'hensel_cascade', 'graph_order'],
    'hard': ['sl2_group_order', 'cyclotomic_cascade', 'lucas_multi', 'spectral_cascade', 'quadratic_resonance', 'hensel_cascade'],
    'very_hard': ['sl2_group_order', 'spectral_cascade', 'cyclotomic_cascade', 'lucas_multi', 'quadratic_resonance', 'hensel_cascade', 'graph_order'],
}


def _run_method(method_name: str, params: dict, N: int):
    """Run a single method and return result."""
    try:
        if method_name == 'cyclotomic_cascade':
            return cyclotomic_cascade_factor(N, **params)
        elif method_name == 'sl2_group_order':
            return sl2_group_order_factor(N, **params)
        elif method_name == 'lucas_multi':
            return lucas_multi_factor(N, **params)
        elif method_name == 'quadratic_resonance':
            return quadratic_resonance_factor(N, **params)
        elif method_name == 'hensel_cascade':
            return hensel_cascade_factor(N, **params)
        elif method_name == 'graph_order':
            return graph_order_cascade_factor(N, **params)
        elif method_name == 'spectral_cascade':
            return spectral_cascade_factor(N, **params)
    except Exception:
        pass
    return None


def self_tuning_factor(N: int, time_budget_ms: float = 30000) -> tuple[int, int] | None:
    """Factor N using a self-tuning adaptive portfolio.

    Estimates difficulty, selects optimal parameters, runs methods in priority order.

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

    difficulty = _estimate_difficulty(N)
    method_names = _METHOD_ORDER[difficulty]

    start_time = time.perf_counter()

    for method_name in method_names:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > time_budget_ms * 0.9:
            break

        params = _METHOD_PARAMS[method_name][difficulty]

        # Time estimate for this method
        scale = 50 if method_name == 'sl2_group_order' else 5
        est_time = params.get('bound', 50000) / 10000 * scale
        if elapsed_ms + est_time > time_budget_ms:
            continue

        result = _run_method(method_name, params, N)
        if result is not None:
            p, q = result
            if p * q == N and 1 < p < N and 1 < q < N:
                return (min(p, q), max(p, q))

    # Fallback to standard factor function
    return factor_with_method(N)