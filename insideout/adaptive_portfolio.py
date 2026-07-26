"""Thompson-Sampled Adaptive Portfolio — Bandit-Based Factoring Budget Allocation.

A bandit-based portfolio that allocates computational budget across multiple
factoring methods using Thompson sampling (Bayesian optimization over success rate).

Theory:
- Thompson sampling: maintain a Beta distribution over "success rate" for each method
- Each round: sample θ_i ~ Beta(α_i, β_i) for each method i, pick argmax
- Observe success/failure, update α/β: α += success, β += failure
- Over time, budget concentrates on methods that actually work for the given N

The portfolio:
1. Builds a pool of factoring methods with configurable time budgets
2. On each `adaptive_factor` call, Thompson-samples method selection
3. Tracks per-method statistics: successes, failures, costs, times
4. Provides `get_recommendations(N)` to suggest best methods for N's bit-size
"""
from __future__ import annotations

import random
import threading
import time
from bisect import bisect_right
from math import isqrt, log2
from typing import Callable, Optional

# ------------------------------------------------------------------
# Method registry
# ------------------------------------------------------------------

# Each method entry: (name, function, default_budget_ms, precond_fn)
# precond_fn checks cheaply whether the method is applicable (can be None)
MethodRegistry: list[tuple[str, Callable, float, Optional[Callable]]] = []


def _register(name: str, fn: Callable, budget_ms: float,
              precond: Optional[Callable] = None) -> None:
    """Register a factoring method with the global registry."""
    MethodRegistry.append((name, fn, budget_ms, precond))


# --- Import and register all methods ---

from .lucas_ppt import lucas_ppt_factor
from .projective_collision import chart_collision_factor as projective_chart_factor
from .projective_collision import projective_collision_factor
from .resonance_cascade import resonance_cascade_factor
from .sl2_group_order import sl2_group_order_factor
from .fibonacci_pythagorean import fibonacci_pythagorean_factor
from .cyclotomic_resultant import cyclotomic_resultant_factor
from .batch_crt_cascade import batch_crt_cascade_factor
from .brahmagupta import fermat_difference_of_squares
from .ppt_form_cascade import squfof_factor
from .fibonacci_factor import fibonacci_gcd_factor
from .spectral_factor import spectral_cascade_factor
from .cf_matrix_cascade import cf_matrix_cascade_factor, cf_cascade_factor
from .lucas_multi import lucas_multi_factor, crt_collision_factor
from .class_group_cascade import class_group_cascade_factor
from .graph_order import graph_order_cascade_factor
from .resultant_cascade import quadratic_resonance_factor
from .inside_out import inside_out_factor
from .wavefront import search_wavefront

# Register methods with name, function, time budget, precond
# (precond None = always applicable; precond returns reason string if inapplicable)
_register("lucas_ppt",         lucas_ppt_factor,          200.0)
_register("projective_chart",  projective_chart_factor,   300.0)
_register("projective_collision", projective_collision_factor, 400.0)
_register("resonance_cascade", resonance_cascade_factor,  500.0)
_register("sl2_group_order",   sl2_group_order_factor,    800.0)
_register("fib_pythagorean",   fibonacci_pythagorean_factor, 300.0)
_register("cyclotomic",        cyclotomic_resultant_factor, 600.0)
_register("batch_crt",         batch_crt_cascade_factor,  400.0)
_register("fermat",            fermat_difference_of_squares, 200.0)
_register("squfof",            squfof_factor,             300.0)
_register("fibonacci_gcd",     fibonacci_gcd_factor,       200.0)
_register("spectral_cascade",  spectral_cascade_factor,   400.0)
_register("cf_matrix_cascade", cf_matrix_cascade_factor,   500.0)
_register("cf_cascade",        cf_cascade_factor,          200.0)
_register("lucas_multi",       lucas_multi_factor,         300.0)
_register("crt_collision",     crt_collision_factor,       300.0)
_register("class_group",      class_group_cascade_factor,  500.0)
_register("graph_order",       graph_order_cascade_factor, 500.0)
_register("quadratic_resonance", quadratic_resonance_factor, 500.0)
_register("inside_out",        inside_out_factor,          800.0)
_register("wavefront",         search_wavefront,           500.0)


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _run_with_timeout(func: Callable, args=(), kwargs=None,
                      timeout_ms: float = 200.0) -> tuple:
    """Run func(*args, **kwargs) with a timeout.

    Returns (result, elapsed_ms, timed_out).
    """
    kwargs = kwargs or {}
    result = [None]
    elapsed = [0.0]
    timed_out = [False]

    def target():
        t0 = time.perf_counter()
        try:
            result[0] = func(*args, **kwargs)
        except Exception:
            result[0] = None
        elapsed[0] = (time.perf_counter() - t0) * 1000

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_ms / 1000.0)
    if t.is_alive():
        timed_out[0] = True
        return None, timeout_ms, True
    return result[0], elapsed[0], timed_out[0]


def _validate_result(result, N: int):
    """Return (p, q) if result is a valid factorization of N, else None."""
    if result is None:
        return None
    try:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return (min(p, q), max(p, q))
    except (TypeError, ValueError):
        pass
    return None


# ------------------------------------------------------------------
# AdaptivePortfolio
# ------------------------------------------------------------------

class AdaptivePortfolio:
    """Thompson-sampled adaptive portfolio for integer factorization.

    Attributes:
        priors: dict[str, tuple(alpha, beta)] — Beta prior for each method.
                 Starts as (1, 1) uniform prior.
        stats: dict[str, dict] — detailed statistics per method.
        history: list[dict] — log of all adaptive_factor calls.
        lock: threading.Lock for thread-safe updates.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self.lock = threading.Lock()

        # Beta priors: (successes, failures) — start with (1,1) uniform
        self.priors: dict[str, tuple[float, float]] = {}
        self.stats: dict[str, dict] = {}     # detailed per-method stats
        self.history: list[dict] = []          # run log

        for name, _, budget, _ in MethodRegistry:
            self.priors[name] = (1.0, 1.0)
            self.stats[name] = {
                "budget_ms": budget,
                "successes": 0,
                "failures": 0,
                "total_calls": 0,
                "total_time_ms": 0.0,
                "best_time_ms": float("inf"),
                "last_result": None,
            }

    # ------------------------------------------------------------------
    # Thompson sampling
    # ------------------------------------------------------------------

    def sample_success_rates(self) -> dict[str, float]:
        """Thompson-sample a success-rate estimate for each method.

        Returns {method_name: sampled_theta}.
        """
        import math
        samples = {}
        with self.lock:
            for name, (a, b) in self.priors.items():
                # Sample from Beta(a, b) using gamma approximation
                if a <= 0:
                    a = 1e-9
                if b <= 0:
                    b = 1e-9
                # Beta(a,b) ~ sum of a exponential(1) / sum of (a+b) exponential(1)
                # Simple approximation: sample from gamma then normalise
                g_a = self._gamma_sample(a, self._rng)
                g_b = self._gamma_sample(b, self._rng)
                samples[name] = g_a / (g_a + g_b) if (g_a + g_b) > 0 else 0.0
        return samples

    @staticmethod
    def _gamma_sample(shape: float, rng: random.Random) -> float:
        """Sample from Gamma(shape) using Marsaglia and Tsang's method."""
        if shape < 1.0:
            return AdaptivePortfolio._gamma_sample(shape + 1.0, rng) * rng.random() ** (1.0 / shape)
        d = shape - 1.0 / 3.0
        c = 1.0 / (9.0 * d) ** 0.5
        while True:
            x = rng.gauss(0.0, 1.0)
            v = 1.0 + c * x
            if v > 0.0:
                v2 = v * v * v
                u = rng.random()
                if u < 1.0 - 0.0331 * (x * x) ** 4:
                    return d * v2
                if log2(u) < 0.5 * x * x + d * (1.0 - v2 + log2(v2)):
                    return d * v2

    def update_posteriors(self, method_name: str,
                          success: bool,
                          cost_ms: float) -> None:
        """Update Beta posterior for method_name based on outcome.

        Args:
            method_name: name of the method
            success: True if method found a factor, False otherwise
            cost_ms: actual time consumed by the method
        """
        with self.lock:
            if method_name not in self.priors:
                return
            a, b = self.priors[method_name]
            if success:
                a += 1.0
            else:
                b += 1.0
            self.priors[method_name] = (a, b)

            s = self.stats[method_name]
            s["total_calls"] += 1
            s["total_time_ms"] += cost_ms
            if cost_ms < s["best_time_ms"]:
                s["best_time_ms"] = cost_ms
            if success:
                s["successes"] += 1
                s["last_result"] = "success"
            else:
                s["failures"] += 1
                s["last_result"] = "failure"

    # ------------------------------------------------------------------
    # Core allocation
    # ------------------------------------------------------------------

    def allocate_budget(self, N: int,
                       time_budget_ms: float = 5000.0,
                       time_per_method_ms: float = 500.0,
                       verbose: bool = False) -> tuple[tuple[int, int], str, float] | None:
        """Allocate the time budget using Thompson-sampled method ordering.

        Repeatedly:
          1. Thompson-sample success rates
          2. Pick the method with highest sampled rate
          3. Give it time_per_method_ms (or remaining budget, whichever is smaller)
          4. Run it
          5. Update posteriors
          6. Stop if factor found or budget exhausted

        Returns ((p, q), method_name, total_time_ms) or None.
        """
        start = time.perf_counter()
        remaining = time_budget_ms
        total_elapsed = 0.0

        tried = set()  # methods we've already tried this call

        while remaining > 50.0:
            # Thompson sample
            samples = self.sample_success_rates()

            # Filter to untried methods only
            available = {n: s for n, s in samples.items() if n not in tried}

            if not available:
                break  # all methods tried

            # Pick method with highest sampled success rate
            best_method = max(available, key=available.__getitem__)
            tried.add(best_method)

            # Look up method info
            method_fn = None
            for name, fn, budget, _ in MethodRegistry:
                if name == best_method:
                    method_fn = fn
                    break
            if method_fn is None:
                continue

            chunk = min(time_per_method_ms, remaining)

            if verbose:
                print(f"  [portfolio] trying {best_method} "
                      f"(θ={samples[best_method]:.3f}) "
                      f"budget={chunk:.0f}ms")

            # Run with timeout
            result, dt, timed_out = _run_with_timeout(
                method_fn, (N,), {}, timeout_ms=chunk
            )
            total_elapsed = (time.perf_counter() - start) * 1000

            # Validate result
            factors = _validate_result(result, N)

            self.update_posteriors(best_method, success=(factors is not None), cost_ms=dt)

            if factors is not None:
                p, q = factors
                return ((p, q), best_method, total_elapsed)

            remaining = time_budget_ms - total_elapsed

        return None

    def adaptive_factor(self, N: int,
                       time_budget_ms: float = 5000.0) -> tuple[tuple[int, int], str, float] | None:
        """Factor N using Thompson-sampled adaptive portfolio.

        Returns ((p, q), method_name, time_ms) on success, or None.
        """
        # Quick preconditions: even N, perfect square, small trial division
        if N < 4:
            return None
        if N % 2 == 0:
            if N == 2:
                return None
            return ((2, N // 2), "even", 0.0)
        s = isqrt(N)
        if s * s == N and s > 1:
            return ((s, s), "perfect_square", 0.0)

        # Quick trial division
        for p in range(3, min(s + 1, 1000), 2):
            if N % p == 0:
                return ((p, N // p), "trial_division", 0.0)

        # Delegate to allocate_budget
        result = self.allocate_budget(N, time_budget_ms)

        if result is not None:
            factors, method, elapsed = result
            self._log_run(N, method, success=True, elapsed_ms=elapsed,
                          factors=factors)
            return result
        else:
            self._log_run(N, "none", success=False, elapsed_ms=time_budget_ms)
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def try_all_methods(self, N: int,
                        time_per_method_ms: float = 500.0) -> tuple[tuple[int, int], str, float] | None:
        """Try all methods in parallel (threaded) and return the first success.

        Returns ((p, q), method_name, wall_clock_time_ms) or None.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if N < 4:
            return None
        if N % 2 == 0:
            if N == 2:
                return None
            return ((2, N // 2), "even", 0.0)
        s = isqrt(N)
        if s * s == N:
            return ((s, s), "perfect_square", 0.0)
        for p in range(3, min(s + 1, 1000), 2):
            if N % p == 0:
                return ((p, N // p), "trial_division", 0.0)

        start = time.perf_counter()

        def run_one(name_fn):
            name, fn, budget, _ = name_fn
            chunk = min(time_per_method_ms, budget)
            result, dt, _ = _run_with_timeout(fn, (N,), {}, timeout_ms=chunk)
            factors = _validate_result(result, N)
            return name, dt, factors

        with ThreadPoolExecutor(max_workers=min(8, len(MethodRegistry))) as ex:
            futures = {ex.submit(run_one, mf): mf[0] for mf in MethodRegistry}
            for fut in as_completed(futures):
                try:
                    name, dt, factors = fut.result()
                    if factors is not None:
                        elapsed = (time.perf_counter() - start) * 1000
                        p, q = factors
                        self.update_posteriors(name, success=True, cost_ms=dt)
                        return ((p, q), name, elapsed)
                except Exception:
                    pass

        return None

    def get_recommendations(self, N: int,
                            top_k: int = 3) -> list[tuple[str, float, float]]:
        """Recommend the top-k best methods for an N of similar bit-size.

        Returns [(method_name, expected_success_rate, empirical_avg_ms)].
        Uses Thompson posterior mean: α/(α+β) as expected success rate.
        """
        with self.lock:
            bit_size = int(log2(N)) + 1 if N > 0 else 1
            recommendations = []
            for name, (a, b) in self.priors.items():
                total = a + b
                mean_success_rate = a / total if total > 0 else 0.0
                s = self.stats[name]
                avg_time = (s["total_time_ms"] / s["total_calls"]
                            if s["total_calls"] > 0 else float("inf"))
                recommendations.append((
                    name,
                    mean_success_rate,
                    avg_time,
                ))

        # Sort by descending success rate
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:top_k]

    def _log_run(self, N: int, method: str,
                 success: bool, elapsed_ms: float,
                 factors: Optional[tuple] = None) -> None:
        """Append a run record to self.history."""
        record = {
            "N": N,
            "N_bits": int(log2(N)) + 1 if N > 0 else 0,
            "method": method,
            "success": success,
            "elapsed_ms": elapsed_ms,
            "timestamp": time.time(),
        }
        if factors is not None:
            record["factors"] = factors
        with self.lock:
            self.history.append(record)

    def summary(self) -> dict:
        """Return a human-readable summary of current posteriors and stats."""
        with self.lock:
            rows = []
            for name, (a, b) in self.priors.items():
                s = self.stats[name]
                calls = s["total_calls"]
                rate = (s["successes"] / calls) if calls > 0 else 0.0
                avg_time = (s["total_time_ms"] / calls) if calls > 0 else 0.0
                rows.append({
                    "method": name,
                    "α": a, "β": b,
                    "θ_mean": a / (a + b) if (a + b) > 0 else 0.0,
                    "calls": calls,
                    "successes": s["successes"],
                    "failures": s["failures"],
                    "success_rate": rate,
                    "avg_time_ms": avg_time,
                    "best_time_ms": s["best_time_ms"],
                })
            rows.sort(key=lambda r: r["θ_mean"], reverse=True)
            return {"methods": rows, "total_runs": len(self.history)}


# ------------------------------------------------------------------
# Module-level singleton for convenience
# ------------------------------------------------------------------

_portfolio: Optional[AdaptivePortfolio] = None


def _get_portfolio() -> AdaptivePortfolio:
    global _portfolio
    if _portfolio is None:
        _portfolio = AdaptivePortfolio(seed=42)
    return _portfolio


def adaptive_factor(N: int,
                    time_budget_ms: float = 5000.0) -> tuple[tuple[int, int], str, float] | None:
    """Factor N using the shared Thompson-sampled adaptive portfolio.

    Thread-safe singleton: all calls share the same posterior state so
    the portfolio learns across multiple invocations.
    """
    return _get_portfolio().adaptive_factor(N, time_budget_ms)


def try_all_methods(N: int,
                    time_per_method_ms: float = 500.0) -> tuple[tuple[int, int], str, float] | None:
    """Try all registered methods concurrently, return first success."""
    return _get_portfolio().try_all_methods(N, time_per_method_ms)


def get_recommendations(N: int, top_k: int = 3) -> list[tuple[str, float, float]]:
    """Return top-k recommended methods for N's bit-size."""
    return _get_portfolio().get_recommendations(N, top_k)


def sample_success_rates() -> dict[str, float]:
    """Thompson-sample success rates from the shared portfolio."""
    return _get_portfolio().sample_success_rates()


def update_posteriors(method_name: str, success: bool, cost_ms: float) -> None:
    """Update the shared portfolio's posterior for a method."""
    _get_portfolio().update_posteriors(method_name, success, cost_ms)


def portfolio_summary() -> dict:
    """Return the shared portfolio's current summary."""
    return _get_portfolio().summary()
