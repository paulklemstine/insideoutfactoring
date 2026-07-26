"""Benchmark with UNBALANCED semiprimes (large factor gap).

The real test: p is small (e.g., 8-bit) and q is large, making
Fermat's method fail and forcing other methods to work.

Usage: python bench_gap.py [32|48|64]
"""
import time
import sys
from math import isqrt
from sympy import nextprime

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

from insideout.cf_guide import cf_factor_check
from insideout.brahmagupta import fermat_difference_of_squares, brahmagupta_fibonacci_factor
from insideout.fibonacci_factor import fibonacci_gcd_factor
from insideout.inside_out import inside_out_factor
from insideout.factor import factor_with_method


def generate_unbalanced_semiprime(p_bits, q_bits, offset=0):
    """Generate semiprime where p has p_bits and q has q_bits."""
    p = nextprime((1 << (p_bits - 1)) + offset * 100)
    q = nextprime((1 << (q_bits - 1)) + offset * 37)
    N = p * q
    return N, p, q


def trial_division_timed(N, max_steps=None):
    if N < 4:
        return None, 0, 0
    if N % 2 == 0:
        return (2, N // 2), 0, 1
    start = time.perf_counter()
    steps = 0
    for p in range(3, isqrt(N) + 1, 2):
        steps += 1
        if max_steps and steps > max_steps:
            return None, time.perf_counter() - start, steps
        if N % p == 0:
            return (min(p, N // p), max(p, N // p)), time.perf_counter() - start, steps
    return None, time.perf_counter() - start, steps


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    try:
        r = func(*args, **kwargs)
        return r, time.perf_counter() - start
    except Exception as e:
        return f"ERROR: {e}", time.perf_counter() - start


def check(r, N):
    """Check if a factorization result is correct."""
    if r is None:
        return "miss"
    if isinstance(r, str):
        return r[:20]
    # factor_with_method returns ((p, q), method_name)
    if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], tuple):
        (p, q), method = r
        return "OK" if p * q == N and 1 < p < N else "WRONG"
    # Direct factorization returns (p, q)
    if isinstance(r, tuple) and len(r) == 2:
        p, q = r
        if isinstance(p, int) and isinstance(q, int):
            return "OK" if p * q == N and 1 < p < N else "WRONG"
    return "?"


print("UNBALANCED SEMIPRIME BENCHMARK (hard for Fermat)")
print("=" * 70)

try:
    import gmpy2
    print(f"gmpy2: v{gmpy2.version()}")
except ImportError:
    print("gmpy2: NOT available")

# Test configurations: (p_bits, q_bits) pairs
configs = [
    # Easy unbalanced: 8-bit p, various q sizes
    (8, 16),   # 24-bit semiprime, small factor
    (8, 24),   # 32-bit with tiny factor
    (8, 32),   # 40-bit with tiny factor
    (8, 40),   # 48-bit with tiny factor
    (8, 56),   # 64-bit with tiny factor
    # Moderate unbalanced
    (12, 20),  # ~32-bit with 12-bit factor
    (16, 32),  # ~48-bit with 16-bit factor
    (16, 48),  # 64-bit with 16-bit factor
    # More balanced but still unbalanced
    (12, 36),  # ~48-bit with 12-bit factor
    (24, 40),  # 64-bit with 24-bit factor
]

for p_bits, q_bits in configs:
    print(f"\n--- p={p_bits}bits, q={q_bits}bits ---")
    N, p, q = generate_unbalanced_semiprime(p_bits, q_bits)
    gap = q - p
    print(f"  N={N.bit_length()}bits, p={p} ({p.bit_length()}b), gap={gap}, sqrt(N)~2^{isqrt(N).bit_length()-1}")

    # CF precheck
    r, t = timed(cf_factor_check, N)
    print(f"  CF precheck:       {check(r, N):10s}  {t:.6f}s")

    # Fermat (should fail for unbalanced)
    r, t = timed(fermat_difference_of_squares, N)
    print(f"  Fermat:            {check(r, N):10s}  {t:.6f}s")

    # Fibonacci
    r, t = timed(fibonacci_gcd_factor, N, bound=5000)
    print(f"  Fibonacci(5K):     {check(r, N):10s}  {t:.6f}s")

    # Brahmagupta (only for small N)
    if N.bit_length() <= 40:
        r, t = timed(brahmagupta_fibonacci_factor, N)
        print(f"  Brahmagupta:      {check(r, N):10s}  {t:.6f}s")

    # Inside-Out (with time limit)
    for max_it in [50000, 500000]:
        start = time.perf_counter()
        r = inside_out_factor(N, max_iterations=max_it)
        t = time.perf_counter() - start
        print(f"  Inside-Out({max_it:>8,}): {check(r, N):10s}  {t:.4f}s")
        if t > 30:
            break

    # Trial division (with step limit)
    max_steps = min(isqrt(N) + 1, 100_000_000)
    r, t, s = trial_division_timed(N, max_steps=max_steps)
    td_status = "OK" if r else f"LIMIT({s:,})"
    print(f"  Trial Div(<=1M):   {td_status:20s}  {t:.4f}s")

    # Full pipeline
    r, t = timed(factor_with_method, N)
    ok = check(r, N)
    method = r[1] if isinstance(r, tuple) and len(r) == 2 and isinstance(r[0], tuple) else "?"
    print(f"  Pipeline:          {ok:10s}  method={str(method):15s}  {t:.4f}s")