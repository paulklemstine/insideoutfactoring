"""Streamlined 64-bit and 96-bit benchmark.
Focuses on pipeline methods and trial division comparison.
"""
import time
import sys
from math import isqrt
from sympy import nextprime

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

from insideout.factor import factor_with_method
from insideout.cf_guide import cf_factor_check
from insideout.brahmagupta import brahmagupta_fibonacci_factor, fermat_difference_of_squares
from insideout.fibonacci_factor import fibonacci_gcd_factor
from insideout.inside_out import inside_out_factor


def generate_semiprime(bits, offset=0):
    half = bits // 2
    p = nextprime((1 << half) + offset * 1000)
    q = nextprime(p + 1 + offset * 37)
    return p * q, p, q


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


def run_64bit():
    print("64-BIT SEMIPRIME BENCHMARK")
    print("=" * 70)

    for i in range(5):
        N, p, q = generate_semiprime(64, offset=i)
        print(f"\n  Sample {i+1}: N={N.bit_length()}bits, p={p.bit_length()}bits, q={q.bit_length()}bits")
        print(f"    p={p}, q={q}")

        # Individual pipeline methods (fast ones first)
        for name, func, kwargs in [
            ("CF precheck", lambda N: cf_factor_check(N), {}),
            ("Brahmagupta", lambda N: brahmagupta_fibonacci_factor(N), {}),
            ("Fermat", lambda N: fermat_difference_of_squares(N), {}),
            ("Fibonacci", lambda N: fibonacci_gcd_factor(N, bound=5000), {}),
        ]:
            start = time.perf_counter()
            r = func(N)
            t = time.perf_counter() - start
            ok = "OK" if r and r[0] * r[1] == N else "miss"
            print(f"    {name:15s}: {ok:5s}  {t:.6f}s")

        # Inside-Out with default limit (50K)
        start = time.perf_counter()
        r = inside_out_factor(N, max_iterations=50000)
        t = time.perf_counter() - start
        ok = "OK" if r and r[0] * r[1] == N else "FAIL"
        print(f"    Inside-Out 50K: {ok:5s}  {t:.4f}s")

        # Full pipeline
        start = time.perf_counter()
        r = factor_with_method(N)
        t = time.perf_counter() - start
        if r and not isinstance(r, str):
            method = r[1]
            ok = "OK" if r[0][0] * r[0][1] == N else "FAIL"
        else:
            method = "?"
            ok = "FAIL"
        print(f"    Pipeline:       {ok:5s}  method={method:15s}  {t:.4f}s")

        # Trial division (10M steps max)
        td_r, td_t, td_s = trial_division_timed(N, max_steps=10_000_000)
        td_ok = "OK" if td_r else f"LIMIT({td_s:,})"
        print(f"    Trial Div 10M:  {td_ok:20s}  {td_t:.4f}s")


def run_96bit():
    print("\n\n96-BIT SEMIPRIME BENCHMARK")
    print("=" * 70)

    for i in range(3):
        N, p, q = generate_semiprime(96, offset=i)
        print(f"\n  Sample {i+1}: N={N.bit_length()}bits, p={p.bit_length()}bits, q={q.bit_length()}bits")

        # Individual pipeline methods (fast ones only)
        for name, func in [
            ("CF precheck", lambda N: cf_factor_check(N)),
            ("Brahmagupta", lambda N: brahmagupta_fibonacci_factor(N)),
            ("Fermat", lambda N: fermat_difference_of_squares(N)),
            ("Fibonacci", lambda N: fibonacci_gcd_factor(N, bound=10000)),
        ]:
            start = time.perf_counter()
            r = func(N)
            t = time.perf_counter() - start
            ok = "OK" if r and r[0] * r[1] == N else "miss"
            print(f"    {name:15s}: {ok:5s}  {t:.6f}s")

        # Inside-Out with default limit
        start = time.perf_counter()
        r = inside_out_factor(N, max_iterations=50000)
        t = time.perf_counter() - start
        ok = "OK" if r and r[0] * r[1] == N else "FAIL"
        print(f"    Inside-Out 50K: {ok:5s}  {t:.4f}s")

        # Full pipeline
        start = time.perf_counter()
        r = factor_with_method(N)
        t = time.perf_counter() - start
        if r and not isinstance(r, str):
            method = r[1]
            ok = "OK" if r[0][0] * r[0][1] == N else "FAIL"
        else:
            method = "?"
            ok = "FAIL"
        print(f"    Pipeline:       {ok:5s}  method={method:15s}  {t:.4f}s")


def run_128bit():
    print("\n\n128-BIT SEMIPRIME BENCHMARK")
    print("=" * 70)

    for i in range(3):
        N, p, q = generate_semiprime(128, offset=i)
        print(f"\n  Sample {i+1}: N={N.bit_length()}bits, p={p.bit_length()}bits, q={q.bit_length()}bits")

        for name, func in [
            ("CF precheck", lambda N: cf_factor_check(N)),
            ("Brahmagupta", lambda N: brahmagupta_fibonacci_factor(N)),
            ("Fermat", lambda N: fermat_difference_of_squares(N)),
            ("Fibonacci", lambda N: fibonacci_gcd_factor(N, bound=10000)),
        ]:
            start = time.perf_counter()
            r = func(N)
            t = time.perf_counter() - start
            ok = "OK" if r and r[0] * r[1] == N else "miss"
            print(f"    {name:15s}: {ok:5s}  {t:.6f}s")

        # Full pipeline
        start = time.perf_counter()
        r = factor_with_method(N)
        t = time.perf_counter() - start
        if r and not isinstance(r, str):
            method = r[1]
            ok = "OK" if r[0][0] * r[0][1] == N else "FAIL"
        else:
            method = "?"
            ok = "FAIL"
        print(f"    Pipeline:       {ok:5s}  method={method:15s}  {t:.4f}s")


if __name__ == "__main__":
    try:
        import gmpy2
        print(f"gmpy2 available: v{gmpy2.version()}")
    except ImportError:
        print("gmpy2 NOT available")

    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    if bits == 64:
        run_64bit()
    elif bits == 96:
        run_96bit()
    elif bits == 128:
        run_128bit()
    else:
        print(f"Unknown bit size: {bits}")