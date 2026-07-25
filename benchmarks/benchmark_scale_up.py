"""Scale-up benchmark: focused timing data for Inside-Out vs trial division.

Usage: python benchmark_scale_up.py [32|48|64|96|128]
"""
import time
import sys
from math import isqrt
from sympy import nextprime

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

from insideout.inside_out import inside_out_factor
from insideout.wavefront import search_wavefront
from insideout.factor import factor_with_method
from insideout.cf_guide import cf_factor_check


def generate_semiprime(bits, offset=0):
    half = bits // 2
    p = nextprime((1 << half) + offset * 1000)
    q = nextprime(p + 1 + offset * 37)
    return p * q, p, q


def trial_division_timed(N, max_steps=None):
    """Returns (result_or_None, elapsed_seconds, steps_attempted)"""
    if N < 4:
        return None, 0, 0
    if N % 2 == 0:
        return (2, N // 2), 0, 1
    start = time.perf_counter()
    steps = 0
    for p in range(3, isqrt(N) + 1, 2):
        steps += 1
        if max_steps and steps > max_steps:
            elapsed = time.perf_counter() - start
            return None, elapsed, steps
        if N % p == 0:
            elapsed = time.perf_counter() - start
            return (min(p, N // p), max(p, N // p)), elapsed, steps
    elapsed = time.perf_counter() - start
    return None, elapsed, steps


def run_benchmark(bits, num_samples=3):
    print(f"\n{'='*70}")
    print(f"  {bits}-BIT SEMIPRIME BENCHMARK")
    print(f"{'='*70}")

    for i in range(num_samples):
        N, p, q = generate_semiprime(bits, offset=i)
        print(f"\n  Sample {i+1}: N={N.bit_length()}bits, p={p.bit_length()}bits, q={q.bit_length()}bits")
        print(f"    p={p}, q={q}")
        sqrt_N = isqrt(N)
        print(f"    sqrt(N)={sqrt_N} (~2^{sqrt_N.bit_length()-1})")

        # CF precheck
        start = time.perf_counter()
        cf_r = cf_factor_check(N)
        cf_t = time.perf_counter() - start
        cf_ok = "OK" if cf_r and cf_r[0] * cf_r[1] == N else "miss"
        print(f"    CF precheck:      {cf_ok:5s}  {cf_t:.6f}s")

        # Inside-Out at default (50K) and higher limits
        for max_it in [50000, 500000]:
            start = time.perf_counter()
            io_r = inside_out_factor(N, max_iterations=max_it)
            io_t = time.perf_counter() - start
            io_ok = "OK" if io_r and io_r[0] * io_r[1] == N else "FAIL"
            print(f"    Inside-Out({max_it:>8,}): {io_ok:5s}  {io_t:.4f}s")
            if io_t > 60:
                print(f"    (skipping higher iteration counts)")
                break

        # Wavefront at default (500) and higher limits
        for max_r in [500, 1000]:
            start = time.perf_counter()
            wf_r = search_wavefront(N, max_radius=max_r)
            wf_t = time.perf_counter() - start
            wf_ok = "OK" if wf_r and wf_r[0] * wf_r[1] == N else "FAIL"
            print(f"    Wavefront({max_r:>6,}):   {wf_ok:5s}  {wf_t:.4f}s")
            if wf_t > 60:
                break

        # Full pipeline (default limits)
        start = time.perf_counter()
        fp_r = factor_with_method(N)
        fp_t = time.perf_counter() - start
        if fp_r and not isinstance(fp_r, str):
            fp_method = fp_r[1]
            fp_ok = "OK" if fp_r[0][0] * fp_r[0][1] == N else "FAIL"
        else:
            fp_method = "?"
            fp_ok = "FAIL"
        print(f"    Pipeline(default): {fp_ok:5s}  method={fp_method:15s}  {fp_t:.4f}s")

        # Trial division (unlimited for <=48bit, limited for larger)
        if bits <= 48:
            td_r, td_t, td_steps = trial_division_timed(N)
            td_ok = "OK" if td_r else "FAIL"
            print(f"    Trial Division:    {td_ok:5s}  {td_t:.6f}s  ({td_steps:,} steps)")
        else:
            for max_steps in [10_000_000]:
                td_r, td_t, td_steps = trial_division_timed(N, max_steps=max_steps)
                td_ok = "OK" if td_r else f"LIMIT({max_steps:,})"
                print(f"    Trial Div(<= {max_steps:,}): {td_ok:20s}  {td_t:.4f}s  ({td_steps:,} steps checked)")
                if td_r:
                    break


if __name__ == "__main__":
    bits = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    try:
        import gmpy2
        print(f"gmpy2 available: v{gmpy2.version()}")
    except ImportError:
        print("gmpy2 NOT available")

    print(f"Python int size: unlimited (arbitrary precision)")

    run_benchmark(bits)