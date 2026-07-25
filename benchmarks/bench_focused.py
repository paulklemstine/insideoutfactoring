"""Focused scale-up benchmark: per-method timing at 32/48/64/96/128 bits.
Skips Brahmagupta at 64+ bits (it's O(sqrt(N)) and would hang).
"""
import time
import sys
from math import isqrt
from sympy import nextprime

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

from insideout.cf_guide import cf_factor_check
from insideout.brahmagupta import fermat_difference_of_squares
from insideout.fibonacci_factor import fibonacci_gcd_factor
from insideout.inside_out import inside_out_factor
from insideout.wavefront import search_wavefront
from insideout.factor import factor_with_method


def generate_semiprime(bits, offset=0):
    half = bits // 2
    p = nextprime((1 << half) + offset * 1000)
    q = nextprime(p + 1 + offset * 37)
    return p * q, p, q


def timed(func, *args, **kwargs):
    """Run func and return (result, seconds). Returns (None, seconds) on timeout/error."""
    start = time.perf_counter()
    try:
        r = func(*args, **kwargs)
        return r, time.perf_counter() - start
    except Exception as e:
        return f"ERROR: {e}", time.perf_counter() - start


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


def run_all(bits, num_samples=3, skip_brahmagupta=False, skip_io=False, td_steps=None):
    print(f"\n{'='*70}")
    print(f"  {bits}-BIT SEMIPRIME BENCHMARK ({num_samples} samples)")
    print(f"{'='*70}")

    results = []
    for i in range(num_samples):
        N, p, q = generate_semiprime(bits, offset=i)
        print(f"\n  --- Sample {i+1}: N={N.bit_length()}bits, p={p.bit_length()}bits, q={q.bit_length()}bits ---")
        print(f"      p={p}, q={q}, gap={q-p}")

        row = {"bits": N.bit_length(), "p": p, "q": q, "gap": q-p}

        # CF precheck (always fast)
        r, t = timed(cf_factor_check, N)
        row["cf_ok"] = r is not None and r[0]*r[1]==N if r else False
        row["cf_time"] = t
        print(f"    CF precheck:      {'OK' if row['cf_ok'] else 'miss':5s}  {t:.6f}s")

        # Brahmagupta (skip for 64+ bits)
        if not skip_brahmagupta:
            from insideout.brahmagupta import brahmagupta_fibonacci_factor
            r, t = timed(brahmagupta_fibonacci_factor, N)
            row["brah_ok"] = r is not None and r[0]*r[1]==N if r else False
            row["brah_time"] = t
            print(f"    Brahmagupta:      {'OK' if row['brah_ok'] else 'miss':5s}  {t:.6f}s")

        # Fermat (fast for close factors)
        r, t = timed(fermat_difference_of_squares, N)
        row["fermat_ok"] = r is not None and r[0]*r[1]==N if r else False
        row["fermat_time"] = t
        print(f"    Fermat:           {'OK' if row['fermat_ok'] else 'miss':5s}  {t:.6f}s")

        # Fibonacci (fast, bounded by Pisano period)
        r, t = timed(fibonacci_gcd_factor, N, bound=5000)
        row["fib_ok"] = r is not None and r[0]*r[1]==N if r else False
        row["fib_time"] = t
        print(f"    Fibonacci(5K):    {'OK' if row['fib_ok'] else 'miss':5s}  {t:.6f}s")

        # Inside-Out (skip for 96+ bits as it's very slow)
        if not skip_io:
            r, t = timed(inside_out_factor, N, max_iterations=50000)
            row["io_ok"] = r is not None and r[0]*r[1]==N if r else False
            row["io_time"] = t
            print(f"    Inside-Out 50K:   {'OK' if row['io_ok'] else 'FAIL':5s}  {t:.4f}s")

        # Full pipeline
        r, t = timed(factor_with_method, N)
        if r and not isinstance(r, str):
            method = r[1]
            ok = r[0][0]*r[0][1]==N
        else:
            method = "?"
            ok = False
        row["pipeline_ok"] = ok
        row["pipeline_method"] = method
        row["pipeline_time"] = t
        print(f"    Pipeline:        {'OK' if ok else 'FAIL':5s}  method={method:15s}  {t:.4f}s")

        # Trial division (with step limit for large N)
        if td_steps:
            r, t, s = trial_division_timed(N, max_steps=td_steps)
            td_ok = r is not None
            td_label = f"TD(<= {td_steps:,})"
        else:
            r, t, s = trial_division_timed(N)
            td_ok = r is not None
            td_label = f"TD(unlimited)"
        row["td_ok"] = td_ok
        row["td_time"] = t
        row["td_steps"] = s
        print(f"    {td_label:20s}{'OK' if td_ok else 'LIMIT/FAIL':5s}  {t:.4f}s  ({s:,} steps)")

        results.append(row)

    return results


if __name__ == "__main__":
    try:
        import gmpy2
        print(f"gmpy2 available: v{gmpy2.version()}")
    except ImportError:
        print("gmpy2 NOT available")

    all_results = []

    # 32-bit
    all_results.extend(run_all(32, num_samples=3))

    # 48-bit
    all_results.extend(run_all(48, num_samples=3, td_steps=None))

    # 64-bit (skip Brahmagupta which is O(sqrt(N)))
    all_results.extend(run_all(64, num_samples=3, skip_brahmagupta=True, skip_io=False, td_steps=10_000_000))

    # 96-bit (skip slow methods)
    all_results.extend(run_all(96, num_samples=2, skip_brahmagupta=True, skip_io=True, td_steps=100_000_000))

    # Summary
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Bits':<6} {'CF':<5} {'Fermat':<7} {'Fib':<5} {'IO':<5} {'Pipeline':<10} {'Method':<16} {'IO(s)':<8} {'TD(s)':<8}")
    print("-" * 80)
    for r in all_results:
        cf = "OK" if r.get("cf_ok") else "miss"
        fermat = "OK" if r.get("fermat_ok") else "miss"
        fib = "OK" if r.get("fib_ok") else "miss"
        io = "OK" if r.get("io_ok") else ("skip" if "io_ok" not in r else "FAIL")
        pipe = "OK" if r.get("pipeline_ok") else "FAIL"
        method = r.get("pipeline_method", "?")
        io_t = f"{r.get('io_time', '-'):.4f}" if "io_time" in r else "-"
        td_t = f"{r.get('td_time', '-'):.4f}" if "td_time" in r else "-"
        print(f"{r['bits']:<6} {cf:<5} {fermat:<7} {fib:<5} {io:<5} {pipe:<10} {method:<16} {io_t:<8} {td_t:<8}")