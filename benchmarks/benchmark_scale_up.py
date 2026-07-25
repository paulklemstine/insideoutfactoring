"""Scale-up benchmark for Inside-Out factoring.

Tests 32-bit through 128-bit semiprimes with increasing iteration limits.
Trial division gets a step limit at larger sizes since sqrt(N) is enormous.
"""
import time
import sys
from math import gcd, isqrt
from sympy import nextprime, isprime

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

from insideout.inside_out import inside_out_factor
from insideout.wavefront import search_wavefront
from insideout.factor import factor_with_method


def generate_semiprime(bits: int, offset: int = 0) -> tuple[int, int, int]:
    """Generate a semiprime with approximately `bits` bits.
    Returns (N, p, q) so we can verify results.
    """
    half = bits // 2
    p = nextprime((1 << half) + offset * 1000)
    q = nextprime(p + 1 + offset * 37)
    return p * q, p, q


def trial_division(N: int, max_steps: int = None) -> tuple[int, int] | None:
    """Trial division with optional step limit."""
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2)
    steps = 0
    for p in range(3, isqrt(N) + 1, 2):
        steps += 1
        if max_steps and steps > max_steps:
            return None
        if N % p == 0:
            return (min(p, N // p), max(p, N // p))
    return None


def timed_call(func, *args, timeout_sec=60, **kwargs):
    """Run func with a wall-clock timeout. Returns (result, elapsed_sec) or (None, elapsed_sec) on timeout."""
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        return (f"ERROR: {e}", time.perf_counter() - start)
    elapsed = time.perf_counter() - start
    if elapsed > timeout_sec:
        return (f"TIMEOUT after {elapsed:.1f}s", elapsed)
    return (result, elapsed)


def fmt_result(r, expected_p, expected_q):
    if r is None:
        return "FAILED"
    if isinstance(r, str):
        return r[:30]
    p, q = r
    if p == expected_p and q == expected_q:
        return f"OK ({p} * {q})"
    return f"WRONG ({p} * {q})"


def benchmark_bits(bits: int, num_samples: int = 3, max_io_iters=500000, max_wf_radius=1000, td_steps=None, time_cap=120):
    """Run benchmark for a given bit size."""
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {bits}-bit semiprimes (io_iters={max_io_iters}, wf_radius={max_wf_radius})")
    print(f"{'='*70}")

    results = []
    for i in range(num_samples):
        N, p, q = generate_semiprime(bits, offset=i)
        print(f"\n  Sample {i+1}: {N.bit_length()}-bit N, factors p={p.bit_length()}-bit, q={q.bit_length()}-bit")
        print(f"    p = {p}, q = {q}")

        # factor_with_method (default limits)
        r, t = timed_call(factor_with_method, N, timeout_sec=time_cap)
        print(f"    factor_with_method: {fmt_result(r, p, q):50s}  {t:.4f}s")
        res_default = (r, t)

        # inside_out with higher limits
        r, t = timed_call(inside_out_factor, N, max_iterations=max_io_iters, timeout_sec=time_cap)
        print(f"    inside_out ({max_io_iters}): {fmt_result(r, p, q):50s}  {t:.4f}s")
        res_io = (r, t)

        # wavefront with higher limits
        r, t = timed_call(search_wavefront, N, max_radius=max_wf_radius, timeout_sec=time_cap)
        print(f"    wavefront ({max_wf_radius}): {fmt_result(r, p, q):50s}  {t:.4f}s")
        res_wf = (r, t)

        # Trial division
        td_max = td_steps
        r, t = timed_call(trial_division, N, max_steps=td_max, timeout_sec=time_cap)
        td_label = f"trial_div (max {td_max} steps)" if td_max else "trial_div (unlimited)"
        print(f"    {td_label}: {fmt_result(r, p, q):50s}  {t:.4f}s")
        res_td = (r, t)

        results.append({
            "bits": N.bit_length(),
            "N": N, "p": p, "q": q,
            "default_result": res_default[0],
            "default_time": res_default[1],
            "io_result": res_io[0],
            "io_time": res_io[1],
            "wf_result": res_wf[0],
            "wf_time": res_wf[1],
            "td_result": res_td[0],
            "td_time": res_td[1],
        })

    return results


def main():
    print("=" * 70)
    print("INSIDE-OUT FACTORING: SCALE-UP BENCHMARK")
    print("=" * 70)

    try:
        import gmpy2
        print(f"gmpy2 available: v{gmpy2.version()}")
    except ImportError:
        print("gmpy2 NOT available")

    all_results = []

    # 32-bit: baseline (trial div should be fast)
    all_results.extend(benchmark_bits(32, num_samples=3, max_io_iters=500000, max_wf_radius=1000, td_steps=None, time_cap=30))

    # 48-bit: moderate (trial div still feasible, sqrt ~ 2^24 = 16M)
    all_results.extend(benchmark_bits(48, num_samples=3, max_io_iters=2000000, max_wf_radius=5000, td_steps=None, time_cap=60))

    # 64-bit: challenging (trial div needs sqrt(N) ~ 2^32 = 4B steps, way too slow)
    # Use step-limited trial div as comparison
    all_results.extend(benchmark_bits(64, num_samples=3, max_io_iters=5000000, max_wf_radius=10000, td_steps=10000000, time_cap=120))

    # 96-bit: very challenging
    all_results.extend(benchmark_bits(96, num_samples=2, max_io_iters=10000000, max_wf_radius=20000, td_steps=50000000, time_cap=180))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Bits':<6} {'Default':<12} {'IO':<12} {'WF':<12} {'TD':<12} {'IO(s)':<8} {'WF(s)':<8} {'TD(s)':<8} {'IO wins?'}")
    print("-" * 90)
    for r in all_results:
        default_ok = "OK" if r["default_result"] and not isinstance(r["default_result"], str) else "FAIL"
        io_ok = "OK" if r["io_result"] and not isinstance(r["io_result"], str) else "FAIL"
        wf_ok = "OK" if r["wf_result"] and not isinstance(r["wf_result"], str) else "FAIL"
        td_ok = "OK" if r["td_result"] and not isinstance(r["td_result"], str) else "FAIL"

        io_wins = "?"
        if io_ok == "OK" and td_ok == "OK":
            io_wins = "YES" if r["io_time"] < r["td_time"] else "NO"
        elif io_ok == "OK":
            io_wins = "IO-only"
        elif td_ok == "OK":
            io_wins = "TD-only"

        print(f"{r['bits']:<6} {default_ok:<12} {io_ok:<12} {wf_ok:<12} {td_ok:<12} {r['io_time']:<8.4f} {r['wf_time']:<8.4f} {r['td_time']:<8.4f} {io_wins}")


if __name__ == "__main__":
    main()