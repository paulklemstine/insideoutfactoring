"""Benchmark script for Inside-Out factoring.

Measures performance across increasing bit sizes and compares
against trial division baseline.
"""
import time
from math import isqrt
from sympy import nextprime

from insideout.factor import factor, factor_with_method


def generate_semiprime(bits: int) -> int:
    """Generate a semiprime with approximately `bits` bits."""
    from sympy import isprime
    # Find two primes near 2^(bits/2)
    p = nextprime(2 ** (bits // 2))
    q = nextprime(p)
    return p * q


def trial_division(N: int) -> tuple[int, int] | None:
    """Simple trial division baseline."""
    if N < 4:
        return None
    if N % 2 == 0:
        return (2, N // 2)
    for p in range(3, isqrt(N) + 1, 2):
        if N % p == 0:
            return (p, N // p)
    return None


def benchmark(bits: int, num_samples: int = 3):
    """Benchmark Inside-Out vs trial division for given bit size."""
    print(f"\n{'='*60}")
    print(f"Bit size: {bits}")
    print(f"{'='*60}")

    for i in range(num_samples):
        N = generate_semiprime(bits)
        print(f"\n  Sample {i+1}: N = {N} ({N.bit_length()} bits)")

        # Inside-Out
        start = time.perf_counter()
        result_io = factor_with_method(N)
        time_io = time.perf_counter() - start

        # Trial division
        start = time.perf_counter()
        result_td = trial_division(N)
        time_td = time.perf_counter() - start

        if result_io:
            factors, method = result_io
            print(f"  Inside-Out: {factors[0]} * {factors[1]} = {factors[0]*factors[1]} "
                  f"(method: {method}, time: {time_io:.6f}s)")
        else:
            print(f"  Inside-Out: FAILED ({time_io:.6f}s)")

        if result_td:
            print(f"  Trial Div:  {result_td[0]} * {result_td[1]} = {result_td[0]*result_td[1]} "
                  f"(time: {time_td:.6f}s)")
        else:
            print(f"  Trial Div:  FAILED ({time_td:.6f}s)")

        if time_io > 0 and time_td > 0:
            speedup = time_td / time_io
            print(f"  Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    for bits in [8, 16, 24, 32]:
        benchmark(bits)