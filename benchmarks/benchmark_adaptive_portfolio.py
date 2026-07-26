"""Fast benchmark for adaptive_portfolio factoring.

Runs semiprimes of varying bit sizes through the adaptive_factor pipeline
with reduced time budgets for faster completion.
"""
import time
import random
import json
import sys
from math import isqrt, log2
from sympy import randprime, nextprime
from typing import Optional

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

from insideout.adaptive_portfolio import adaptive_factor
from insideout.factor import factor_with_method


def generate_semiprime(bits: int, seed: Optional[int] = None) -> tuple[int, int, int]:
    """Generate a random semiprime with approximately equal factors.

    Returns (N, p, q) where N = p * q and p < q.
    """
    if seed is not None:
        random.seed(seed)

    half = bits // 2
    # Generate primes in the appropriate range
    p = randprime(1 << (half - 1), 1 << half)
    q = randprime(1 << (half - 1), 1 << half)

    # Ensure p != q
    while q == p:
        q = randprime(1 << (half - 1), 1 << half)

    p, q = min(p, q), max(p, q)
    N = p * q
    return N, p, q


def run_benchmark_suite():
    """Run benchmarks across different bit sizes."""

    # Test configurations - reduced samples for speed
    configs = [
        (32, 3),   # (bits, num_samples)
        (40, 3),
        (48, 3),
        (56, 2),
        (64, 2),
        (72, 2),
        (80, 1),
    ]

    all_results = []

    for bits, num_samples in configs:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING {bits}-BIT SEMIPRIMES ({num_samples} samples)")
        print(f"{'='*70}")

        method_success_counts = {}

        for sample in range(num_samples):
            seed = bits * 1000 + sample
            N, p, q = generate_semiprime(bits, seed=seed)

            print(f"\n--- Sample {sample+1}/{num_samples}: {bits} bits ---")
            print(f"  N = {N}")
            print(f"  p = {p} ({p.bit_length()} bits)")
            print(f"  q = {q} ({q.bit_length()} bits)")
            print(f"  ratio q/p = {q/p:.4f}")

            # Run adaptive_factor with 30s budget
            start = time.perf_counter()
            result = adaptive_factor(N, time_budget_ms=30000)
            elapsed = (time.perf_counter() - start) * 1000

            if result is not None:
                factors, method, method_time = result
                fp, fq = factors
                success = fp * fq == N
            else:
                success = False
                method = "FAILED"
                method_time = elapsed

            print(f"  Result: {'OK' if success else 'FAIL'} ({elapsed:.2f}ms)")
            print(f"    Method: {method} ({method_time:.2f}ms)")

            # Track stats
            method_success_counts[method] = method_success_counts.get(method, 0) + (1 if success else 0)

            all_results.append({
                "bits": bits,
                "sample": sample + 1,
                "N": N,
                "p": p,
                "q": q,
                "ratio_q_p": q / p,
                "adaptive_success": success,
                "adaptive_method": method,
                "adaptive_time_ms": elapsed,
                "adaptive_method_time_ms": method_time,
            })

        # Summary for this bit size
        print(f"\n--- {bits}-BIT SUMMARY ---")
        print(f"  Method distribution: {method_success_counts}")

    return all_results


def analyze_results(all_results):
    """Analyze results and generate summary."""

    print(f"\n{'='*70}")
    print("FINAL BENCHMARK SUMMARY")
    print(f"{'='*70}")

    # Group by bit size
    by_bits = {}
    for r in all_results:
        bits = r["bits"]
        if bits not in by_bits:
            by_bits[bits] = []
        by_bits[bits].append(r)

    # Table
    print("\n{:<8} {:>10} {:>12} {:>12} {:>12}".format(
        "Bits", "Samples", "Success%", "Avg(ms)", "Max(ms)"))
    print("-" * 60)

    for bits in sorted(by_bits.keys()):
        results = by_bits[bits]
        total = len(results)
        successful = sum(1 for r in results if r["adaptive_success"])
        avg_time = sum(r["adaptive_time_ms"] for r in results) / total
        max_time = max(r["adaptive_time_ms"] for r in results)

        print("{:<8} {:>10} {:>11.1f}% {:>12.2f} {:>12.2f}".format(
            bits, total, successful / total * 100, avg_time, max_time))

    # Method statistics
    print("\n" + "="*70)
    print("METHOD SUCCESS STATISTICS")
    print("="*70)

    method_stats = {}
    for r in all_results:
        m = r["adaptive_method"]
        if m not in method_stats:
            method_stats[m] = {"successes": 0, "total": 0, "total_time": 0}
        method_stats[m]["total"] += 1
        method_stats[m]["total_time"] += r["adaptive_time_ms"]
        if r["adaptive_success"]:
            method_stats[m]["successes"] += 1

    # Sort by success rate
    sorted_methods = sorted(
        method_stats.items(),
        key=lambda x: (x[1]["successes"], -x[1]["total_time"] / max(x[1]["total"], 1)),
        reverse=True
    )

    print("\n{:<30} {:>10} {:>10} {:>12}".format(
        "Method", "Success", "Total", "Avg(ms)"))
    print("-" * 65)
    for method, stats in sorted_methods:
        if stats["total"] > 0:
            print("{:<30} {:>10} {:>10} {:>12.2f}".format(
                method, stats["successes"], stats["total"],
                stats["total_time"] / stats["total"]))

    # Hardest cases
    print("\n" + "="*70)
    print("HARDEST CASES")
    print("="*70)
    sorted_by_time = sorted(all_results, key=lambda x: x["adaptive_time_ms"], reverse=True)
    for i, r in enumerate(sorted_by_time[:5], 1):
        print(f"  {i}. {r['bits']:3d} bits, {r['adaptive_time_ms']:12.2f}ms, method={r['adaptive_method']}")
        print(f"       N = {r['N']}")

    # Recommendations
    print("\n" + "="*70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*70)

    # Find methods with failures
    failures = set()
    for r in all_results:
        if not r["adaptive_success"]:
            failures.add(r["adaptive_method"])

    if failures:
        print("\nMethods that failed in benchmarks:")
        for m in sorted(failures):
            print(f"  - {m}")

    # Methods to potentially prune (expensive but never succeed)
    expensive_failures = {}
    for r in all_results:
        if not r["adaptive_success"] and r["adaptive_time_ms"] > 100:
            m = r["adaptive_method"]
            if m not in expensive_failures:
                expensive_failures[m] = 0
            expensive_failures[m] += 1

    if expensive_failures:
        print("\nExpensive methods with no successes (consider pruning):")
        for m, count in sorted(expensive_failures.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {m}: {count} failures")

    return {
        "all_results": all_results,
        "method_stats": method_stats,
        "by_bits": by_bits,
    }


if __name__ == "__main__":
    print("ADAPTIVE PORTFOLIO BENCHMARK SUITE")
    print("="*70)

    # Run benchmark
    all_results = run_benchmark_suite()

    # Analyze
    results = analyze_results(all_results)

    # Save results
    output_file = "/home/raver1975/insideoutfactoring/benchmarks/adaptive_portfolio_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "all_results": all_results,
            "method_stats": results["method_stats"],
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")

    # Summary
    total = len(all_results)
    successful = sum(1 for r in all_results if r["adaptive_success"])
    print(f"\nTotal: {total} tests, {successful} successes ({successful/total*100:.1f}%)")
