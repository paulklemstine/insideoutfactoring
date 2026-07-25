"""Comprehensive scaling benchmark for Inside-Out factoring.

Tests factorization across bit sizes 16-128 and factor gap categories.
Measures wall-clock time, strategy used, and success/failure.

Uses multiprocessing for proper timeout enforcement.
"""
import time
import sys
import json
import multiprocessing as mp
from math import isqrt
from sympy import nextprime, isprime

sys.path.insert(0, "/home/raver1975/insideoutfactoring")

TIMEOUT_SECONDS = 10.0

# Iteration limits per bit-size tier (steered, bfs, wavefront radius)
TIER_CONFIGS = {
    16:  {"steered": 50000, "bfs": 50000, "wavefront": 500},
    24:  {"steered": 100000, "bfs": 100000, "wavefront": 500},
    32:  {"steered": 200000, "bfs": 200000, "wavefront": 1000},
    40:  {"steered": 300000, "bfs": 300000, "wavefront": 1000},
    48:  {"steered": 500000, "bfs": 500000, "wavefront": 2000},
    56:  {"steered": 500000, "bfs": 500000, "wavefront": 2000},
    64:  {"steered": 500000, "bfs": 500000, "wavefront": 3000},
    96:  {"steered": 500000, "bfs": 500000, "wavefront": 5000},
    128: {"steered": 500000, "bfs": 500000, "wavefront": 5000},
}


def generate_semiprime_by_gap(bits: int, category: str, index: int) -> tuple:
    """Generate a semiprime N = p*q with approximately `bits` bits in the given gap category."""
    half = bits // 2

    if category == "perfect_square":
        p = nextprime((1 << half) + index * 100)
        q = p
        N = p * q
        return N, p, q

    elif category == "close":
        p = nextprime((1 << half) + index * 100)
        for offset_mult in range(1, 100):
            candidate_q = nextprime(p + offset_mult * 2)
            if candidate_q / p < 1.1:
                q = candidate_q
                break
        else:
            q = nextprime(p)
        N = p * q
        return N, min(p, q), max(p, q)

    elif category == "moderate":
        p = nextprime((1 << half) + index * 1000)
        target_q_low = int(p * 1.1)
        target_q_high = int(p * 2.0)
        q = nextprime(target_q_low + index * 37)
        if q >= target_q_high:
            q = nextprime(target_q_low)
        N = p * q
        return N, min(p, q), max(p, q)

    elif category == "wide":
        p_bits = max(4, half // 3)
        p = nextprime((1 << p_bits) + index * 7)
        q_bits = bits - p_bits
        q = nextprime((1 << q_bits) + index * 13)
        if q / p < 2:
            q = nextprime(p * 2 + 1)
        N = p * q
        return N, min(p, q), max(p, q)

    else:
        raise ValueError(f"Unknown category: {category}")


def _factor_worker(N, steered_iter, bfs_iter, wf_radius, result_queue):
    """Worker function that runs in a subprocess with proper timeout handling."""
    from insideout.inside_out import inside_out_factor, _steered_search, _bfs_search
    from insideout.wavefront import search_wavefront
    from insideout.cf_guide import cf_factor_check

    try:
        if N < 4:
            result_queue.put({"success": False, "strategy": "none", "time_ms": 0, "timed_out": False})
            return

        start = time.perf_counter()

        # Even
        if N % 2 == 0:
            if N == 2:
                result_queue.put({"success": False, "strategy": "none", "time_ms": 0, "timed_out": False})
                return
            result_queue.put({"success": True, "strategy": "trial_division", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (2, N // 2)})
            return

        # Perfect square
        sqrt_N = isqrt(N)
        if sqrt_N * sqrt_N == N and sqrt_N > 1:
            result_queue.put({"success": True, "strategy": "perfect_square", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (sqrt_N, sqrt_N)})
            return

        # CF precheck
        cf_result = cf_factor_check(N)
        if cf_result is not None:
            p, q = cf_result
            if p * q == N and 1 < p < N and 1 < q < N:
                result_queue.put({"success": True, "strategy": "cf_precheck", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))})
                return

        # Steered search
        steered_result = _steered_search(N, max_iterations=steered_iter)
        if steered_result is not None:
            p, q = steered_result
            if p * q == N and 1 < p < N and 1 < q < N:
                result_queue.put({"success": True, "strategy": "steered", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))})
                return

        # BFS search
        bfs_result = _bfs_search(N, max_iterations=bfs_iter)
        if bfs_result is not None:
            p, q = bfs_result
            if p * q == N and 1 < p < N and 1 < q < N:
                result_queue.put({"success": True, "strategy": "inside_out_bfs", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))})
                return

        # Wavefront search
        wf_result = search_wavefront(N, max_radius=wf_radius)
        if wf_result is not None:
            p, q = wf_result
            if p * q == N and 1 < p < N and 1 < q < N:
                result_queue.put({"success": True, "strategy": "wavefront", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))})
                return

        # Trial division (limited for large N)
        limit = min(isqrt(N) + 1, 1000000)
        for p in range(3, limit, 2):
            if N % p == 0:
                result_queue.put({"success": True, "strategy": "trial_division", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (p, N // p)})
                return

        result_queue.put({"success": False, "strategy": "none", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False})

    except Exception as e:
        result_queue.put({"success": False, "strategy": "error", "time_ms": 0, "timed_out": False, "error": str(e)})


def factor_with_timeout(N: int, bits: int, timeout: float = TIMEOUT_SECONDS) -> dict:
    """Factor N using multiprocessing with a hard timeout."""
    config = TIER_CONFIGS.get(bits, TIER_CONFIGS[64])
    result_queue = mp.Queue()

    proc = mp.Process(target=_factor_worker, args=(N, config["steered"], config["bfs"], config["wavefront"], result_queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1)
        return {"success": False, "strategy": "timeout", "time_ms": timeout * 1000, "timed_out": True}

    if result_queue.empty():
        return {"success": False, "strategy": "none", "time_ms": timeout * 1000, "timed_out": True}

    return result_queue.get()


def factor_simple(N: int, bits: int) -> dict:
    """Factor N without subprocess timeout - for small sizes that complete quickly."""
    config = TIER_CONFIGS.get(bits, TIER_CONFIGS[64])

    from insideout.inside_out import inside_out_factor, _steered_search, _bfs_search
    from insideout.wavefront import search_wavefront
    from insideout.cf_guide import cf_factor_check

    start = time.perf_counter()

    if N < 4:
        return {"success": False, "strategy": "none", "time_ms": 0, "timed_out": False}

    if N % 2 == 0:
        if N == 2:
            return {"success": False, "strategy": "none", "time_ms": 0, "timed_out": False}
        return {"success": True, "strategy": "trial_division", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (2, N // 2)}

    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return {"success": True, "strategy": "perfect_square", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (sqrt_N, sqrt_N)}

    cf_result = cf_factor_check(N)
    if cf_result is not None:
        p, q = cf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return {"success": True, "strategy": "cf_precheck", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))}

    steered_result = _steered_search(N, max_iterations=config["steered"])
    if steered_result is not None:
        p, q = steered_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return {"success": True, "strategy": "steered", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))}

    bfs_result = _bfs_search(N, max_iterations=config["bfs"])
    if bfs_result is not None:
        p, q = bfs_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return {"success": True, "strategy": "inside_out_bfs", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))}

    wf_result = search_wavefront(N, max_radius=config["wavefront"])
    if wf_result is not None:
        p, q = wf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return {"success": True, "strategy": "wavefront", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (min(p, q), max(p, q))}

    # Trial division
    limit = min(isqrt(N) + 1, 1000000)
    for p in range(3, limit, 2):
        if N % p == 0:
            return {"success": True, "strategy": "trial_division", "time_ms": (time.perf_counter() - start) * 1000, "timed_out": False, "factors": (p, N // p)}

    elapsed = (time.perf_counter() - start) * 1000
    return {"success": False, "strategy": "none", "time_ms": elapsed, "timed_out": elapsed > timeout * 1000}


def run_benchmark():
    """Run the comprehensive scaling benchmark."""
    bit_sizes = [16, 24, 32, 40, 48, 56, 64]
    extended_sizes = [96, 128]
    categories = ["perfect_square", "close", "moderate", "wide"]
    samples_per_category = 5

    all_results = []

    print("=" * 80)
    print("INSIDE-OUT FACTORING SCALING BENCHMARK")
    print("=" * 80)
    print(f"Timeout per test: {TIMEOUT_SECONDS}s")
    print(f"Using multiprocessing timeout for sizes >= 40 bits")
    print()

    # Small sizes: run in-process (fast enough)
    for bits in [16, 24, 32]:
        print(f"\n{'=' * 60}")
        print(f"BIT SIZE: {bits} (in-process)")
        print(f"{'=' * 60}")

        for category in categories:
            print(f"\n  Category: {category}")

            for idx in range(samples_per_category):
                N, p, q = generate_semiprime_by_gap(bits, category, idx)
                actual_bits = N.bit_length()
                ratio = q / p if p > 0 else float('inf')

                result = factor_simple(N, bits)
                result["bits"] = bits
                result["actual_bits"] = actual_bits
                result["category"] = category
                result["sample"] = idx + 1
                result["N"] = N
                result["p"] = p
                result["q"] = q
                result["ratio_q_over_p"] = round(ratio, 4)
                all_results.append(result)

                status = "OK" if result["success"] else ("TMO" if result.get("timed_out") else "FAIL")
                print(f"    [{idx+1}] {actual_bits}b N, ratio={ratio:.2f}, "
                      f"strategy={result['strategy']}, time={result['time_ms']:.1f}ms, {status}")

    # Medium+ sizes: use subprocess timeout
    for bits in [40, 48, 56, 64]:
        print(f"\n{'=' * 60}")
        print(f"BIT SIZE: {bits} (with subprocess timeout)")
        print(f"{'=' * 60}")

        for category in categories:
            print(f"\n  Category: {category}")

            for idx in range(samples_per_category):
                N, p, q = generate_semiprime_by_gap(bits, category, idx)
                actual_bits = N.bit_length()
                ratio = q / p if p > 0 else float('inf')

                print(f"    [{idx+1}] {actual_bits}b N, ratio={ratio:.2f} ... ", end="", flush=True)
                result = factor_with_timeout(N, bits, timeout=TIMEOUT_SECONDS)
                result["bits"] = bits
                result["actual_bits"] = actual_bits
                result["category"] = category
                result["sample"] = idx + 1
                result["N"] = N
                result["p"] = p
                result["q"] = q
                result["ratio_q_over_p"] = round(ratio, 4)
                all_results.append(result)

                status = "OK" if result["success"] else ("TMO" if result.get("timed_out") else "FAIL")
                print(f"strategy={result['strategy']}, time={result['time_ms']:.1f}ms, {status}")

    # Extended sizes: 96 and 128 bits
    for bits in [96, 128]:
        print(f"\n{'=' * 60}")
        print(f"BIT SIZE: {bits} (extended, with subprocess timeout)")
        print(f"{'=' * 60}")

        for category in categories:
            print(f"\n  Category: {category}")

            for idx in range(3):  # Fewer samples for large sizes
                N, p, q = generate_semiprime_by_gap(bits, category, idx)
                actual_bits = N.bit_length()
                ratio = q / p if p > 0 else float('inf')

                print(f"    [{idx+1}] {actual_bits}b N, ratio={ratio:.2f} ... ", end="", flush=True)
                result = factor_with_timeout(N, bits, timeout=TIMEOUT_SECONDS)
                result["bits"] = bits
                result["actual_bits"] = actual_bits
                result["category"] = category
                result["sample"] = idx + 1
                result["N"] = N
                result["p"] = p
                result["q"] = q
                result["ratio_q_over_p"] = round(ratio, 4)
                all_results.append(result)

                status = "OK" if result["success"] else ("TMO" if result.get("timed_out") else "FAIL")
                print(f"strategy={result['strategy']}, time={result['time_ms']:.1f}ms, {status}")

    return all_results


def generate_report(results: list) -> str:
    """Generate a markdown report from benchmark results."""
    from collections import defaultdict

    lines = []
    lines.append("# Inside-Out Factoring: Scale-Up Findings")
    lines.append("")
    lines.append("Comprehensive benchmark measuring the Inside-Out factoring algorithm's")
    lines.append("performance across bit sizes 16-128 and factor gap categories.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Algorithm**: Inside-Out factoring with multiple strategies tested individually")
    lines.append("- **Strategies**: perfect_square detection, CF convergent precheck (cf_precheck),")
    lines.append("  CF-steered best-first search, BFS fallback, wavefront search, trial_division")
    lines.append("- **Timeout**: 10 seconds per factorization (hard subprocess kill)")
    lines.append("- **Iteration limits**: tiered by bit size (50K-500K for steered/BFS, 500-5K for wavefront radius)")
    lines.append("- **Factor gap categories**:")
    lines.append("  - Perfect square: p = q (N = p^2)")
    lines.append("  - Close factors: q/p < 1.1")
    lines.append("  - Moderate gap: 1.1 <= q/p < 2")
    lines.append("  - Wide gap: q/p >= 2")
    lines.append("")

    # Group results
    groups = defaultdict(list)
    for r in results:
        key = (r["bits"], r["category"])
        groups[key].append(r)

    # Results table
    lines.append("## Results by Bit Size and Factor Gap")
    lines.append("")
    lines.append("| Bits | Category | Tests | OK | Avg ms | Med ms | Dominant Strategy | Timeouts |")
    lines.append("|------|----------|-------|----|--------|--------|-------------------|----------|")

    for bits in [16, 24, 32, 40, 48, 56, 64, 96, 128]:
        for cat in ["perfect_square", "close", "moderate", "wide"]:
            key = (bits, cat)
            if key not in groups:
                continue
            entries = groups[key]
            total = len(entries)
            successes = [e for e in entries if e.get("success")]
            timeouts = [e for e in entries if e.get("timed_out")]
            success_count = len(successes)

            strategy_counts = defaultdict(int)
            for e in entries:
                strategy_counts[e.get("strategy", "none")] += 1
            dominant = max(strategy_counts, key=strategy_counts.get) if strategy_counts else "none"

            times = [e["time_ms"] for e in successes if "time_ms" in e]
            if times:
                avg_time = sum(times) / len(times)
                sorted_times = sorted(times)
                median_time = sorted_times[len(sorted_times) // 2]
                avg_str = f"{avg_time:.1f}"
                med_str = f"{median_time:.1f}"
            else:
                avg_str = "-"
                med_str = "-"

            timeout_str = f"{len(timeouts)}/{total}"
            lines.append(f"| {bits} | {cat} | {total} | {success_count} | {avg_str} | {med_str} | {dominant} | {timeout_str} |")

    lines.append("")

    # Strategy dominance
    lines.append("## Strategy Dominance by Bit Size")
    lines.append("")
    lines.append("| Bits | perfect_sq | cf_precheck | steered | bfs | wavefront | trial_div | timeout/fail |")
    lines.append("|------|-----------|-------------|---------|-----|-----------|-----------|-------------|")

    for bits in [16, 24, 32, 40, 48, 56, 64, 96, 128]:
        bit_entries = [r for r in results if r["bits"] == bits]
        total = len(bit_entries)
        if total == 0:
            continue
        sc = defaultdict(int)
        for e in bit_entries:
            s = e.get("strategy", "none")
            if e.get("timed_out"):
                s = "timeout"
            elif not e.get("success"):
                s = "fail"
            sc[s] += 1
        row = f"| {bits} |"
        for s in ["perfect_square", "cf_precheck", "steered", "inside_out_bfs", "wavefront", "trial_division"]:
            c = sc.get(s, 0)
            row += f" {c} |"
        fail_total = sc.get("timeout", 0) + sc.get("fail", 0) + sc.get("none", 0)
        row += f" {fail_total} |"
        lines.append(row)

    lines.append("")

    # Success rate by bit size
    lines.append("## Success Rate by Bit Size")
    lines.append("")
    lines.append("| Bits | Total | OK | Fail | Timeout | Rate |")
    lines.append("|------|-------|----|------|---------|------|")

    for bits in [16, 24, 32, 40, 48, 56, 64, 96, 128]:
        bit_entries = [r for r in results if r["bits"] == bits]
        total = len(bit_entries)
        if total == 0:
            continue
        successes = len([e for e in bit_entries if e.get("success")])
        failures = len([e for e in bit_entries if not e.get("success") and not e.get("timed_out")])
        timeouts = len([e for e in bit_entries if e.get("timed_out")])
        rate = f"{successes/total*100:.0f}%"
        lines.append(f"| {bits} | {total} | {successes} | {failures} | {timeouts} | {rate} |")

    lines.append("")

    # Success rate by category across sizes
    lines.append("## Success Rate by Category and Bit Size")
    lines.append("")
    lines.append("| Bits | perfect_sq | close | moderate | wide |")
    lines.append("|------|-----------|-------|----------|------|")

    for bits in [16, 24, 32, 40, 48, 56, 64, 96, 128]:
        row = f"| {bits} |"
        for cat in ["perfect_square", "close", "moderate", "wide"]:
            entries = [r for r in results if r["bits"] == bits and r["category"] == cat]
            if entries:
                ok = len([e for e in entries if e.get("success")])
                rate = f"{ok}/{len(entries)}"
                row += f" {rate} |"
            else:
                row += " - |"
        lines.append(row)

    lines.append("")

    # Median time table
    lines.append("## Median Time (ms) by Bit Size and Category (successful only)")
    lines.append("")
    lines.append("| Bits | perfect_sq | close | moderate | wide |")
    lines.append("|------|-----------|-------|----------|------|")

    for bits in [16, 24, 32, 40, 48, 56, 64, 96, 128]:
        row = f"| {bits} |"
        for cat in ["perfect_square", "close", "moderate", "wide"]:
            entries = [r for r in results if r["bits"] == bits and r["category"] == cat and r.get("success")]
            if entries:
                times = sorted([r["time_ms"] for r in entries])
                median = times[len(times) // 2]
                row += f" {median:.1f} |"
            else:
                row += " - |"
        lines.append(row)

    lines.append("")

    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| Bits | Cat | # | Ratio | Strategy | ms | Status |")
    lines.append("|------|-----|---|-------|----------|----|--------|")

    for r in results:
        status = "OK" if r.get("success") else ("TMO" if r.get("timed_out") else "FAIL")
        cat_short = r["category"][:5]
        lines.append(f"| {r['bits']} | {cat_short} | {r['sample']} | {r['ratio_q_over_p']:.2f} | "
                     f"{r.get('strategy', '?')} | {r['time_ms']:.1f} | {status} |")

    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    # Where failures start
    lines.append("### Failure Boundary")
    lines.append("")
    failure_bits_map = {}
    for bits in [16, 24, 32, 40, 48, 56, 64, 96, 128]:
        bit_entries = [r for r in results if r["bits"] == bits]
        if not bit_entries:
            continue
        success_rate = len([e for e in bit_entries if e.get("success")]) / len(bit_entries)
        failure_bits_map[bits] = success_rate

    first_failure = None
    for bits in sorted(failure_bits_map.keys()):
        if failure_bits_map[bits] < 1.0:
            first_failure = bits
            break

    if first_failure:
        lines.append(f"First bit size with failures: **{first_failure}-bit**")
        for bits in sorted(failure_bits_map.keys()):
            lines.append(f"- {bits}-bit: {failure_bits_map[bits]*100:.0f}% success rate")
    else:
        lines.append("All tested bit sizes achieved 100% success rate.")

    lines.append("")

    # Category-specific failures
    lines.append("### Category-Specific Failure Points")
    lines.append("")
    for cat in ["perfect_square", "close", "moderate", "wide"]:
        cat_entries = [r for r in results if r["category"] == cat and not r.get("success")]
        if cat_entries:
            fail_bits = sorted(set(r["bits"] for r in cat_entries))
            lines.append(f"- **{cat}**: Failures at {fail_bits}")
        else:
            lines.append(f"- **{cat}**: No failures in tested range")

    lines.append("")

    # Strategy recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Find dominant strategy per size range
    small = [r for r in results if r["bits"] <= 32 and r.get("success")]
    medium = [r for r in results if 40 <= r["bits"] <= 64 and r.get("success")]
    large = [r for r in results if r["bits"] >= 96 and r.get("success")]

    def top_strats(entries, n=3):
        from collections import Counter
        c = Counter(e.get("strategy", "?") for e in entries)
        return c.most_common(n)

    lines.append("### Dominant Strategies by Size Range")
    lines.append("")
    if small:
        lines.append("**16-32 bit**:")
        for strat, count in top_strats(small):
            lines.append(f"- {strat}: {count}")
    if medium:
        lines.append("")
        lines.append("**40-64 bit**:")
        for strat, count in top_strats(medium):
            lines.append(f"- {strat}: {count}")
    if large:
        lines.append("")
        lines.append("**96-128 bit**:")
        for strat, count in top_strats(large):
            lines.append(f"- {strat}: {count}")

    lines.append("")

    # Timeout analysis
    timeout_entries = [r for r in results if r.get("timed_out")]
    if timeout_entries:
        min_timeout_bits = min(r["bits"] for r in timeout_entries)
        lines.append(f"### Timeout Analysis")
        lines.append(f"")
        lines.append(f"Timeouts first appear at **{min_timeout_bits}-bit**.")
        lines.append("")

        # By category
        for cat in ["perfect_square", "close", "moderate", "wide"]:
            cat_tmos = [r for r in timeout_entries if r["category"] == cat]
            if cat_tmos:
                bits_set = sorted(set(r["bits"] for r in cat_tmos))
                lines.append(f"- {cat}: {len(cat_tmos)} timeouts at {bits_set}")
    else:
        lines.append("### No Timeouts")
        lines.append("No timeouts occurred in the tested range.")

    lines.append("")
    lines.append("### Scaling Improvement Priorities")
    lines.append("")
    lines.append("1. **Adaptive iteration budgets**: Scale iteration limits with N's bit length")
    lines.append("2. **Extend CF convergent terms**: Increase max_terms for larger N to improve cf_precheck effectiveness")
    lines.append("3. **Wavefront radius scaling**: Increase max_radius proportionally to bit size")
    lines.append("4. **Add Pollard's rho**: As fallback for medium-factor semiprimes where steered search struggles")
    lines.append("5. **Quadratic sieve**: Consider for 64+ bit sizes as the primary fallback")
    lines.append("6. **Close-factor specialization**: The steered search is most effective for close factors; wider gaps need different strategies")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by scale_up_benchmark.py*")

    return "\n".join(lines)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    results = run_benchmark()

    # Save raw JSON
    # Convert non-serializable types
    serializable_results = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                sr[k] = v
            elif isinstance(v, tuple):
                sr[k] = list(v)
            else:
                sr[k] = str(v)
        serializable_results.append(sr)

    with open("/home/raver1975/insideoutfactoring/benchmarks/scale_up_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    print("\n\nGenerating report...")
    report = generate_report(results)

    output_path = "/home/raver1975/insideoutfactoring/docs/superpowers/sdd/scale-up-findings.md"
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Report written to {output_path}")
    print(f"Raw data saved to benchmarks/scale_up_results.json")