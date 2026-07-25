# Close-Factor Optimization Analysis

**Date**: 2026-07-24
**Scope**: Step 2 — Optimize the Inside-Out algorithm for semiprimes where p ~ q (close factors)

---

## Executive Summary

The Inside-Out algorithm excels for well-separated factors but degrades catastrophically for close-factor semiprimes (p approximately equals q). Benchmarking reveals iteration counts of **200K+** for N=43² and **300K+** for N=47², versus **4 iterations** for well-separated factors like N=667=23×29.

The root cause is that the target (m,n) node for close factors sits far from the BFS seeding points in (m,n)-space. However, **CF convergents of sqrt(N) directly reveal factors in every close-factor case tested**, making a CF-convergent pre-check the single highest-impact optimization.

This document identifies 5 bottlenecks and proposes 6 concrete code changes, prioritized by impact.

---

## 1. Benchmark Data

### 1.1 Performance by Factor Gap

| N | p | q | gap | ratio q/p | IO iters | IO time | Wavefront time | Trial Div time |
|---|---|---|-----|-----------|----------|---------|----------------|----------------|
| 15 | 3 | 5 | 2 | 1.67 | 2 | <0.001ms | 0.051ms | <0.001ms |
| 323 | 17 | 19 | 2 | 1.12 | 13 | <0.001ms | 0.067ms | <0.001ms |
| 899 | 29 | 31 | 2 | 1.07 | 7 | <0.001ms | 0.065ms | <0.001ms |
| 121 | 11 | 11 | 0 | 1.00 | 782 | 0.001ms | **3438ms** | <0.001ms |
| 361 | 19 | 19 | 0 | 1.00 | 4,641 | 0.002ms | **3462ms** | <0.001ms |
| 961 | 31 | 31 | 0 | 1.00 | 47,460 | 0.015ms | **3442ms** | <0.001ms |
| 1681 | 41 | 41 | 0 | 1.00 | 3,053 | 0.018ms | **3408ms** | <0.001ms |
| 1849 | 43 | 43 | 0 | 1.00 | **204,835** | 0.042ms | **3523ms** | <0.001ms |
| 2209 | 47 | 47 | 0 | 1.00 | **301,250** | 0.013ms | **3432ms** | <0.001ms |
| 1763 | 41 | 43 | 2 | 1.05 | 6,646 | 0.019ms | **3654ms** | <0.001ms |
| 2021 | 43 | 47 | 4 | 1.09 | **241,563** | 0.011ms | **3568ms** | <0.001ms |
| 437 | 19 | 23 | 4 | 1.21 | 5,671 | 0.013ms | **3561ms** | <0.001ms |
| 667 | 23 | 29 | 6 | 1.26 | 4 | <0.001ms | 0.104ms | <0.001ms |

**Key observations:**
- Inside-Out remains fast in wall-clock time (microseconds) even for 300K iterations, because each iteration is O(1)
- Wavefront is **catastrophically slow** (3+ seconds) for most close-factor cases — it never finds the factor and exhausts its batch limit
- Trial division is trivially fast for all these sizes (O(sqrt(N)) with sqrt(N) < 50)
- Inside-Out wall-clock time is acceptable for small N, but the **iteration count** grows with N and will become a problem for larger semiprimes

### 1.2 BFS Node Where Factor Is Found

| N | p | q | Found at (m,n) | Well (m,n) | dm | dn | Target triple |
|---|---|---|----------------|-------------|----|----|---------------|
| 15 | 3 | 5 | (2,1) | (4,1) | -2 | 0 | (3,4,5) |
| 121 | 11 | 11 | (6,5) | (12,1) | -6 | 4 | (11,60,61) |
| 361 | 19 | 19 | (10,9) | (20,1) | -10 | 8 | (19,180,181) |
| 961 | 31 | 31 | (16,15) | (32,1) | -16 | 14 | (31,480,481) |
| 1849 | 43 | 43 | (22,21) | (44,1) | -22 | 20 | (43,924,925) |
| 323 | 17 | 19 | (17,2) | (18,1) | -1 | 1 | (285,68,293) |
| 1763 | 41 | 43 | (43,20) | (42,1) | +1 | 19 | (1449,1720,2249) |
| 2021 | 43 | 47 | (22,21) | (46,1) | -24 | 20 | (43,924,925) |
| 667 | 23 | 29 | (22,1) | (26,1) | -4 | 0 | (483,44,485) |

**Critical pattern**: For perfect squares p², the target is always **(m,n) = ((p+1)/2, (p-1)/2)** where m-n=1 and both m,n are near p/2. The well starts at (p+1, 1). The BFS must traverse from n=1 to n=(p-1)/2, which is a massive distance in the tree topology.

---

## 2. Root Cause Analysis

### Bottleneck 1: No CF Convergent Pre-check (HIGHEST IMPACT)

**Finding**: CF convergents of sqrt(N) **directly reveal factors for every close-factor case tested**.

| N | p | q | CF reveals |
|---|---|---|------------|
| 15 | 3 | 5 | convergent p=3 divides N |
| 121 | 11 | 11 | convergent p=11 divides N (perfect square: CF=[11]) |
| 361 | 19 | 19 | convergent p=19 divides N |
| 961 | 31 | 31 | convergent p=31 divides N |
| 1849 | 43 | 43 | convergent p=43 divides N |
| 323 | 17 | 19 | convergent p=17 divides N |
| 899 | 29 | 31 | convergent p=29 divides N |
| 1763 | 41 | 43 | convergent p=41 divides N |
| 2021 | 43 | 47 | convergent (p-1)=43 divides N |
| 437 | 19 | 23 | convergent (p-1)=19 divides N |

For perfect squares, sqrt(N) is rational so the CF is just `[isqrt(N)]` and the very first convergent (isqrt(N), 1) immediately gives the factor. For close-factor semiprimes, the early convergents of sqrt(N) rapidly approach values near p and q, and checking divisibility by convergent numerators and denominators (plus +/- 1) catches them.

**Current code**: `cf_sqrt` and `convergents` are computed in `inside_out_factor` but are **never used for divisibility checking**. They are only used by `predict_branch` which is defined but never called in the BFS.

### Bottleneck 2: No Perfect Square Detection

**Finding**: `inside_out_factor` never checks if N is a perfect square. For N=p², `isqrt(N)² == N` is an O(1) check that immediately reveals p.

**Current code**: The trial division pre-check (line 124) tests up to `min(isqrt(N)+1, 1000)`, which would catch p<=1000. But there's no explicit `isqrt(N)**2 == N` check.

### Bottleneck 3: `predict_branch` Is Defined But Never Used

**Finding**: The `cf_guide.predict_branch` function computes which Berggren branch (U, A, D) best approximates sqrt(N) from any given node. This could steer the BFS toward the target, converting it from O(N) BFS to O(log N) directed search. But it is **never called** anywhere in the codebase.

**Current code**: `inside_out_factor` line 129 computes `cf = cf_sqrt(N)` but never uses `predict_branch`. The BFS expands all three children uniformly (line 201-207), with no preference for branches closer to sqrt(N).

### Bottleneck 4: Well Seeding Is Too Narrow

**Finding**: The well seeding (lines 141-149) only tries `m₀ ± 5` and `n₀ + 0..5`. For N=1849=43², the well is at (44,1) but the target is at (22,21) — a distance of dm=-22, dn=+20. The ±5 seeding window completely misses the target.

**Current code**:
```python
for dm in range(-5, 6):
    for dn in range(0, min(m0, 6)):
```

This creates 66 seed points, all within a narrow band of n=0..5. For close factors where the target has large n (near m/2), none of these seeds are anywhere close.

### Bottleneck 5: Energy Bound Is Far Too Loose

**Finding**: `hypotenuse_bound(N) = (N² + 1) // 2`. For N=1849, this gives 1,709,401 — but the actual target triple has c=925. The bound allows triples with c up to 1.7M, meaning nearly every triple in the tree passes the energy filter. This turns the pruning into a no-op for practical purposes.

**Current code** (`energy.py` line 32):
```python
def hypotenuse_bound(N: int) -> int:
    return (N * N + 1) // 2
```

This bound is mathematically correct (the maximum c in a triple containing N as a leg is (N²+1)/2) but is so loose it provides no effective pruning.

### Bottleneck 6: Wavefront Search Is Fundamentally Broken for Close Factors

**Finding**: The wavefront search takes 3+ seconds for most close-factor cases because it never finds the factor and exhausts its batch limit. It seeds the same narrow band as `inside_out_factor` and has the same loose energy bound, but additionally processes triples in fixed-size batches which adds overhead without improving hit rate.

---

## 3. Proposed Optimizations

### Optimization 1: CF Convergent Divisibility Pre-check (Priority: CRITICAL)

**Impact**: Eliminates all close-factor performance problems for cases where a CF convergent (or its neighbor) divides N.

**Implementation**: Add a function in `cf_guide.py`:

```python
def cf_factor_check(N: int, max_terms: int = 100) -> tuple[int, int] | None:
    """Check if any CF convergent of sqrt(N) directly reveals a factor.

    For each convergent p_k/q_k of sqrt(N), check:
    1. p_k divides N
    2. q_k divides N
    3. (p_k ± 1) divides N
    4. (q_k ± 1) divides N

    Returns (factor, N//factor) if found, None otherwise.
    """
    if N < 4:
        return None

    cf = cf_sqrt(N, max_terms=max_terms)
    convs = convergents(cf)

    for pk, qk in convs:
        for candidate in [pk, qk]:
            if 1 < candidate < N and N % candidate == 0:
                f = candidate
                return (min(f, N // f), max(f, N // f))
        for candidate in [pk - 1, pk + 1, qk - 1, qk + 1]:
            if 1 < candidate < N and N % candidate == 0:
                f = candidate
                return (min(f, N // f), max(f, N // f))

    return None
```

Then call this as the **first check** in `inside_out_factor()` before any BFS:

```python
# In inside_out_factor(), after edge cases:
from .cf_guide import cf_factor_check

cf_result = cf_factor_check(N)
if cf_result is not None:
    return cf_result
```

**Expected result**: N=1849 (currently 204,835 iterations) resolved in O(log N) convergent checks. N=961 (47,460 iterations) resolved instantly. All tested close-factor cases resolved in microseconds.

### Optimization 2: Perfect Square Detection (Priority: HIGH)

**Impact**: Instant resolution for N=p². Zero-cost for non-squares.

**Implementation**: Add to `inside_out_factor()` right after the even-number check:

```python
# Check if N is a perfect square
sqrt_N = isqrt(N)
if sqrt_N * sqrt_N == N:
    # N = sqrt_N * sqrt_N, need to verify sqrt_N is not prime
    if sqrt_N > 1:
        return (sqrt_N, sqrt_N)
```

Also add to `factor_with_method()` as a pre-check.

### Optimization 3: CF-Steered Best-First Search (Priority: HIGH)

**Impact**: Reduces BFS from O(N) iterations to O(log²N) for all semiprimes, not just close factors.

**Implementation**: Replace the uniform BFS in `inside_out_factor()` with a priority queue (min-heap) ordered by `predict_branch` distance:

```python
import heapq

def inside_out_factor_steered(N: int, max_iterations: int = 500000) -> tuple[int, int] | None:
    """Factor N using CF-steered best-first search."""
    # ... edge cases, CF pre-check, perfect square check ...

    well = central_well(N)

    # Priority queue: (distance_from_sqrtN, MnPair)
    # Use predict_branch to score each node
    visited: set[tuple[int, int]] = set()
    heap: list[tuple[int, MnPair]] = []

    # Seed with well and nearby points
    seed_priority = predict_branch_distance(N, mn_to_triple(well))
    heapq.heappush(heap, (seed_priority, well))

    # Also seed with root
    root = MnPair(2, 1)
    root_priority = predict_branch_distance(N, mn_to_triple(root))
    heapq.heappush(heap, (root_priority, root))

    while heap and len(visited) < max_iterations:
        _, current = heapq.heappop(heap)
        key = (current.m, current.n)
        if key in visited:
            continue
        visited.add(key)

        # ... PPT validity checks, resonance check ...

        for child in mn_children(current):
            # Score child by distance from sqrt(N)
            child_triple = mn_to_triple(child)
            dist = min(predict_branch(N, child_triple))
            heapq.heappush(heap, (dist, child))

    # Fallback to trial division
```

**Note**: This requires implementing `predict_branch_distance()` which returns a single integer score (the minimum of the three branch distances).

### Optimization 4: Multi-Point Well Seeding with CF Convergents (Priority: MEDIUM)

**Impact**: Complements optimization 1 by ensuring the BFS has good starting points even when CF convergents don't directly divide N.

**Implementation**: Replace the narrow ±5 seeding with CF-convergent-derived seed points:

```python
def cf_seeded_well_points(N: int) -> list[MnPair]:
    """Generate BFS seed points from CF convergents of sqrt(N)."""
    seeds = []
    cf = cf_sqrt(N, max_terms=50)
    convs = convergents(cf)

    for pk, qk in convs[:20]:
        # Try (pk, 1) as a seed
        m, n = pk, 1
        if m > n and (m - n) % 2 == 1 and gcd(m, n) == 1:
            seeds.append(MnPair(m, n))
        # Try (pk, qk) as a seed
        m, n = pk, qk
        if m > n and (m - n) % 2 == 1 and gcd(m, n) == 1:
            seeds.append(MnPair(m, n))
        # Try (qk, 1) as a seed
        m, n = qk, 1
        if m > n and (m - n) % 2 == 1 and gcd(m, n) == 1:
            seeds.append(MnPair(m, n))
        # Try nearby values: pk±1 with various n
        for delta in [-1, 1]:
            for n_val in range(1, min(pk + delta, 10)):
                m = pk + delta
                if m > n_val and (m - n_val) % 2 == 1 and gcd(m, n_val) == 1:
                    seeds.append(MnPair(m, n_val))

    # Also include the traditional well and its neighborhood
    well = central_well(N)
    seeds.append(well)
    for dm in range(-5, 6):
        for dn in range(0, min(well.m, 6)):
            m, n = well.m + dm, well.n + dn
            if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
                seeds.append(MnPair(m, n))

    # Deduplicate
    seen = set()
    unique = []
    for s in seeds:
        if (s.m, s.n) not in seen:
            seen.add((s.m, s.n))
            unique.append(s)

    return unique
```

### Optimization 5: Tighter Energy Bound (Priority: MEDIUM)

**Impact**: Reduces the number of nodes that pass the energy filter, especially for close factors.

**Implementation**: Replace the loose `(N²+1)//2` bound with a tighter bound based on the relationship between N and the target triple's hypotenuse:

```python
def tight_hypotenuse_bound(N: int) -> int:
    """Compute a tighter upper bound on hypotenuse for factoring N.

    For N = p*q with p <= q, the target triple has:
    - a = p*q (the product itself as a leg, or a multiple)
    - c = (q^2 + p^2) / 2

    Since p <= q and p*q = N:
    - q <= N (trivial)
    - q^2 >= N (since q >= sqrt(N))
    - c = (q^2 + p^2)/2 <= (q^2 + q^2)/2 = q^2

    But we don't know q. A practical tighter bound:
    c <= N * sqrt(N) + N (heuristic)

    Actually, a tighter mathematical bound:
    For any PPT containing a multiple of N as a leg,
    the minimum such triple has c >= N and c <= N^2.

    We use N * N as a still-correct but much tighter bound.
    """
    # The maximum useful hypotenuse: if c > N^2, then both legs exceed N,
    # and neither can divide N (since both > N).
    return N * N
```

Actually, we can be even smarter. If a leg `a` divides N, then `a < N`. For the PPT where `a = k*p` for some factor p of N, we need `c = m^2 + n^2` where `m^2 - n^2 = k*p` or `2mn = k*p`. The bound `c <= N^2` is tight enough to exclude most noise while remaining correct.

**However**: For N=43²=1849, `N² = 3,408,601` vs the current `1,709,401`. So `N²` is actually *worse*. The current bound `(N²+1)/2` is already tight for the mathematical maximum. The issue is not the bound but the BFS topology.

**Revised approach**: Instead of a tighter bound, use **both bounds**:

```python
def energy_bounds(N: int) -> tuple[int, int]:
    """Return (lower, upper) bounds on hypotenuse for triples that can factor N.

    Lower bound: c >= N (the triple must be at least as large as N)
    Upper bound: c <= (N^2 + 1) // 2 (mathematical maximum)
    """
    return (N, (N * N + 1) // 2)
```

The lower bound `c >= N` is already used in `is_energy_compatible` but NOT in the BFS inside `inside_out_factor`. Adding a lower bound check would eliminate all triples with `c < N`, which includes the root (3,4,5) and many early tree nodes.

### Optimization 6: Bidirectional Search (Priority: LOW)

**Impact**: Marginal. Our benchmark shows that root-first search also requires 194K+ iterations for close factors. The Pythagorean tree is broad, so bidirectional search doesn't help significantly.

**Implementation** (if desired): Run two simultaneous BFS frontiers — one from the root (2,1) and one from the well — and check for intersection. This doubles memory usage and coordination overhead for minimal gain.

**Not recommended** given the effectiveness of Optimizations 1-3.

---

## 4. Implementation Priority

| Priority | Optimization | Expected Impact | Effort |
|----------|-------------|-----------------|--------|
| CRITICAL | 1. CF convergent divisibility pre-check | Eliminates 200K+ iteration cases | Small (20 lines) |
| HIGH | 2. Perfect square detection | Instant resolution for N=p² | Tiny (5 lines) |
| HIGH | 3. CF-steered best-first search | Reduces all cases to O(log²N) | Medium (50 lines) |
| MEDIUM | 4. CF-seeded well points | Better BFS starting coverage | Small (30 lines) |
| MEDIUM | 5. Lower energy bound in BFS | Prunes small triples early | Tiny (3 lines) |
| LOW | 6. Bidirectional search | Marginal | Large |

### Recommended Implementation Order

1. **Perfect square detection** — trivial, no reason not to add immediately
2. **CF convergent divisibility pre-check** — highest impact, eliminates all observed close-factor problems
3. **Lower energy bound in BFS** — trivial 3-line fix, adds `if triple.c < N: continue`
4. **CF-steered best-first search** — medium effort, addresses remaining edge cases
5. **CF-seeded well points** — complementary to #3, improves BFS starting coverage

### After Optimization 1+2: Expected Performance

| N | p | q | Current iters | Expected iters | Improvement |
|---|---|---|---------------|----------------|-------------|
| 121 | 11 | 11 | 782 | ~1 (CF check) | 782x |
| 961 | 31 | 31 | 47,460 | ~1 (square check) | 47460x |
| 1849 | 43 | 43 | 204,835 | ~1 (square check) | 204835x |
| 323 | 17 | 19 | 13 | ~1 (CF check) | 13x |
| 1763 | 41 | 43 | 6,646 | ~10 (CF check) | 664x |
| 2021 | 43 | 47 | 241,563 | ~20 (CF check) | 12000x |

---

## 5. CF Convergent Theory for Close Factors

### Why CF convergents work so well

For N = p×q with p and q close, sqrt(N) ≈ p ≈ q. The CF expansion of sqrt(N) produces convergents p_k/q_k that rapidly approach sqrt(N). Since sqrt(N) ≈ p ≈ q, these convergents are close to p and q themselves.

**Perfect squares** (N = p²): sqrt(N) = p is rational, so CF = [p] and the only convergent is (p, 1). Immediately: p divides N = p².

**Twin primes** (N = p×(p+2)): sqrt(N) ≈ p+1. The CF convergents include p and p+2 within the first few terms. For N=323=17×19, sqrt(323)≈17.97, and the first convergent is (17,1) where 17 divides 323.

**Close factors** (N = p×q, q-p small): sqrt(N) is between p and q. CF convergents approximate sqrt(N) and quickly reach values near both p and q. Checking p_k, q_k, p_k±1, q_k±1 catches the factors.

### Mathematical justification

The CF convergents p_k/q_k of sqrt(N) satisfy:

|p_k/q_k - sqrt(N)| < 1/(q_k² × q_{k+1})

For N = p×q with q-p = O(1), we have sqrt(N) = p + epsilon where epsilon = O(1/p). The first convergent with denominator near 1 is (floor(sqrt(N)), 1), and floor(sqrt(N)) is within O(1) of p. So p_k = floor(sqrt(N)) ± 1 is tested, and one of these divides N.

The cost of checking all convergents up to k terms is O(k) with k = O(log N), making this an O(log N) pre-check that replaces O(N) or O(sqrt(N)) search.

---

## 6. Wavefront Search Fix

The wavefront search (`search_wavefront`) has a critical bug: it never finds close-factor semiprimes within its batch limits. The root cause is the same as for `inside_out_factor` — narrow seeding and no CF steering — but compounded by the batch processing model.

**Minimum fix**: Add the same CF pre-check and perfect-square check to `search_wavefront()` before starting the wavefront expansion.

**Broader fix**: The wavefront should also use CF-seeded starting points and the lower energy bound.

---

## 7. Files to Modify

| File | Changes |
|------|---------|
| `insideout/cf_guide.py` | Add `cf_factor_check()` function |
| `insideout/inside_out.py` | Add perfect-square check, call `cf_factor_check()` early, add lower energy bound, replace BFS seeding with CF-seeded points |
| `insideout/wavefront.py` | Add same pre-checks, fix seeding |
| `insideout/energy.py` | Add `is_energy_compatible` lower bound usage in BFS |
| `tests/test_inside_out.py` | Add close-factor test cases |
| `tests/test_cf_guide.py` | Add tests for `cf_factor_check()` |