# Projective Collision Factoring — Two-Phase Improvement Design

## Status
Approved for implementation 2026-07-26.

## Goal
Improve the `projective_collision.py` worker through chart compression,
distinguished walks, batch GCD, and fingerprints (Phase 1), then add an
orbit-to-smooth-relation NFS lane as a separate module (Phase 2).

## Phase 1: Chart-Compressed Distinguished-Walk Worker

### What Changes
Generic complexity stays at `N^(1/4)` — this phase improves constants, memory,
and robustness.

### Conic Chart Compression
The projective Pythagorean conic `a² + b² = c²` is rational. Instead of
evaluating all 3 minors between triples (a,b,c) and (u,v,w):

    av − bu,  aw − cu,  bw − cv

use one affine chart:

- **Chart 1:** slope s = a / (c + b)
- **Chart 2:** slope s = a / (c − b)  (exceptional locus fallback)

Algorithm for one step:
1. Compute s = a · inv(c + b) mod N
2. If inv fails → gcd(c + b, N) is a factor (certificate)
3. Collision between two points: a₂(c₁+b₁) − a₁(c₂+b₂) mod N → gcd with N

Single determinant replaces three minors. Exceptional points (where c+b ∉ (Z/NZ)*)
automatically yield factors — no extra work.

### Distinguished-Point Walks
- Replace breadth-first orbit storage with salted pseudorandom branch walks
- Store only distinguished endpoints (low-bit-prefix collision)
- On cross-walk distinguished collision: replay both walks to locate merge point
- Distinguishable by walk salt and step count → unique replay trace
- Memory drops by distinguished density factor (~1/256 with 8 bits)

### Batch GCD with Product Trees
- Accumulate 100–1000 chart determinants before a GCD batch
- Use recursive batch GCD (Bernstein-style): shared products, exact isolation
- Count the `gcd = N` (failed inversion) path separately in benchmarks

### Fingerprint Pre-Filtering
- Auxiliary small primes create fingerprints of orbit states
- Bucket likely matches before full big-integer determinant work
- False positives still go through gcd(det, N) — never certifies a factor alone

### Integration
- `projective_chart_factor(N)` added to `factor.py` as `"projective_chart"`
- Placed after `lucas_ppt` in the fallback chain (fast sub-ms → slower ms-scale)
- Replaces the old 3-minor worker as the primary projective method

### Files
- `insideout/projective_collision.py` — rewrite with chart compression + distinguished walks
- `insideout/factor.py` — add `projective_chart_factor` import and fallback entry
- `tests/test_new_methods.py` — unit + integration tests
- `benchmarks/benchmark.py` — add comparison column

### Tests
1. **Unit:** chart normalization equivalence (chart determinant = 0 iff all 3 minors = 0 mod p)
2. **Unit:** failed inversion → gcd(c+b, N) finds a factor
3. **Unit:** distinguished predicate correctness
4. **Integration:** factor 15–20 known semiprimes (balanced and unbalanced)
5. **Comparison:** same corpus, same budget — chart worker vs original 3-minor worker vs Brent rho

---

## Phase 2: Orbit-to-Smooth-Relation NFS Lane

### What Changes
Runs as a **separate parallel experiment** (not a replacement for the collision
worker). Its goal is generating smooth relations from projective orbits for
NFS-style linear algebra — the only lane currently aimed at subexponential
asymptotic improvement.

### Orbit-to-Integer Norm Mapping
- Fix a seed vector v (e.g., the root triple (3,4,5))
- For each bounded-length branch word W, accumulate the Berggren matrix product M_W
- Compute integer norm: norm(W) = ||M_W · v||²  (clearing denominators)
- The norm is an integer; its size is bounded by the spectral radius of M_W

### Smoothness Sieving
- For branch words of bounded length L (L = 20–40), compute norm(W)
- Check if norm(W) is B-smooth (all prime factors ≤ B)
- Start B = 2^20, scale with N as needed
- Also track one-large-prime relations (norm has one factor > B)

### Relation Collection
- Each smooth norm yields a linear relation over exponent vectors of factor base
- Factor base: first 100–200 primes below B
- Collect target: |factor base| + 20 independent relations

### Sparse Linear Algebra
- Build relation matrix over GF(2) (parity of exponents)
- Matrix dimensions: ~200 relations × ~200 factor base primes
- Sparse Gaussian elimination (not full NFS lattice)
- A nullvector gives a congruence x² ≡ y² (mod N)

### Congruence Extraction
- From nullvector: construct x, y from relation exponents
- Factor = gcd(x − y, N)

### Module
- `insideout/orbit_smooth_relation.py` — `orbit_smooth_relation_factor(N)`
- Callable from `factor.py` as `"orbit_relation"` after all other methods fail
- Also callable standalone for isolated experimentation

### Files
- `insideout/orbit_smooth_relation.py` — new module
- `insideout/factor.py` — add import and fallback entry
- `tests/test_new_methods.py` — unit + integration tests

### Tests
1. **Unit:** norm computation correctness (norm is integer, matches explicit matrix application)
2. **Unit:** smoothness detection (known smooth number detected)
3. **Integration:** generates ≥1 smooth relation for a known semiprime
4. **Integration:** linear algebra step produces a correct factor from collected relations
5. **Stress:** nullspace computation on ≥50 relations succeeds and yields gcd ≠ 1, N

---

## Architecture

```
factor(N)
  ├── lucas_ppt            (fast: <1ms)
  ├── projective_chart     (Phase 1: chart + distinguished + batch GCD)
  ├── orbit_relation       (Phase 2: smooth relations → NFS-style)
  ├── resonance_cascade    (slow: 50–100ms)
  └── ...remaining chain...
```

Phase 2 is placed after most established methods in the fallback chain, so it
only runs when cheaper methods have exhausted their time budget.

---

## Complexity Note
Phase 1 remains O(N^1/4) generic complexity — it is a constant-factor and
memory improvement over the current 3-minor worker.

Phase 2 is the only path that could move below O(N^1/4). It requires
collecting many smooth relations and solving a sparse linear system.
Success is not guaranteed; it is a research experiment.
