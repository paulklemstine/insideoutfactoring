# Inside-Out Factoring: Full Spectral Toolkit — Design Spec

**Date:** 2026-07-24  
**Goal:** Implement and expand the Inside-Out Factoring framework from `paper.md`, validating the theory on small semiprimes and then scaling to compete with established methods.

---

## 1. Architecture

### Module Structure

```
insideout/
├── __init__.py          # Package init, version
├── berggren.py          # U, A, D matrices + inverses + tree traversal primitives
├── triples.py           # PPT generation, validation, scaling, (m,n) parametrization
├── energy.py            # Energy spectrum E(v)=ln(c), spectral gap computation, resonance detection
├── cf_guide.py          # Continued fraction convergents of √N, Berggren branch prediction
├── modular.py           # Modular resonance filters (pruning by small prime compatibility)
├── gaussian.py          # Gaussian integer (m,n) representation ↔ triple conversion, 2×2 ops
├── inside_out.py        # Main Inside-Out algorithm: radial expansion from energy well
├── wavefront.py         # Parallel wavefront expansion: evaluate energy shells in batch
└── factor.py            # Top-level API: factor(N) → (p, q)
tests/
├── test_berggren.py
├── test_triples.py
├── test_energy.py
├── test_cf_guide.py
├── test_modular.py
├── test_gaussian.py
├── test_inside_out.py
├── test_wavefront.py
└── test_factor.py
```

### Data Flow

```
N → factor(N)
    ├── cf_guide: compute CF convergents of √N → branch predictions
    ├── inside_out: start at energy well, expand radially
    │   ├── berggren: apply U⁻¹/A⁻¹/D⁻¹ (or Gaussian 2×2 equivalents)
    │   ├── energy: compute E(v), check spectral gaps, prune by energy bounds
    │   ├── modular: eliminate nodes incompatible with N (mod small primes)
    │   ├── cf_guide: steer toward predicted branch at each level
    │   └── wavefront: evaluate energy shells in parallel batches
    └── gaussian: (m,n) representation for algebraic shortcuts
```

### Key Design Decision: No Floating Point

All arithmetic uses Python's built-in arbitrary-precision integers. Energy comparisons avoid ln() by comparing hypotenuses directly: c₁ < c₂ ⟺ E(v₁) < E(v₂). The continued fraction of √N is computed via integer-only algorithms (standard CF algorithm for quadratic irrationals). The only place a transcendental function appears is in theoretical analysis — never in the hot path.

---

## 2. Core Algorithms

### 2.1 Berggren Matrices & Inverse Descent (`berggren.py`)

The three Berggren matrices generate all PPTs from (3,4,5):

```
U = [[ 1, -2, 2], [ 2, -1, 2], [ 2, -2, 3]]
A = [[ 1,  2, 2], [ 2,  1, 2], [ 2,  2, 3]]
D = [[-1,  2, 2], [-2,  1, 2], [-2,  2, 3]]
```

Their inverses (all unimodular, det=1):

```
U⁻¹ = [[ 1,  2, -2], [ 2,  1, -2], [ 2,  2, -3]]  (verify numerically)
A⁻¹ = [[-1,  2, -2], [-2,  1,  2], [-2, -2,  3]]  (verify numerically)
D⁻¹ = [[ 1, -2,  2], [-2,  1,  2], [-2,  2,  3]]  (verify numerically)
```

**Inverse verification is a unit test requirement:** U·U⁻¹ = A·A⁻¹ = D·D⁻¹ = I₃.

The Inside-Out search starts near the pseudo-node at the well (legs ≈ √N) and applies inverse transformations to descend toward the root, or forward transformations to climb away from the well. We jump to the neighborhood of the target and search locally.

### 2.2 Continued Fraction Steering (`cf_guide.py`)

The CF expansion of √N produces convergents pₖ/qₖ that approximate √N. Since tan(θ/2) = p/q at the target node, these convergents predict the ideal branch at each tree level.

**Algorithm:**
1. Compute regular CF expansion of √N (integer-only, using the quadratic irrational algorithm)
2. For each convergent pₖ/qₖ, compute the "slope" pₖ/qₖ
3. At each tree node v = (a, b, c), compare the node's slope b/a with the CF-predicted slope
4. Choose the Berggren child (U, A, or D) whose slope is closest to the CF prediction
5. Also evaluate the other two children at limited depth as fallback branches

This prunes the 3-ary tree to ~1 branch per level, reducing search from exponential to logarithmic depth.

### 2.3 Energy Spectrum (`energy.py`)

Energy of node v = (a, b, c): E(v) = ln(c). In practice, we compare c values directly.

**Spectral gap:** ΔE(v) = E(child) − E(v) ≈ ln(c_child / c). The spectral resonance (Theorem 3.1) manifests as minimized ΔE along the correct branch path.

**Energy bounds for pruning:** If c > N + ε for some tolerance ε, the node cannot produce a factorization of N. We use energy bounds to terminate branches early.

### 2.4 Modular Resonance Filters (`modular.py`)

A PPT (a, b, c) can reveal factors of N only if N is compatible with the triple's residue structure. For each small prime ℓ ∈ {2, 3, 5, 7, 11, 13, ...}:

- Precompute which residue classes mod ℓ can appear as the first leg of a PPT
- Check compatibility with N mod ℓ
- Eliminate ~70-80% of candidates with O(1) per-node cost

This is a sieve-like prefilter applied before expensive perfect-square checks.

### 2.5 Gaussian Integer Parametrization (`gaussian.py`)

Every PPT (a, b, c) with a = m²−n², b = 2mn, c = m²+n² corresponds to z = m+ni ∈ Z[i]. The Berggren matrices simplify to 2×2 transformations on (m, n).

Benefits:
- Matrix operations: 3×3 → 2×2 (faster)
- Energy: |z|² = m²+n² = c (direct)
- Coprimality: gcd(m,n) = 1 is easy to check
- Multiplication in Z[i] gives algebraic shortcuts for resonance checks

### 2.6 Wavefront Parallel Search (`wavefront.py`)

Instead of evaluating nodes one at a time, group all nodes at "energy distance" R from the well into a wavefront:

- Generate all nodes with c in [exp(R), exp(R+ε)] for small ε
- Apply modular filters to the entire batch
- Check resonance for survivors in parallel (embarrassingly parallel)
- Implementation: Python `multiprocessing.Pool` or `concurrent.futures`

### 2.7 Main Algorithm (`factor.py`)

```python
def factor(N: int) -> tuple[int, int]:
    """Factor a semiprime N = p*q using Inside-Out traversal."""
    # 1. Edge cases
    if N < 4 or N % 2 == 0:
        return _handle_trivial(N)

    # 2. CF convergents of √N (branch predictions)
    cf_convergents = cf_guide.convergents(N)

    # 3. Initialize at energy well
    well = inside_out.central_well(N)

    # 4. Radial expansion with CF steering + modular filtering
    for wavefront in wavefront.expand(well, cf_convergents, N):
        candidates = modular.filter_wavefront(wavefront, N)
        for v in candidates:
            if resonance.check(N, v):
                p, q = resonance.extract_factors(N, v)
                assert p * q == N
                return (min(p, q), max(p, q))

    # 5. Fallback: trial division up to N^{1/3}
    return _fallback(N)
```

---

## 3. New Techniques Beyond the Paper

### 3.1 CF-Guided Descent (extending Theorem 3.1)

The paper's proof sketch connects Berggren matrices to Möbius transformations and continued fractions. We formalize this connection:

- Each Berggren matrix corresponds to a specific Möbius transformation on the slope parameter
- The CF convergents of √N encode the ideal sequence of Möbius transformations
- This gives a *deterministic* path prediction, not just a heuristic

### 3.2 Modular Resonance Sieving

Not in the paper. We observe that PPTs satisfy strong congruence constraints mod small primes. By precomputing compatibility tables, we achieve sieve-like filtering with O(1) cost per node — analogous to the wheel sieve in trial division, but applied to the Pythagorean tree topology.

### 3.3 Gaussian Integer Acceleration

Not in the paper. The (m, n) parametrization reduces matrix operations from 3×3 to 2×2 and makes the coprimality constraint trivial. The norm map |z|² = c directly gives energy without computing a hypotenuse.

### 3.4 Wavefront Batching

The paper's algorithm is described sequentially. We add batch evaluation of energy shells, enabling parallel search. Each wavefront at radius R contains O(3^R) nodes (before pruning), but CF steering + modular filters reduce this to approximately O(1) per level.

---

## 4. Testing & Validation

### Unit Tests
- Berggren matrices: U·U⁻¹ = I, children of (3,4,5) are valid PPTs
- PPT validation: a²+b²=c², gcd(a,b)=1, a,b opposite parity
- Gaussian ↔ triple round-trip: (m,n) → (a,b,c) → (m,n)
- Energy ordering: c₁ < c₂ ⟹ E(v₁) < E(v₂) without ln
- Modular filters: known triples pass, incompatible ones fail
- CF convergents: verify against known CF expansions

### Integration Tests (known semiprimes)
- 15 = 3×5, 21 = 3×7, 35 = 5×7, 77 = 7×11
- 437 = 19×23, 667 = 23×29
- 5-digit products of 2-3 digit primes

### Scaling Benchmarks
- 8-bit → 16-bit → 32-bit → 64-bit semiprimes
- Metrics: nodes explored, wall-clock time, success rate
- Baselines: trial division, Fermat's method

### Validation Principle
Every result: `p * q == N` and `is_prime(p)` and `is_prime(q)`.

### Scaling Strategy
1. Validate on 2-3 digit semiprimes (all techniques working)
2. Benchmark on 4-8 digit semiprimes (measure CF steering advantage)
3. Tune modular filter thresholds and wavefront sizes on 10-20 digit
4. Push to 30-50 digit with gmpy2 acceleration
5. Explore 100+ digit frontier

---

## 5. Dependencies

- **Python 3.10+** (stdlib only for core)
- **pytest** for testing
- **sympy** for prime verification and CF utilities (optional, for validation)
- **gmpy2** for fast arithmetic on large integers (optional, for scaling phase)

---

## 6. Future Work (not in scope for initial implementation)

- Quantum walks on the Pythagorean Cayley graph (as mentioned in paper §5)
- Sub-exponential complexity proof for unbalanced factors
- GPU acceleration of wavefront evaluation
- Integration with GNFS for hybrid approach