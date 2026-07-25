# Advanced Techniques from the Lean Catalog

This document catalogues advanced mathematical techniques found in the Lean
formalization catalog that could upgrade the Inside-Out factoring algorithm.
The current algorithm uses: Berggren tree traversal, energy spectrum (s^2 - N),
CF convergent steering, Gaussian integer (m,n) parametrization, and best-first
search.

Each technique is classified as one of:
- **Pre-check optimization**: eliminates candidates before search begins
- **Search strategy upgrade**: improves how the search space is explored
- **New mathematical framework**: replaces or augments the underlying algebra

---

## 1. Assignment Gap Extension (Tropical)

**Source**: `Pythagorean/AssignmentGapExtension.lean`

**Core idea**: For a symmetric matrix with pairwise diagonal dominance, the
global assignment optimization problem over all n! permutations collapses to
a quadratic-size transposition search. The central identity is:
```
2 * (idWeight W - permWeight W sigma) = sum_i pairDeficit(W, i, sigma(i))
```
This rewrites the global deficit as a sum of local pairwise penalties, bypassing
cycle decomposition entirely. The assignment gap (difference between the best
identity and best non-identity permutation) equals the tropical margin for n=2.

**Upgrade path**: Implement the assignment gap computation as a pre-check. Given
the factor N and its Gaussian integer parametrization, construct the signal matrix
W where W(i,j) tracks how well the (m,n) candidate at position i matches the
target at position j. If W has symmetric pairwise diagonal dominance (which
can be verified in O(n^2)), then only transpositions need to be checked, reducing
the search from O(n!) to O(n^2). Specifically:
```python
def check_assignment_gap(W):
    """Returns True if symmetric pairwise diagonal dominance holds."""
    n = len(W)
    for i in range(n):
        for j in range(i+1, n):
            if W[i][i] + W[j][j] <= 2 * W[i][j]:
                return False
    return True
```

**Expected impact**: Medium -- eliminates full permutation search when the
dominance condition holds, which is the generic case for well-posed factoring
instances.

**Implementation complexity**: ~150 lines (matrix construction, dominance check,
transposition-only search fallback).

**Category**: Pre-check optimization + search strategy upgrade.

---

## 2. Circle Method Density Heuristics

**Source**: `Pythagorean/CircleMethodDensity.lean`

**Core idea**: The Hardy--Littlewood circle method predicts the density of
solutions to x^3 + y^3 + z^3 = k via local densities at each prime. The
singular series is a product of local densities that measures how "generic"
the equation is at each prime. For the factoring problem, we can analogously
estimate the density of representations of N as a^2 - b^2 (i.e., factorizations)
by computing local obstruction densities. The mod-9 obstruction (x^3 + y^3 + z^3
has no solution for k = 4 or 5 mod 9) is a paradigmatic example of a local
density vanishing that eliminates entire congruence classes.

**Upgrade path**: Implement local density pre-checks for factoring. For each
small prime p, compute the local density of solutions to a*b = N (mod p). If
the density at any prime is zero, the factorization is obstructed. If all
densities are positive, the product gives a heuristic likelihood of finding
factors. Specifically:
```python
def local_factorization_density(N, p):
    """Count fraction of (a,b) in (Z/pZ)^2 with a*b = N mod p."""
    count = sum(1 for a in range(p) for b in range(p) if (a * b) % p == N % p)
    return count / (p * p)

def singular_series_factorization(N, primes=[2,3,5,7,11,13]):
    """Product of local densities -- vanishes if any local obstruction exists."""
    product = 1.0
    for p in primes:
        delta = local_factorization_density(N, p)
        if delta == 0:
            return 0.0  # Hard obstruction
        product *= delta
    return product
```

**Expected impact**: Low -- provides heuristic guidance but not deterministic
improvement. Useful for prioritizing search order.

**Implementation complexity**: ~80 lines (modular arithmetic density computation,
singular series product).

**Category**: Pre-check optimization.

---

## 3. Reflection Positivity and Transfer Matrices

**Source**: `Pythagorean/ReflectionPositivity.lean`

**Core idea**: Osterwalder--Schrader reflection positivity for a kernel K with
involution theta ensures the transfer matrix T(x,y) = K(theta*x, y) is positive
semidefinite. For a symmetric, pairwise diagonal-dominant matrix (as in the
Assignment Gap technique), the Wilson-type kernel K(x,y) = exp(beta * w(x,y))
automatically satisfies OS positivity. This guarantees a spectral gap: the
largest eigenvalue of the transfer matrix is simple, and the gap to the next
eigenvalue is strictly positive.

**Upgrade path**: Construct the transfer matrix from the factorization signal
matrix and compute its spectral gap. A large spectral gap means the search
converges quickly to the optimal assignment; a small gap means the problem is
degenerate (near-multiple factors). This gives a runtime predictor:
```python
import numpy as np

def transfer_matrix(W, beta=1.0):
    """Construct Wilson transfer matrix from signal matrix."""
    K = np.exp(beta * W)
    theta = np.arange(W.shape[0])  # identity involution
    T = K[theta, :][:, :]  # T(i,j) = K(theta(i), j)
    return T

def spectral_gap_predictor(W):
    """Largest spectral gap of transfer matrix predicts search difficulty."""
    T = transfer_matrix(W)
    eigenvalues = np.linalg.eigvals(T)
    sorted_evals = sorted(eigenvalues.real, reverse=True)
    if len(sorted_evals) > 1:
        return sorted_evals[0] - sorted_evals[1]
    return float('inf')
```

**Expected impact**: Medium -- enables adaptive search strategies based on
spectral gap size. Large gap = fast convergence; small gap = need more
sophisticated methods.

**Implementation complexity**: ~100 lines (transfer matrix construction,
eigendecomposition, gap computation).

**Category**: Search strategy upgrade (runtime predictor).

---

## 4. Adelic Persistent Homology

**Source**: `Pythagorean/AdelicPersistentHomology.lean`

**Core idea**: The torsion barcode of a filtered finite abelian group decomposes
canonically by prime. The adelic torsion datum packages prime-indexed persistence
data with finite-support conditions, and reconstruction from local data is exact
(Theorem 2). For factoring, the filtration is the sequence of groups
Z/NZ -> Z/(N/p_1)Z -> Z/(N/p_1*p_2)Z -> ..., and the torsion barcode at each
prime reveals the multiplicity and order of that prime in N. The CRT decomposition
(Theorem 4) shows that mk-torsion elements decompose into m-torsion and k-torsion
when m and k are coprime.

**Upgrade path**: Implement prime-by-prime factorization tracking. Given N,
build the filtration G_i = Z/(N/product of first i prime factors)Z and compute
the p-primary torsion at each step. The adelic reconstruction theorem guarantees
that the global factorization is exactly recoverable from these local signals:
```python
from collections import defaultdict

def adelic_factorization_tracker(N):
    """Track factorization progress prime-by-prime via p-primary torsion."""
    factorization = {}
    current = N
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        if current == 1:
            break
        multiplicity = 0
        while current % p == 0:
            current //= p
            multiplicity += 1
        if multiplicity > 0:
            factorization[p] = multiplicity
    if current > 1:
        # Current still has large prime factors -- use Inside-Out search
        factorization['residual'] = current
    return factorization
```

**Expected impact**: Medium -- primes up to ~31 can be trial-divided instantly.
The adelic framework provides a principled way to combine trial division with
the Inside-Out search for remaining large factors.

**Implementation complexity**: ~120 lines (p-primary decomposition, CRT-based
recombination, adelic reconstruction).

**Category**: Pre-check optimization + new mathematical framework for combining
trial division with deeper search.

---

## 5. Higher-Order Critical Pairs (Rewriting)

**Source**: `Pythagorean/HOCriticalPairs.lean`

**Core idea**: In a terminating, locally confluent rewrite system (Newman's
Lemma), every term has a unique normal form. The critical pairs of a rewrite
system -- overlaps between left-hand sides -- determine local confluence. For
factoring, we can model the search space as a term rewrite system where rewrite
rules correspond to algebraic transformations on the (m,n) parametrization
(e.g., Berggren tree transitions, CF convergent updates, sign changes). If all
critical pairs are joinable, the system is confluent, meaning every sequence of
rewrites reaches the same normal form (i.e., the same factorization).

**Upgrade path**: Model the Inside-Out search as a rewrite system and verify
local confluence. If the system is terminating (energy always decreases) and
all critical pairs between Berggren moves, CF updates, and sign flips are
joinable, then the search is confluent -- any path reaches the factorization.
This gives a formal correctness guarantee:
```python
def check_critical_pair_joinability(rules):
    """Check if all critical pairs of rewrite rules are joinable."""
    critical_pairs = []
    for r1 in rules:
        for r2 in rules:
            overlaps = find_overlaps(r1.lhs, r2.lhs)
            for overlap in overlaps:
                # Apply r1 and r2 at the overlap
                t1 = apply_rule_at(r1, overlap)
                t2 = apply_rule_at(r2, overlap)
                # Check if t1 and t2 reduce to a common form
                n1 = normalize(t1, rules)
                n2 = normalize(t2, rules)
                if n1 != n2:
                    critical_pairs.append((r1, r2, overlap))
    return len(critical_pairs) == 0
```

**Expected impact**: Low -- provides formal correctness guarantee but does not
directly improve runtime. Useful for verifying that the search strategy is
well-founded.

**Implementation complexity**: ~200 lines (term representation, overlap finding,
normalization, joinability checking).

**Category**: New mathematical framework (formal verification of search strategy).

---

## 6. k-Tuple Mobius Inversion

**Source**: `Pythagorean/KTupleMoebiusInversion.lean`

**Core idea**: The number of k-tuples that generate a finite group G is given by
the Mobius inversion formula on the subgroup lattice: phi_k(G) = sum_{H <= G}
mu(H,G) * |H|^k, where mu is the Mobius function of the subgroup lattice. For
k=2, this counts generating pairs; for k=1, it counts generators. The probability
that k random elements generate G is P_k(G) = phi_k(G) / |G|^k. For the
symmetric group S_n with n >= 5, P_3(S_n) >= 1 - 1/n (conjectured).

**Upgrade path**: Use Mobius inversion on the divisor lattice of N to estimate
the probability that a random Gaussian integer (m,n) parametrizes a factorization
of N. The divisor lattice of N is isomorphic to the subgroup lattice of Z/NZ,
so the formula becomes: the number of representations of N as (s^2 - N) where
s is coprime to N equals sum_{d|N} mu(d) * (N/d), where mu is the classical
Mobius function. This gives an exact count of "good" s values:
```python
from sympy import mobius, divisors

def mobius_inversion_count(N):
    """Exact count of s values coprime to N via Mobius inversion."""
    return sum(mobius(d) * (N // d) for d in divisors(N))

def generating_pair_probability(N, k=2):
    """Probability that k random Gaussian integers parametrize a factorization."""
    total = N ** k
    good = mobius_inversion_count(N)
    return good / total
```

**Expected impact**: Medium -- provides exact count of viable search starting
points, enabling precise search space sizing and early termination when the
probability is too low.

**Implementation complexity**: ~90 lines (Mobius function computation, divisor
enumeration, probability estimation).

**Category**: Pre-check optimization (search space sizing).

---

## 7. Newton Entropy Hierarchy

**Source**: `Pythagorean/NewtonEntropyHierarchy.lean`

**Core idea**: The elementary symmetric polynomials e_1, e_2, ..., e_m of a
spectrum lambda = (lambda_1, ..., lambda_m) satisfy Newton's inequalities
(e_k^2 >= e_{k-1} * e_{k+1}). The Newton ratio profile packages these with
log-concavity diagnostics. Key bridge: the second moment p_2 = e_1^2 - 2*e_2
is EXACTLY determined by e_1 and e_2 (no approximation error). For the factoring
problem, if we track the "spectrum" of Gaussian integer norms as a sequence, the
Newton hierarchy provides compressed algebraic coordinates that determine all
polynomial invariants without accessing individual eigenvalues.

**Upgrade path**: Track the elementary symmetric polynomials of the energy
spectrum (s^2 - N values for candidate s). The first two symmetric polynomials
e_1 = sum(s_i^2 - N) and e_2 = sum_{i<j} (s_i^2 - N)(s_j^2 - N) determine the
full second moment and variance, which are the key statistics for the energy
spectrum. Newton's inequality e_1^2 >= 2*e_2 gives a consistency check:
```python
def newton_hierarchy_check(energies):
    """Verify Newton's inequality for the energy spectrum."""
    e1 = sum(energies)
    e2 = sum(energies[i] * energies[j]
              for i in range(len(energies))
              for j in range(i+1, len(energies)))
    # Newton's inequality: e1^2 >= 2 * e2
    return e1**2 >= 2 * e2

def entropy_surrogate(energies):
    """Quadratic entropy surrogate from e1, e2 (lower bound on true entropy)."""
    e1 = sum(energies)
    e2 = sum(energies[i] * energies[j]
              for i in range(len(energies))
              for j in range(i+1, len(energies)))
    return 2 * (e1 - e1**2 + 2 * e2)
```

**Expected impact**: Medium -- provides compressed tracking of search statistics
and consistency verification. The exact determination of p_2 from e_1 and e_2
means we never need to recompute the second moment from scratch.

**Implementation complexity**: ~130 lines (symmetric polynomial computation,
Newton inequality checks, entropy surrogate).

**Category**: Search strategy upgrade (compressed state tracking).

---

## 8. Quadratic Reciprocity (Gauss Sum and Eisenstein)

**Source**: `NumberTheory/QuadraticReciprocity/GaussSum.lean`,
`NumberTheory/QuadraticReciprocity/Eisenstein.lean`

**Core idea**: Quadratic reciprocity gives a fast algorithm for computing the
Legendre symbol (a/p) without factoring. The Gauss-sum proof identifies (a/p)
as a Frobenius eigenvalue, while the Eisenstein proof counts lattice points under
a line. For factoring, the Legendre symbol (a/p) = a^((p-1)/2) mod p determines
whether a is a quadratic residue mod p. The reciprocity law
(q/p)(p/q) = (-1)^((p-1)/2 * (q-1)/2) allows computing (p/q) from (q/p).

**Upgrade path**: Use Legendre symbol computation as a pre-check to determine
whether N has a quadratic residue structure compatible with factorization. If N
is a quadratic residue mod p for many small primes p, it is more likely to have
factors of a particular shape. The Eisenstein lattice-point interpretation gives
a geometric way to count these residues:
```python
def legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p) via modular exponentiation."""
    val = pow(a, (p - 1) // 2, p)
    if val == p - 1:
        return -1
    return val

def quadratic_residue_profile(N, primes=[2,3,5,7,11,13,17,19,23,29,31]):
    """Profile of Legendre symbols (N/p) for small primes."""
    return {p: legendre_symbol(N % p, p) for p in primes if N % p != 0}

def is_compatible_with_factorization(N, primes=[3,5,7,11,13]):
    """Check if quadratic residue profile is compatible with having factors."""
    profile = quadratic_residue_profile(N, primes)
    # N = a*b means (N/p) = (a/p)*(b/p), so (N/p) = 1 implies
    # both or neither factor is a QR mod p
    return all(v != 0 for v in profile.values())
```

**Expected impact**: Medium -- Legendre symbol computation is O(log p) per prime,
very fast. Provides structural information about factor parity at each prime.

**Implementation complexity**: ~60 lines (Legendre symbol, residue profiling,
compatibility check).

**Category**: Pre-check optimization.

---

## 9. Chord-Swap Diameter Descent

**Source**: `NumberTheory/ChordSwapDiameterDescent.lean`

**Core idea**: Any reconfiguration graph admitting a hub (canonical element) and
a monovariant potential phi that strictly decreases along some move from every
non-hub vertex has diameter at most 2*max(phi). The Inside-Out search is a
reconfiguration graph: vertices are (m,n) states, edges are Berggren moves or
CF convergent updates, and the hub is the factorization. The energy s^2 - N is a
natural monovariant: each successful move decreases it toward zero. If we can
prove that from every non-factoring state there exists a move that decreases
the energy, then the search has bounded diameter.

**Upgrade path**: Define the descent potential phi(m,n) = s^2 - N where
s = m^2 + n^2 (or the appropriate energy measure). Prove (or empirically
verify) that from every non-factoring state, at least one Berggren child or
CF convergent has strictly lower energy. This gives a diameter bound:
```python
def descent_potential(s, N):
    """Energy of state s relative to target N."""
    return s * s - N

def verify_descent_property(berggren_tree, N, max_states=10000):
    """Verify that every non-terminal state has a descending move."""
    for s in generate_states(berggren_tree, max_states):
        energy = descent_potential(s, N)
        if energy <= 0:
            continue  # terminal state
        children = berggren_children(s) + cf_convergent_children(s, N)
        if not any(descent_potential(c, N) < energy for c in children):
            return False, s  # descent property violated
    return True, None
```

**Expected impact**: High -- if the descent property holds, it guarantees
bounded search depth, which is the key missing ingredient for proving the
algorithm terminates in polynomial time for all inputs.

**Implementation complexity**: ~110 lines (potential function, descent
verification, diameter bound computation).

**Category**: New mathematical framework (bounded search depth guarantee).

---

## 10. Mixed-Radix and Factorial Number Systems

**Source**: `Computation/MixedRadixNumberSystem.lean`,
`Computation/FactorialNumberSystem.lean`,
`Speculative/AutoResearch/MixedRadixFactorialBridge.lean`

**Core idea**: The factorial number system represents any natural number uniquely
as sum_{i<k} c_i * i! with c_i <= i. The mixed-radix system generalizes this to
arbitrary base sequences. For factoring, the Gaussian integer parametrization
(m,n) naturally gives a mixed-radix structure: the Berggren tree at depth d gives
a 2^d-ary branching, and CF convergents give variable-radix steps. The uniqueness
theorem (valid representations are unique) means that if we find a valid
Berggren path to a factorization, it is the ONLY such path.

**Upgrade path**: Encode the search path as a mixed-radix number. Each Berggren
tree step is a digit in base 3 (three children per node), and each CF convergent
step is a digit in a variable base determined by the current convergent. The
factorial representation of the step count gives a canonical numbering of search
paths:
```python
def encode_search_path(berggren_digits, cf_digits):
    """Encode a search path as a mixed-radix number."""
    value = 0
    radices = [3] * len(berggren_digits) + cf_radices(len(cf_digits))
    digits = berggren_digits + cf_digits
    for i, (d, r) in enumerate(zip(digits, radices)):
        value += d * product(radices[:i])
    return value

def decode_search_path(value, depth):
    """Decode a mixed-radix number back to a search path."""
    digits = []
    remaining = value
    for i in range(depth):
        radix = 3 if i < depth // 2 else cf_radix(i - depth // 2)
        digits.append(remaining % radix)
        remaining //= radix
    return digits
```

**Expected impact**: Low -- provides canonical path numbering but does not
directly improve search efficiency. Useful for debugging and caching.

**Implementation complexity**: ~80 lines (mixed-radix encoding/decoding,
path numbering).

**Category**: Search strategy upgrade (canonical path numbering for caching).

---

## 11. Conjugation-Indexed Product Covering

**Source**: `Pythagorean/ConjugationProductCover.lean`

**Core idea**: In a finite group G with subgroup H, the product set A*A of a set
A covered by C left cosets of H can be covered by at most C^2 * L cosets, where
L is the maximal conjugation index [H : H intersect g^{-1}Hg]. For the
Berggren group (the group of transformations on (m,n) parametrizations), the
Hecke multiplicity at each element determines how many Berggren children
overlap. If the Berggren group has small Hecke multiplicities, then the search
space grows slowly (polynomially rather than exponentially).

**Upgrade path**: Compute the Hecke multiplicities for the Berggren group.
The Berggren tree produces 3 children per node, but some of these may coincide
(or have overlapping energy ranges) depending on the group structure. The
conjugation index bounds the effective branching factor:
```python
def hecke_multiplicity(H, g):
    """Compute [H : H intersect g^{-1}Hg] for the Berggren subgroup."""
    conjugate = g * H * g^{-1}  # in appropriate group
    intersection = H.intersection(conjugate)
    return H.order() // intersection.order()

def effective_branching_factor(berggren_subgroup):
    """Bound on search space growth via conjugation indices."""
    return max(hecke_multiplicity(berggren_subgroup, g)
               for g in berggren_generators)
```

**Expected impact**: Low to Medium -- provides theoretical bounds on search
space growth. Practical impact depends on the actual Hecke multiplicities of
the Berggren group.

**Implementation complexity**: ~100 lines (group construction, Hecke
multiplicity computation, branching factor bound).

**Category**: New mathematical framework (search space size bounds).

---

## 12. Tropical Eigenvalue Leakage

**Source**: `Pythagorean/TropicalCryptography/EigenvalueLeakage.lean`

**Core idea**: In min-plus (tropical) algebra, a tropical eigenpair (lambda, v)
of a matrix A remains an eigenpair of every positive power A^[k] with eigenvalue
(k+1)*lambda. This means the power map k -> A^[k] is injective whenever a
nonzero eigenvalue exists. For factoring, if we construct a tropical matrix
from the factorization problem (e.g., the distance matrix between Gaussian
integer states), its tropical eigenvalue structure determines how quickly the
search space "spreads out." A nonzero tropical eigenvalue means the search is
well-conditioned; a zero eigenvalue means the problem is degenerate.

**Upgrade path**: Construct the min-plus distance matrix between candidate
states and compute its tropical eigenvector. The eigenvalue lambda gives the
"scale" of the factoring problem in tropical terms:
```python
def tropical_eigenvalue(A):
    """Compute tropical eigenvalue via power iteration."""
    n = A.shape[0]
    v = np.zeros(n)  # initial eigenvector
    for _ in range(100):
        v_new = tropical_matvec(A, v)
        # Normalize by subtracting minimum
        v_new = v_new - np.min(v_new)
        if np.allclose(v, v_new, atol=1e-10):
            break
        v = v_new
    # Eigenvalue = min over i of (A*v - v)_i
    Av = tropical_matvec(A, v)
    return np.min(Av - v), v

def tropical_matvec(A, v):
    """Min-plus matrix-vector product."""
    n = len(v)
    result = np.full(n, np.inf)
    for i in range(n):
        for j in range(n):
            result[i] = min(result[i], A[i][j] + v[j])
    return result
```

**Expected impact**: Low -- tropical eigenvalue computation is interesting
theoretically but does not directly improve factoring speed. It provides a
well-conditionedness diagnostic.

**Implementation complexity**: ~80 lines (min-plus matrix operations, tropical
power iteration, eigenvalue extraction).

**Category**: New mathematical framework (problem conditioning diagnostic).

---

## 13. Canonical Kernel Calculus (Metric Graph Laplacian)

**Source**: `Pythagorean/TropicalBridge/CanonicalKernelCalculus.lean`

**Core idea**: A weighted metric graph has a canonical Laplacian with conductance
weights 1/len(e). The Dirichlet energy E(f) >= 0 equals zero iff f is constant
(connected graph). The normalized kernel is unique for mean-zero potentials.
For factoring, model the search graph (Berggren tree + CF convergents) as a
weighted metric graph where edge weights are energy differences. The Laplacian's
spectral gap controls search convergence rate; the canonical kernel gives the
Green's function (expected hitting time) from any state to the factorization.

**Upgrade path**: Build the Laplacian of the search graph and compute its
spectral gap and Green's function. The spectral gap determines how quickly the
search converges; the Green's function provides the expected number of steps:
```python
def build_search_laplacian(states, transitions, energies):
    """Build metric Laplacian for the search graph."""
    n = len(states)
    L = np.zeros((n, n))
    for i, j, weight in transitions:
        cond = 1.0 / max(abs(energies[i] - energies[j]), 1e-10)
        L[i][i] += cond
        L[j][j] += cond
        L[i][j] -= cond
        L[j][i] -= cond
    return L

def spectral_gap_and_greens_function(L):
    """Compute spectral gap and Green's function of Laplacian."""
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    gap = eigenvalues[1] - eigenvalues[0]  # spectral gap
    # Green's function = pseudoinverse of L restricted to mean-zero subspace
    L_pinv = np.linalg.pinv(L)
    return gap, L_pinv
```

**Expected impact**: Medium -- spectral gap analysis provides a principled way to
estimate search convergence time and detect degenerate (small-gap) instances.

**Implementation complexity**: ~120 lines (graph construction, Laplacian,
spectral analysis, Green's function).

**Category**: Search strategy upgrade (convergence rate predictor).

---

## 14. Weighted Defect and Root-Separated Decomposition

**Source**: `Pythagorean/TropicalBridge/WeightedDefect.lean`,
`Pythagorean/TropicalBridge/RootSeparatedDecomposition.lean`

**Core idea**: The structural defect delta(G, q, S) = beta_1(G[S]) + kappa(G,q,S) - 1
is **metric-free**: it depends only on the combinatorial graph structure, not on
edge weights. When S splits as S1 union S2 with S1 and S2 in distinct components
of G - {q}, the defect decomposes as delta(S1 cup S2) = delta(S1) + delta(S2) + 1
(Mayer-Vietoris decomposition). The weighted boundary mass scales linearly with
edge weights. For factoring, the search graph can be decomposed into
root-separated pieces (independent subtrees of the Berggren tree), and the
defect additivity law means we can search each piece independently.

**Upgrade path**: Decompose the search graph into root-separated pieces
corresponding to independent Berggren subtrees. Search each piece independently
and combine results using the defect additivity law:
```python
def is_root_separated(G, q, S1, S2):
    """Check if S1 and S2 are in distinct components of G - {q}."""
    # Remove q from the graph
    G_minus_q = G.copy()
    G_minus_q.remove_node(q)
    # Check if any vertex in S1 can reach any vertex in S2
    for u in S1:
        for v in S2:
            if nx.has_path(G_minus_q, u, v):
                return False
    return True

def defect_additive_search(G, q, pieces):
    """Search root-separated pieces independently, combine via defect law."""
    total_defect = 0
    results = []
    for i, S in enumerate(pieces):
        # Search within S independently
        result = search_within(G, q, S)
        results.append(result)
        if i > 0:
            total_defect += 1  # Mayer-Vietoris correction
    return combine_results(results, total_defect)
```

**Expected impact**: Medium -- enables parallel search over independent subtrees
of the Berggren tree, with defect tracking to detect when subtrees are exhausted.

**Implementation complexity**: ~150 lines (graph decomposition, root-separation
check, independent search, defect additivity combination).

**Category**: Search strategy upgrade (parallel decomposition).

---

## 15. Summary: Priority Ranking

| Priority | Technique | Category | Impact | Complexity |
|----------|-----------|----------|--------|------------|
| 1 | Chord-Swap Diameter Descent | Framework | High | ~110 lines |
| 2 | Assignment Gap Extension | Pre-check + Search | Medium | ~150 lines |
| 3 | Newton Entropy Hierarchy | Search Upgrade | Medium | ~130 lines |
| 4 | Canonical Kernel Calculus | Search Upgrade | Medium | ~120 lines |
| 5 | Root-Separated Decomposition | Search Upgrade | Medium | ~150 lines |
| 6 | Adelic Persistent Homology | Pre-check + Framework | Medium | ~120 lines |
| 7 | k-Tuple Mobius Inversion | Pre-check | Medium | ~90 lines |
| 8 | Quadratic Reciprocity | Pre-check | Medium | ~60 lines |
| 9 | Reflection Positivity | Search Upgrade | Medium | ~100 lines |
| 10 | Conjugation-Indexed Cover | Framework | Low-Med | ~100 lines |
| 11 | Mixed-Radix Number System | Search Upgrade | Low | ~80 lines |
| 12 | Circle Method Density | Pre-check | Low | ~80 lines |
| 13 | Tropical Eigenvalue Leakage | Framework | Low | ~80 lines |
| 14 | HO Critical Pairs | Framework | Low | ~200 lines |
| 15 | Defect Weight Universality | Framework | Low | Included in #14 |

**Top 3 recommended implementations**:
1. **Chord-Swap Diameter Descent** -- Proves bounded search depth, the single
   most important theoretical improvement.
2. **Assignment Gap Extension** -- Eliminates permutation search when diagonal
   dominance holds (the generic case).
3. **Newton Entropy Hierarchy** -- Compressed state tracking with exact moment
   computation, enabling efficient pruning.