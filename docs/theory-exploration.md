# Theory Exploration: Lean Catalog Mathematical Ideas for Inside-Out Factoring

This document extracts mathematical ideas, theorems, and computational approaches from the Lean Catalog that could enhance integer factoring, particularly for the Inside-Out Factoring project (Pythagorean tree spectra, Berggren matrices, energy spectrum, continued fraction steering).

---

## 1. Pythagorean Energy Spectrum (PythagoreanEnergySpectrum.lean)

### Core Mathematical Idea
A strictly convex **energy functional** `E(N, s) = s^2 - N` whose integer spectrum detects and locates non-trivial factors of an odd number N via Fermat's difference-of-squares method. The key insight: the most balanced factorization has the least energy, so scanning s from ceil(sqrt(N)) upward reaches a factor deterministically.

### Key Theorems
- **composite_iff_diff_squares**: An odd N is composite iff there exists a non-trivial integer point on s^2 = N + t^2
- **factor_from_repr**: From such a point, s - t is a proper divisor of N
- **energy_strictConvexOn / energy_strictMonoOn**: The energy functional is strictly convex and monotone on [0, inf), guaranteeing deterministic descent
- **balanced_minimizes_energy**: More balanced factorizations have strictly smaller energy
- **leg_sq_factorization**: In any Pythagorean triple, a^2 = (c-b)(c+b), connecting triples to difference-of-squares
- **fermatSearch**: A provably correct and complete deterministic Fermat search algorithm

### Connection to Factoring
This is the direct theoretical backbone of the Inside-Out approach. The energy functional formalizes why scanning from sqrt(N) outward is optimal: the first perfect square hit yields the most balanced factor pair. The Berggren bridge (a^2 = (c-b)(c+b)) connects Pythagorean triples to the same difference-of-squares form.

### Implementation in Python
Already partially implemented as the energy spectrum in `insideout`. Key enhancement opportunity: **formally-guided search order**. The energy functional's strict convexity means we can prioritize search directions that decrease energy fastest. The `fermatSearch` algorithm provides a reference implementation:
```python
def fermat_search(N):
    for t in range(N + 1):
        m = N + t * t
        s = isqrt(m)
        if s * s == m and t < s and s - t > 1:
            return (s - t, s + t)
```

---

## 2. Berggren Tree Completeness (BerggrenCompleteness.lean)

### Core Mathematical Idea
Every primitive Pythagorean triple with c > 5 has a **unique parent** via one of three inverse Berggren matrices, and the parent's hypotenuse is strictly smaller. This gives well-founded descent: every primitive triple reaches (3,4,5) by a unique path. The Berggren tree is a 3-regular tree that exhausts all primitive triples.

### Key Theorems
- **descent_step**: Every primitive triple with c > 5 has a unique parent with strictly smaller hypotenuse
- **parent_unique**: At most one inverse branch produces a positive triple (the sigma invariants sigma1, sigma2 determine which)
- **universal_parent_hypotenuse**: All three inverses give the same parent hypotenuse c' = 3c - 2a - 2b
- **parentHyp_lt**: c' < c, enabling well-founded induction
- **parentHyp_decrease_bound**: c - c' >= 2, giving a quantitative descent rate
- **Berggren matrices have det = +/-1**, placing them in GL_3(Z), and they preserve the Lorentz form Q = diag(1,1,-1)

### Connection to Factoring
The unique descent path `PrimTriple -> List(Fin 3)` with O(log c) length is a **collision-resistant encoding**. The sigma invariants (sigma1 = a + 2b - 2c, sigma2 = 2a + b - 2c) classify which branch to take, providing an efficient navigation scheme. The Lorentz form preservation connects Pythagorean triples to hyperbolic geometry and special relativity.

### Implementation in Python
- **Berggren tree navigation**: Given a target N, search Pythagorean triples (a,b,c) where a^2 = (c-b)(c+b) gives a factor of a^2. Use the unique inverse path to navigate efficiently.
- **Sigma-invariant branch selection**: Compute sigma1 and sigma2 to determine which inverse to apply, avoiding exhaustive search of all three branches.
- **Hypotenuse descent as search bound**: Since c' = 3c - 2a - 2b and c - c' >= 2, the tree depth is O(c), providing bounds on search depth.

```python
def berggren_inverse(triple):
    a, b, c = triple
    s1 = a + 2*b - 2*c  # sigma1
    s2 = 2*a + b - 2*c   # sigma2
    if s1 > 0 and s2 < 0:
        return (s1, -s2, 3*c - 2*a - 2*b)  # A-inverse
    elif s1 > 0 and s2 > 0:
        return (s1, s2, 3*c - 2*a - 2*b)     # B-inverse
    elif s1 < 0 and s2 > 0:
        return (-s1, s2, 3*c - 2*a - 2*b)    # C-inverse
```

---

## 3. Gaussian Pythagorean (GaussianPythagorean.lean)

### Core Mathematical Idea
The behavior of a rational prime p in the Gaussian integers Z[i] is completely determined by p mod 4:
- p = 1 (mod 4): p splits as p = a^2 + b^2 = (a+bi)(a-bi) in Z[i]
- p = 3 (mod 4): p is inert (remains prime in Z[i])

A sum of two squares is never congruent to 3 modulo 4.

### Key Theorems
- **prime_one_mod_four_sum_two_squares**: Every prime p = 1 (mod 4) is a sum of two squares
- **prime_three_mod_four_not_sum_two_squares**: No prime p = 3 (mod 4) is a sum of two squares
- **gaussian_inert / gaussian_split**: Ring-theoretic forms in Z[i]

### Connection to Factoring
This is directly relevant to the Pythagorean parametrization: if N has a prime factor p = 1 (mod 4), then p = a^2 + b^2 for some a, b, and the Gaussian factorization p = (a+bi)(a-bi) provides a structural decomposition. For the Inside-Out approach:
- Primes p = 3 (mod 4) cannot appear as hypotenuses of primitive Pythagorean triples (they are "obstructions" in the Pythagorean tree)
- Primes p = 1 (mod 4) split in Z[i], and this splitting connects to the Berggren tree structure
- The sum-of-two-squares representation gives an alternative path to factoring: finding a, b with p = a^2 + b^2

### Implementation in Python
```python
def prime_mod4_behavior(p):
    """Classify a prime by its behavior in Z[i]."""
    if p % 4 == 1:
        return 'splits'  # p = a^2 + b^2 for some a, b
    elif p % 4 == 3:
        return 'inert'    # p remains prime in Z[i]
    elif p == 2:
        return 'ramifies'  # 2 = (1+i)(1-i)
```

---

## 4. Carmichael Composite (CarmichaelComposite.lean)

### Core Mathematical Idea
Carmichael's theorem: for composite n >= 14, the Fibonacci number F(n) has a primitive prime divisor (a prime dividing F(n) that does not divide F(k) for any k < n). This uses "entry point theory" -- the smallest k > 0 with p | F(k), denoted alpha(p).

### Key Theorems
- **fib_carmichael**: For n >= 13, F(n) has a primitive prime divisor
- **fibEntryPt_dvd_of_fib_dvd**: The entry point alpha(p) divides n whenever p | F(n)
- **primitive_of_fibCoprimePart_pos**: If the "coprime part" of F(n) relative to F(d) for all proper d | n is > 1, then F(n) has a primitive prime divisor
- **fib_coprime_part_pos_small**: Computational verification for 14 <= n <= 10000

### Connection to Factoring
Entry point theory provides a **Fibonacci-based primality/factoring test**: if n is composite, F(n) has a prime factor p whose entry point alpha(p) divides n. This connects to:
- **Lucas primality test**: A number n is prime iff for every prime q | n-1, there exists a with a^(n-1) = 1 and gcd(a^((n-1)/q) - 1, n) = 1
- **Williams p+1 factoring**: Uses Lucas sequences (closely related to Fibonacci) to find factors p where p+1 is smooth
- **Fibonacci factoring**: If we can find a primitive prime divisor of F(n), we find a factor of n

### Implementation in Python
```python
def fibonacci_entry_point(p):
    """Find the smallest k > 0 with p | F(k)."""
    a, b = 0, 1
    for k in range(1, p * 2):  # Entry point divides p-1 or p+1 or 2p
        a, b = b, (a + b) % p
        if a == 0:
            return k
    return 0

def fibonacci_factor(n):
    """Use Fibonacci entry points to find a factor of n."""
    g = gcd(fibonacci(n) % n, n)  # gcd(F(n) mod n, n)
    if 1 < g < n:
        return g
    return None
```

---

## 5. Prime Splitting (PrimeSplitting.lean)

### Core Mathematical Idea
Same as GaussianPythagorean (file #3) -- the splitting/inertness of rational primes in Z[i]. The key structural result: p = 1 (mod 4) splits as a sum of two squares, p = 3 (mod 4) is inert.

### Additional Connection to Factoring
The splitting criterion p = 1 (mod 4) is a necessary condition for a prime to appear as a hypotenuse of a primitive Pythagorean triple. This means:
- In the Berggren tree, only primes p = 1 (mod 4) can be hypotenuses
- For factoring N, if we can determine the mod-4 residue of N's prime factors, we can predict which branches of the Berggren tree are relevant
- The Cornacchia algorithm for finding a, b with p = a^2 + b^2 is directly applicable

---

## 6. Chordal Spill Bounds (ChordalSpillBounds.lean)

### Core Mathematical Idea
A theory of register allocation on interference graphs where "spilling" (moving variables to memory) is governed by **clique constraints**: in any clique K of size q with k registers, at least q - k vertices must be spilled. The spill-free allocations are exactly the k-colorings.

### Key Theorems
- **clique_spill_lower_bound**: Any valid k-register allocation on a clique of size q spills at least q - k vertices
- **global_spill_lower_bound**: Any valid allocation spills at least omega(G) - k vertices (omega = clique number)
- **completeGraph_spill_achievable / completeGraph_spill_optimal**: On a complete graph of q vertices, exactly q - k spills are necessary and sufficient
- **zero_spill_iff_colorable**: Zero spills iff the interference graph is k-colorable

### Connection to Factoring
While this file is about register allocation, the **clique structure** analysis transfers directly to factoring:
- **Clique-based bounds**: In the factorization graph where vertices represent candidate factors and edges represent conflicts, clique bounds give minimum search effort
- **Chordal graph structure**: If the search graph is chordal, efficient allocation algorithms exist
- **Pigeonhole in cliques**: The core argument (injective coloring on cliques) mirrors the pigeonhole arguments in the geometric cryptanalysis file

### Implementation in Python
The key transferable idea: **clique-based lower bounds on computational effort**. If we model factoring as searching through a structure with clique-like obstructions, we can derive minimum effort bounds.

---

## 7. Heegner-Rabinowitsch Bridge (HeegnerRabinowitschBridge.lean)

### Core Mathematical Idea
The Rabinowitsch phenomenon: for certain primes p (corresponding to Heegner discriminants 43, 67, 163), the polynomial f_p(n) = n^2 + n + p is prime for all 0 <= n <= p-2. Structural consequences:
- A Rabinowitsch prime p >= 3 forms a twin prime pair (p, p+2)
- The run values are strictly increasing and all lie below p^2
- A sharp run packs exactly p-1 distinct primes into [p, p^2)

### Key Theorems
- **rabinowitsch_prime_is_prime**: A Rabinowitsch prime is prime
- **rabinowitsch_gives_twin_prime**: A Rabinowitsch prime p >= 3 sits in a twin prime pair (p, p+2)
- **sharp_run_prime_packing**: p-1 distinct primes packed into [p, p^2)
- **eulerPoly_strictMono**: The Euler polynomial n^2 + n + p is strictly increasing

### Connection to Factoring
- **Prime-rich polynomials**: The polynomial n^2 + n + p generates dense clusters of primes, which could be used as "factoring probes" -- testing whether N shares factors with values of these polynomials
- **Heegner numbers and quadratic fields**: The discriminants -43, -67, -163 are the Heegner numbers (class number 1), connecting to the theory of imaginary quadratic fields
- **Twin prime connection**: If N has a factor p near a Heegner-related prime, the twin prime pair constrains p's possible values

### Implementation in Python
```python
def rabinowitsch_probes(N, p):
    """Use the polynomial n^2 + n + p to probe N for factors."""
    probes = []
    for n in range(p - 1):
        val = n * n + n + p
        g = gcd(val, N)
        if 1 < g < N:
            probes.append((n, val, g))
    return probes

# Heegner-related primes: p = 11 (d=-43), p = 17 (d=-67), p = 41 (d=-163)
heegner_probes = [11, 17, 41]
```

---

## 8. Selberg Sieve Weight (SelbergSieveWeight.lean)

### Core Mathematical Idea
The Selberg sieve weight identity: for every positive integer n,
```
mu^2(n) = sum_{d^2 | n} mu(d)
```
where mu is the Mobius function. This is proven via the "square-root part" sqrtPart(n), the largest m with m^2 | n, which satisfies:
- d^2 | n iff d | sqrtPart(n)
- n is squarefree iff sqrtPart(n) = 1

### Key Theorems
- **selberg_sieve_weight**: mu(n)^2 = sum_{d^2 | n} mu(d)
- **dvd_sq_iff**: d^2 | n iff d | sqrtPart(n)
- **squarefree_iff_sqrtPart**: n is squarefree iff sqrtPart(n) = 1

### Connection to Factoring
The Selberg sieve is a fundamental tool in analytic number theory for bounding the size of sifted sets. For factoring:
- **Squarefree detection**: If sqrtPart(N) > 1, then N has a square factor, which immediately gives a factor
- **Mobius-weighted sums**: The identity mu^2(n) = sum mu(d) over d^2 | n provides a way to detect squarefree numbers, relevant for distinguishing prime from composite
- **Sieve acceleration**: The Selberg sieve can accelerate the search for numbers coprime to N (candidates for factors) by efficiently eliminating multiples

### Implementation in Python
```python
def sqrt_part(n):
    """Largest m such that m^2 divides n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        count = 0
        while temp % d == 0:
            temp //= d
            count += 1
        result *= d ** (count // 2)
        d += 1
    if temp > 1:
        result *= temp  # prime factor with exponent 1 (odd, so floor(1/2)=0)
    return result

def is_squarefree(n):
    """Check if n is squarefree using sqrtPart."""
    return sqrt_part(n) == 1
```

---

## 9. GL(1) Correspondence (GL1Correspondence.lean)

### Core Mathematical Idea
The GL(1) Langlands correspondence over Q: Dirichlet characters mod n are isomorphic (as a group) to 1-dimensional complex representations of Gal(Q(zeta_n)/Q). The bridge is Artin reciprocity: Gal(Q(zeta_n)/Q) ~= (Z/nZ)^*.

### Key Theorems
- **langlandsGL1**: Group isomorphism DirichletCharacter(C, n) ~=* (Gal(Q(zeta_n)/Q) ->* C^*)
- **card_galois_reps_eq_totient**: # of 1-dim Galois reps = phi(n)
- **card_galois_reps_prime**: For prime p, there are p-1 such reps

### Connection to Factoring
- **Character-based factoring**: Dirichlet characters mod n provide systematic ways to probe N for factors. Each character chi gives a multiplicative function chi(a) that can be used to construct sieves.
- **Galois structure**: The Galois group Gal(Q(zeta_p)/Q) ~ (Z/pZ)^* has order p-1. If N has a prime factor p, then the structure of (Z/pZ)^* constrains the possible residues.
- **Totient-based approach**: phi(n) is computable from the factorization of n, and conversely, knowing phi(n) and n determines the factorization (via the quadratic formula, as in the Wiener attack)

### Implementation in Python
```python
def dirichlet_characters(n):
    """Generate Dirichlet characters mod n (for factoring probes)."""
    from math import gcd
    chars = []
    for a in range(1, n):
        if gcd(a, n) == 1:
            # Character chi_a sending 1 -> exp(2*pi*i*a/n)
            # This is a multiplicative character
            chars.append(a)
    return chars  # These are (Z/nZ)^* elements, index by phi(n) characters
```

---

## 10. Structural Bounds (StructuralBounds.lean)

### Core Mathematical Idea
Quantum Latin squares of order 6 have structural bounds on their "ray cardinality":
- A commutative pair construction from 6 labels produces at most 21 distinct outputs (the triangular number 6*7/2)
- Direct sum constructions add cardinalities when ray sets are disjoint (19 + 4 = 23)
- Orthonormality forces row/column entries to be distinct rays

### Key Theorems
- **commutative_pair_cardinality_le_twenty_one**: Symmetric Schur-product ceiling of 21
- **directSum_image_cardinality**: Cardinalities add for disjoint populations
- **row_ray_inequivalent / column_ray_inequivalent**: Distinct entries give distinct rays

### Connection to Factoring
While quantum Latin squares seem distant from factoring, the structural principles transfer:
- **Commutative ceiling**: Symmetric operations on n objects generate at most n(n+1)/2 distinct outputs -- this bounds the number of distinct "spectrum lines" from n Pythagorean generators
- **Direct sum decomposition**: If N = pq with p, q having different residue structures, the factoring approach can decompose into independent subproblems whose complexities add
- **Orthonormality constraint**: In the energy spectrum, different factor representations give distinct "energy levels" (analogous to distinct rays)

---

## 11. Effective Bounds (EffectiveBounds.lean)

### Core Mathematical Idea
Effective approximation bounds for the Euler-Mascheroni constant gamma:
- The sequences H_n - log(n+1) < gamma < H_n - log(n) bracket gamma
- Both one-sided errors are strictly less than log(1 + 1/n) ~ 1/n
- These bounds are effective but too weak for irrationality proofs (rate ~1/n vs geometric rates needed)

### Key Theorems
- **eulerMascheroni_trap_width_eq**: seq'(n) - seq(n) = log(n+1) - log(n)
- **eulerMascheroniConstant_sub_seq_lt**: gamma - seq(n) < log(1 + 1/n)
- **seq'_sub_eulerMascheroniConstant_lt**: seq'(n) - gamma < log(1 + 1/n)

### Connection to Factoring
- **Logarithmic bounds**: The effective bound log(1 + 1/n) ~ 1/n gives a rate estimate for how quickly the harmonic numbers approximate gamma. This is relevant for estimating the "balance" of factorizations: if N = de, then log(N) = log(d) + log(e) and the imbalance is |log(d/e)|.
- **Diophantine approximation**: The quality of rational approximations to gamma is analogous to the quality of rational approximations to sqrt(N) in the Fermat method. The bound ~1/n shows why brute-force approaches are slow -- you need O(sqrt(N)) trials.
- **Speed comparison**: Geometric convergence (as in continued fractions) is fundamentally faster than the ~1/n rate of the harmonic sequence, justifying the use of CF-based methods in the Inside-Out approach.

---

## 12. Wiener Factorization (WienerFactorization.lean)

### Core Mathematical Idea
The complete end-to-end Wiener attack on RSA: from partial knowledge of p+q (via continued fraction convergents of e/n), recover the private exponent d, then factor n via the quadratic formula. The key identity is the **perfect-square discriminant**: (p+q)^2 - 4*p*q = (p-q)^2.

### Key Theorems
- **discriminant_eq**: (p+q)^2 - 4*p*q = (p-q)^2 (the discriminant is always a perfect square)
- **factor_from_sum_prod**: The quadratic formula recovers p, q from their sum S = p+q and product N = p*q
- **factor_n_from_totient**: n and phi(n) determine p+q = n - phi(n) + 1, hence the factorization
- **modified_wiener_end_to_end**: Under the modified-Wiener hypotheses, recovering d is equivalent to factoring n

### Connection to Factoring
This is the most directly applicable result for the Inside-Out project:
- **CF convergents and factoring**: The continued fraction expansion of e/n provides convergents that may reveal d/p+q. This directly connects to "continued fraction steering" in the Inside-Out approach.
- **Perfect-square discriminant**: The fact that (p+q)^2 - 4n = (p-q)^2 is a perfect square means that once we know p+q (even approximately), we can factor n exactly.
- **Totient recovery**: If phi(n) can be determined (even approximately), the factorization follows.

### Implementation in Python
```python
def factor_from_sum_product(S, N):
    """Factor N given S = p+q and N = p*q, using the quadratic formula."""
    from math import isqrt
    disc = S * S - 4 * N
    sqrt_disc = isqrt(disc)
    if sqrt_disc * sqrt_disc == disc:
        p = (S + sqrt_disc) // 2
        q = (S - sqrt_disc) // 2
        return p, q
    return None

def factor_from_totient(n, phi_n):
    """Factor n given phi(n)."""
    S = n - phi_n + 1  # S = p + q
    return factor_from_sum_product(S, n)
```

---

## 13. Geometric Cryptanalysis (GeometricCryptanalysis.lean)

### Core Mathematical Idea
Bounded-box collision theorem: if (2B+1)^n > q, then two distinct vectors x, y in the box {|x_i| <= B} collide under the modular linear form x -> sum(a_i * x_i) mod q. The difference z = x - y gives a nonzero short vector in the kernel lattice with |z_i| <= 2B.

### Key Theorems
- **bounded_box_mod_collision**: Pigeonhole principle for modular linear forms -- if the box has more vectors than the modulus, a collision is guaranteed
- **bounded_box_collision_yields_short_kernel_vector**: The collision yields a short kernel vector (shortest integer solution to the modular equation)
- **bounded_box_sis_witness**: Matrix generalization -- for A in Z^{m x n}, if (2B+1)^n > q^m, there exists a nonzero bounded vector in the kernel

### Connection to Factoring
This is a **lattice-based factoring approach**:
- **Birthday-style attacks**: If we can enumerate enough "bounded candidates" relative to the modulus N, a collision gives a factor
- **SIS connection**: The Short Integer Solution problem is the foundation of lattice-based cryptography. Conversely, finding short vectors in the kernel lattice of a matrix modulo N can reveal factors.
- **Berggren lattice**: The Berggren matrices form a subgroup of GL_3(Z), and the Pythagorean triples are integer points in a cone. The bounded-box collision theorem can be applied to find collisions among Pythagorean parametrizations modulo N.

### Implementation in Python
```python
def bounded_box_collision(N, B, n):
    """Search for bounded-box collisions modulo N."""
    from itertools import product
    seen = {}
    for x in product(range(-B, B+1), repeat=n):
        val = sum(a * xi for a, xi in zip(coefficients, x)) % N
        if val in seen:
            y = seen[val]
            if x != y:
                z = tuple(xi - yi for xi, yi in zip(x, y))
                return z  # Short kernel vector
        else:
            seen[val] = x
    return None
```

---

## 14. Ramanujan Rho Factorization (RamanujanRhoFactorization.lean)

### Core Mathematical Idea
The denominator of Ramanujan's mock theta function rho(q) has a **telescoping product factorization**:
- Each factor is 1 + q^{2k+1} + q^{4k+2}, a "cyclotomic-type trinomial"
- The identity (1-Y)(1+Y+Y^2) = 1-Y^3 with Y = q^{2k+1} gives the single-factor cube identity
- The full denominator factors as Prod(1 - q^{6k+3}) / Prod(1 - q^{2k+1})

### Key Theorems
- **factor_cube_identity**: (1 - X^{2k+1})(1 + X^{2k+1} + X^{4k+2}) = 1 - X^{6k+3}
- **denominator_factorization**: Telescoping product factorization of the full denominator

### Connection to Factoring
- **Cyclotomic structure**: The factorization reveals that the denominator has a mod-3 residue structure tied to cyclotomic polynomials. The factors 1 + Y + Y^2 = (Y^3 - 1)/(Y - 1) are the 3rd cyclotomic polynomial evaluated at Y.
- **Telescoping products**: The technique of multiplying single-factor identities across a range to get telescoping cancellations can be applied to construct special polynomials whose GCD with N reveals factors.
- **Power residue testing**: Evaluating these products at specific roots of unity modulo N can reveal information about N's factor structure.

### Implementation in Python
```python
def ramanujan_rho_probes(N, m):
    """Use Ramanujan rho denominator factors to probe N."""
    from math import gcd
    probes = []
    for k in range(m):
        # Factor: 1 + N^{2k+1} + N^{4k+2} mod N
        # This is (N^{6k+3} - 1) / (N^{2k+1} - 1) when N^{2k+1} != 1
        exponent = 2 * k + 1
        val = (1 + pow(N, exponent, N) + pow(N, 2 * exponent, N)) % N
        g = gcd(val, N)
        if 1 < g < N:
            probes.append((k, g))
    return probes
```

---

## 15. Counterfactual Prime Factorization (CounterfactualPrimeFactorization.lean)

### Core Mathematical Idea
Unique factorization fails in the "Hilbert multiplicative universe" of numbers congruent to 1 (mod 4):
- 9 = 9, 21 = 21, 49 = 49 are all "Hilbert primes" (irreducible in the submonoid)
- But 441 = 9 * 49 = 21 * 21, giving two distinct factorizations into Hilbert primes
- Every ordinary prime p = 1 (mod 4) remains a Hilbert prime
- There are infinitely many Hilbert primes

### Key Theorems
- **hilbertPrime_nine / hilbertPrime_twentyOne / hilbertPrime_fortyNine**: 9, 21, 49 are Hilbert primes
- **unique_factorization_collapses**: 441 = 9*49 = 21*21 with distinct multisets of prime factors
- **hilbertPrime_of_prime_mod_four**: Ordinary primes p = 1 (mod 4) remain Hilbert primes
- **infinitely_many_hilbertPrimes**: There are infinitely many Hilbert primes

### Connection to Factoring
- **Non-unique factorization and factorization**: The failure of unique factorization in Z[1 mod 4] shows that factoring in restricted multiplicative structures can be fundamentally different from factoring in Z. This is analogous to how factoring in Z[sqrt(-5)] fails to be unique.
- **Modular arithmetic and factoring**: If we restrict to numbers = 1 (mod 4), the "factoring problem" changes character -- some composites become "prime" (like 9, 21, 49), while unique factorization breaks down.
- **Connection to Gaussian integers**: The Hilbert universe H = {n : n = 1 (mod 4)} is the norm form of the Gaussian integers restricted to a + bi with both a, b odd. The failure of unique factorization in H mirrors the non-trivial class group behavior.

### Implementation in Python
```python
def hilbert_prime(n):
    """Check if n is a Hilbert prime (irreducible in {1 mod 4})."""
    if n % 4 != 1 or n < 5:
        return False
    for d in range(5, isqrt(n) + 1, 4):  # only check 1 mod 4 divisors
        if n % d == 0:
            return False
    return True

# Key examples: 9, 21, 49 are Hilbert primes
# 441 = 9*49 = 21*21 (non-unique factorization)
```

---

## Cross-Cutting Connections and Implementation Priorities

### Highest Priority for Inside-Out Factoring

1. **Energy Spectrum (File 1)**: Already the backbone of the approach. The formal verification of `fermatSearch` correctness and completeness, plus the convexity/descent guarantees, should be incorporated into the Python implementation for correctness assurance.

2. **Berggren Tree (File 2)**: The unique descent path via sigma invariants provides an O(log c) navigation scheme for the Pythagorean tree. This should be integrated into the tree search component of Inside-Out.

3. **Wiener Factorization (File 12)**: The perfect-square discriminant identity and CF-based recovery of p+q from e/n directly supports continued fraction steering. The `factor_from_totient` function should be a key component.

4. **Gaussian Pythagorean / Prime Splitting (Files 3, 5)**: The mod-4 classification of primes determines which branches of the Berggren tree are relevant for a given N. This should guide the search strategy.

### Medium Priority

5. **Geometric Cryptanalysis (File 13)**: The bounded-box collision theorem provides a theoretical foundation for birthday-style attacks and SIS-based factoring. Could enhance the "collision search" component.

6. **Carmichael / Entry Points (File 4)**: Fibonacci entry point theory provides an alternative factoring mechanism (Williams p+1 method). Could be implemented as a fallback method.

7. **Selberg Sieve (File 8)**: The squarefree detection via sqrtPart and the Mobius sieve identity provide tools for filtering candidates in the factoring search.

8. **Heegner-Rabinowitsch (File 7)**: Prime-rich polynomials n^2 + n + p can be used as probes, especially for N with factors near Heegner-related primes (11, 17, 41).

### Lower Priority (Theoretical Interest)

9. **Ramanujan Rho (File 14)**: The telescoping product factorization reveals cyclotomic structure that could inform polynomial-based factoring, but the connection is indirect.

10. **Counterfactual Primes (File 15)**: Illustrates how factoring changes in restricted multiplicative structures. More of theoretical interest than practical implementation.

11. **GL(1) Correspondence (File 9)**: The Dirichlet character structure is relevant for understanding the arithmetic of Z/nZ but doesn't directly yield factoring algorithms.

12. **Structural Bounds (File 10)**: Quantum Latin square bounds have an analogy to energy spectrum bounds but are not directly applicable.

13. **Effective Bounds (File 11)**: The Euler-Mascheroni bounds illustrate why brute-force methods are slow (~1/n rate), justifying the use of faster methods, but don't provide new algorithms.

14. **Chordal Spill (File 6)**: Register allocation bounds transfer conceptually (clique-based effort bounds) but are not directly applicable to factoring.