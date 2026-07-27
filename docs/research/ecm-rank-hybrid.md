# ECM × Rank-of-Apparition Hybrid Factoring

## Key Insight: Theoretical Unification

All "smooth order" factoring methods are instances of the same framework:

| Method | Algebraic Group | Group Order | Success Condition |
|--------|----------------|-------------|-------------------|
| Pollard p−1 | (ℤ/pℤ)* multiplicative | p−1 = Φ₁(p) | p−1 is B-smooth |
| Williams p+1 | Lucas sequence ring ℤ[√D] | p−(D/p) ∈ {p−1, p+1} | p+1 is B-smooth |
| **Rank-of-apparition** | Lucas/Fibonacci/Mersenne recurrence | rank r \| p−(D/p) | rank r is B-smooth |
| **ECM** | Elliptic curve E(ℤ/pℤ) | #E ∈ [p+1−2√p, p+1+2√p] | #E is B-smooth |
| Cyclotomic (Bach-Shalit) | k-th cyclotomic extension | Φ_k(p) | Φ_k(p) is B-smooth |

**Both rank-of-apparition and ECM succeed when the order of an element in the appropriate group is smooth.**

## Can Rank Guide ECM?

**Yes, indirectly.** If a rank worker finds that gcd(U_ℓ^k, n) = 1 for some prime ℓ, we learn that ℓ does NOT divide the rank. This lets us:
- **Skip curves** whose order would require ℓ-divisibility
- **Prioritize curves** whose order is divisible only by primes consistent with the rank structure

## Combined Algorithm: "Smooth-Order Portfolio with Cross-Pruning"

```
Phase 1: Cheap probes (simultaneous)
  - Trial division, GCD with U_M across multiple (P,Q) pairs
  - GCD with Fibonacci/Lucas/Mersenne sequences

Phase 2: Rank-ECM crossover
  - Extract "smooth part" of rank from partial results
  - Construct ECM curves with torsion structure matching the smooth part

Phase 3: Adaptive ECM with rank guidance
  - Prioritize curves whose order's prime factorization overlaps with rank smooth part
  - Cross-prune: if rank fails for prime ℓ, skip curves requiring ℓ | #E

Phase 4: Two-stage cascade with shared batch GCD
```

## Probability Analysis

For B₁ = 10⁴ and a 64-bit factor p:
- P(p-1 smooth) ≈ 0.004
- P(rank smooth) ≈ 0.004 (but covers p+1 case)
- P(ECM success with 100 curves) ≈ 0.04

**Running all in parallel**: P(success) ≈ 1 − (0.996)(0.996)(0.96) ≈ 0.08 per batch

## Recommendations

1. **Shared factor base**: Both rank workers and ECM workers use the same prime bound B₁
2. **Cross-pruning**: When rank workers fail for prime ℓ, mark ℓ as "unlikely" for ECM
3. **Suyama + Lucas hybrid**: Seed ECM curve parameters from Lucas sequence values
4. **Batch GCD**: All workers accumulate into a shared batch GCD structure

## Theoretical Open Problem

Is there a single function giving the probability that at least one of {p−1, p+1, #E₁, #E₂, ...} is smooth? This requires understanding correlations between these quantities — currently unknown.
