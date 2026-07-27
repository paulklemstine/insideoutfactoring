# Extending Rank-of-Apparition Methods to Balanced Semiprimes
## Research Summary — 2026-07-26

### The Core Problem

For N = pq with p ≈ q (balanced semiprime, RSA-style), standard rank-of-apparition methods face a structural obstacle:

- **Rank size scales with factor size**: For any strong-divisibility sequence u_k (Fibonacci, Lucas, Mersenne), the rank of apparition α(p) = min{k : p | u_k} satisfies α(p) | p ± O(1). When p ≈ q, both α(p) and α(q) are ~max(p,q).
- **Smoothness correlation**: If α(p) is B-smooth, then α(q) is *also* likely B-smooth because both divide numbers of similar magnitude. The gcd(u_M, N) then reveals *both* factors simultaneously → gcd = N, no split.
- **The asymmetry requirement**: Standard methods need α(p) smooth AND α(q) non-smooth. For balanced semiprimes, this asymmetry is rare.

---

### RQ1: Multiple Recurrence Families — Simultaneous Probing

**Finding**: Multiple families help, but only if they target *structurally independent* divisors.

The codebase already implements multi-family probing (`rank_apparition.py`: 7 families; `lucas_multi.py`: 16+ parameters; `batch_crt_cascade.py`: 32 parameters). The key insight:

For Lucas sequences U_k(P, Q=1) with discriminant D = P² − 4:
- If (D/p) = +1 (QR): α_P(p) | (p − 1)
- If (D/p) = −1 (NR): α_P(p) | (p + 1)
- If D ≡ 0 (mod p): α_P(p) | p

**Critical for balanced semiprimes**: By choosing P values where the Legendre symbol (D/p) ≠ (D/q), we force α_P(p) | (p−1) and α_P(q) | (q+1) [or vice versa]. Since p−1 and q+1 are *independent* random numbers, the smoothness events decorrelate.

**Recommendation**: The `batch_crt_cascade.py` already covers Legendre symbol classes (P = 1, 2, 3, 5, 7, ...). This is the right approach. Extend to P values specifically chosen so P² − 4 spans QR/NR classes systematically. The probability that a random P gives (D/p) ≠ (D/q) is 1/2, so ~50% of parameters target independent smoothness conditions.

**Probability estimate**: For 64-bit balanced semiprime (p, q ≈ 2³²), P(p−1 is 10⁴-smooth) ≈ ρ(32·ln2/ln(10⁴)) ≈ ρ(2.85) ≈ 0.04. With independent targets (p−1 vs q+1), P(exactly one smooth) ≈ 2·0.04·0.96 ≈ 7.7% per parameter. With 16 parameters: P(at least one hit) ≈ 1 − (0.923)¹⁶ ≈ 72%.

---

### RQ2: Combining Ranks via CRT Collision — The Strongest Lever

**Finding**: The CRT collision lane (already in `lucas_multi.py` and `batch_crt_cascade.py`) is the most promising direction for balanced semiprimes.

**Mechanism**: For two Lucas parameters P₁, P₂ with states (U_M(P₁), V_M(P₁)) and (U_M(P₂), V_M(P₂)):
```
cross = U_M(P₁)·V_M(P₂) − U_M(P₂)·V_M(P₁) (mod N)
g = gcd(cross, N)
```
This vanishes mod p iff the projective states (U:V) are proportional mod p. For random P₁, P₂, the states mod p and mod q are essentially independent random points in P¹(F_p) and P¹(F_q).

**Why this helps for balanced semiprimes**: The cross-product detects a factor when states coincide mod p but NOT mod q (or vice versa). This requires NO smoothness at all — only that the two sequences land in different projective positions mod p vs mod q. The probability of collision per pair is ~1/p + 1/q ≈ 2/p for balanced case, but with K parameters we have K(K−1)/2 pairs, giving effective probability ~K²/p.

**The batch CRT cascade** (`batch_crt_cascade.py`) computes the product of all cross-terms with O(K²) multiplications + O(1) GCD, which is already optimal for this approach.

**Limitation honestly stated**: This is still exponentially unlikely for cryptographic sizes. For 64-bit, K=32 gives ~1024 pairs, each with ~2/2³² chance → ~5×10⁻⁷ per M value. Need ~2M different M values — infeasible.

**Recommendation**: The CRT collision lane should be viewed as a *complement* to smooth-rank probing, not a replacement. Use it within the adaptive portfolio for small-to-medium semiprimes (≤80 bits) where K²/p is non-negligible.

---

### RQ3: CRT Structure of Ranks — Reconstructing Factors from Rank Information

**Finding**: If we could determine α(p) and α(q) independently, we could reconstruct p and q via CRT. But this reduces to the hardness of the problem itself.

**Mathematical structure**: For Lucas U_k(P, Q=1):
- α(p) | p − (D/p), so p ≡ (D/p) (mod α(p))
- If we know α(p) = r, then p ≡ ε (mod r) where ε ∈ {+1, −1, 0}

**The approach**: Compute gcd(u_M, N) for many M. When gcd ≠ 1, N, we've found a factor. When gcd = N, both α(p)|M and α(q)|M. The *smallest* M where gcd(u_M, N) ≠ 1 gives lcm(α(p), α(q)) information.

**Novel idea — rank GCD lattice**: Compute u_M for M in a smoothness lattice. Record which M values give gcd = 1, gcd = p, gcd = q, gcd = N. The set of M with gcd = N reveals lcm(α(p), α(q)). The set with gcd = p reveals α(p). If we can separate these sets, we extract both ranks.

**Practical barrier**: For balanced semiprimes, the first non-trivial gcd is almost always N (both ranks smooth simultaneously), so we can't separate α(p) from α(q).

**Recommendation**: This direction is unlikely to yield a polynomial-time method. The information-theoretic barrier is fundamental: the sequence mod N is a single value encoding both mod-p and mod-q information, and without a split, we cannot disentangle them.

---

### RQ4: Recurrence Families with Different Rank Structure

**Finding**: The discriminant D = P² − 4Q is the key structural parameter. Three families with fundamentally different rank behavior:

1. **Mersenne-type (a^k − 1)**: rank = ord_p(a) | p − 1. Always divides p−1.
2. **Lucas QR (D = QR mod p)**: rank | p − 1. Same as Mersenne but different divisor structure.
3. **Lucas NR (D = NR mod p)**: rank | p + 1. Targets p+1 instead of p−1.
4. **Cyclotomic Φ_m**: rank | p^gcd(m,·) − 1. Targets p^k − 1 for various k.

**The cyclotomic resultant cascade** (`cyclotomic_resultant.py`) is the most general: it simultaneously targets p−1 (m=1), p+1 (m=2), p²+p+1 (m=3), p²+1 (m=4), p²−p+1 (m=6). For balanced semiprimes, each m gives an independent smoothness target.

**Novel recommendation — order-mixing cascade**: For each cyclotomic order m, the rank divides p^k − 1 where k | φ(m). The key insight: p² + 1 (from m=4) is NOT correlated with p − 1 or p + 1. A factor p where p−1 is non-smooth but p²+1 is smooth would be missed by p−1/p+1 methods but caught by m=4.

**Probability**: For random p, P(p²+1 is B-smooth) is much smaller than P(p−1 is B-smooth) since p²+1 ≈ p² is larger. But for the cases where p−1 and q−1 are both non-smooth (the hard balanced case), higher cyclotomic orders provide the only rank-based avenue.

---

### Specific Recommendations for Handling Balanced Semiprimes

#### 1. **QR/NR Legendre Symbol Stratification** (implement next)
Choose Lucas parameters P in pairs (P_QR, P_NR) where P_QR² − 4 is QR mod p and P_NR² − 4 is NR mod q. This forces α_QR(p) | p−1 and α_NR(q) | q+1, decorrelating the smoothness condition. Already partially done in `batch_crt_cascade.py` but should be made explicit and adaptive.

#### 2. **Adaptive Bound Escalation Based on Bit-Balance**
For balanced semiprimes (bit-length ratio p/q close to 1), escalate smoothness bounds faster since both factors need the same bound. The adaptive portfolio (`adaptive_portfolio.py`) should detect balance (via Fermat proximity: if (p+q)/2 − √N is small) and allocate more budget to cyclotomic/higher-order methods.

#### 3. **ECRT (Elliptic Curve) Rank Generalization**
The natural extension: replace "rank divides p±1" with "group order divides #E(F_p) = p + 1 − a_p" where a_p ∈ [−2√p, 2√p]. ECM works for balanced semiprimes because it tries many curves with different a_p values, effectively randomizing the group order. The rank-of-apparition analogue: try many Lucas/Cyclotomic parameters to randomize the rank. **This is exactly what the codebase already does** — the multi-parameter approach is the Lucas-analogue of ECM's curve randomization.

#### 4. **Hybrid: Rank Probing + Fermat Proximity**
For balanced semiprimes, p + q is close to 2√N. Combine:
- Rank methods to find small factors or near-misses
- Fermat's method (already in `brahmagupta.py`) which is fast when p ≈ q
- The adaptive portfolio should run Fermat first for balanced inputs, then escalate to rank methods only if Fermat times out

#### 5. **Higher Cyclotomic Orders for the Hard Balanced Case**
When both p−1 and q−1 are non-smooth (the genuinely hard case), higher cyclotomic orders (m = 5, 7, 8, 10, 12) target p^φ(m) − 1 structures. The `cyclotomic_resultant_factor` should be extended to m up to 20+ with adaptive order selection.

---

### Honest Assessment

**The fundamental barrier remains**: Rank-of-apparition methods are smooth-order methods. For balanced semiprimes where both p−1 and q−1 are non-smooth, no rank-based method can succeed without:
1. Randomizing the rank target (many parameters/curves — already implemented)
2. Moving to higher-order targets (cyclotomic — partially implemented)
3. Using non-smooth-order methods (CF, NFS, lattice — available in portfolio)

**The best near-term improvement**: The adaptive portfolio (`adaptive_portfolio.py`) with Thompson sampling already allocates budget to methods that work. For balanced semiprimes, it should learn to deprioritize pure rank methods and favor cyclotomic, SL₂, and CF-based methods that have different structural assumptions.

**The research frontier**: A truly novel approach would exploit the *relationship* between rank mod p and rank mod q for the SAME sequence. If α(p) and α(q) share a large gcd (which happens when p ≡ q (mod many small primes)), then gcd(u_M, N) = N for many M, but the *structure* of the failure set reveals information about p − q. This "failure set cryptanalysis" direction is unexplored.

---

### Key Files for Implementation
- `insideout/rank_apparition.py` — base rank method (153 lines)
- `insideout/lucas_multi.py` — multi-parameter + CRT collision (519 lines)
- `insideout/batch_crt_cascade.py` — batch CRT, Legendre symbol coverage (299 lines)
- `insideout/cyclotomic_resultant.py` — cyclotomic orders m=1,2,3,4,6,10,12 (752 lines)
- `insideout/sl2_group_order.py` — SL₂ group-order cascade, L_p[1/2] (133 lines)
- `insideout/adaptive_portfolio.py` — Thompson-sampled budget allocation
