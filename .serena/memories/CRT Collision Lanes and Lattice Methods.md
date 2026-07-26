# CRT Collision Lanes and Multi-Dimensional Lattice Methods for Integer Factoring

## State of the Art Summary (2024-2026)

### 1. CRT Collision Methods Extend to Multi-Dimensional Lattices

**Algebraic-group methods extend via:**
- **Algebraic tori**: Williams `p+1` generalizes to `T_k(F_p)` for degree-d extensions; exponent vectors form lattices over `Z^d`
- **Prime basis lattice embedding** (Schnorr 2021): Relation searching over smooth basis `{p_1,...,p_n}` embeds into lattices spanned by logarithm vectors `(ln p_1, ..., ln p_n, C ln N)^T`
- **Multivariate Coppersmith lattices**: CRT root-finding extends to multivariate polynomials `f(x_1,...,x_k)` with lattice shifts `x_1^{i_1}...x_k^{i_k} f^m N^t`

### 2. Collision Lane Theory

A **collision lane** represents a parallel evaluation path where integers factor over smooth basis `B`. Collision occurs when exponent vectors `e_i = (e_{i,1},...,e_{i,k})^T` combine to satisfy `sum c_i e_i ≡ 0 (mod 2)`, producing `x^2 ≡ y^2 (mod N)`.

**Batch GCD** (Bernstein-Heninger 2012): Product/remainder trees detect pairwise collisions in `O(M log² M)` time. Multi-dimensional generalizations search for multi-term combinations where no two relations collide independently.

**Inside-Out implementation**: `batch_crt_cascade.py` computes `C = ∏_{i<j} (U_M(P_i)·V_M(P_j) - U_M(P_j)·V_M(P_i)) mod N`. One gcd(C,N) detects ALL pairwise CRT collisions. This is O(K²) multiplications + O(1) GCD vs O(K²) pairwise GCDs.

### 3. Pairwise to Full Lattice Collision Detection

| Method | Complexity | Status |
|--------|-----------|--------|
| Pairwise GCD | O(K²) | `lucas_multi.py` |
| Batch CRT cross-product | O(K²) multiplications + O(1) GCD | `batch_crt_cascade.py` |
| Kannan's embedding (CVP→SVP) | Lattice dimension d | Not implemented |
| Full lattice reduction | O(d⁵ log³ B) for LLL | `lattice_factor.py` (simplified) |

### 4. Lattice Reduction for Factor Combinations

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| **LLL** (1982) | `O(d⁵ log³ B)` | Guarantees `||b₁|| ≤ 2^{(d-1)/4} (det L)^{1/d}` |
| **BKZ** | `2^{O(β log β)}` | Stronger reduction with block size β |
| **BKZ 2.0** | Exponential | Chen-Nguyen improvement |
| **SWIFT** (2024) | Heuristic | SAT + lattice hybrid |
| **Coppersmith** | Polynomial | Small root finding mod N |

**Critical**: Schnorr's 2021 claim of subexponential factoring via SVP was **rebutted** by Ducas & van de Pol -- lattice dimensions grow too rapidly for cryptographically significant moduli.

### 5. Sparse Matrix Complexity (QS/GNFS Linear Algebra)

Matrix `M ∈ F_2^{m×n}` with weight `w ≈ 50n–100n`:
- **Block Lanczos** (Montgomery 1995): `O(w·n/w_word)`, SIMD-friendly
- **Block Wiedemann** (Coppersmith 1994): Same asymptotic, highly distributed
- **CAIRN 2/3** (Izu et al. 2007): FPGA hardware acceleration

## Key Papers

| Paper | Contribution |
|-------|-------------|
| Coppersmith 1996 | Foundational LLL-based small root finding |
| Howgrave-Graham 1997 | Simplified Coppersmith theorem |
| Kannan 1987 | CVP→SVP embedding technique |
| Williams 1982 | `p+1` method in quadratic extensions |
| Lenstra 1987 | ECM (elliptic curve method) |
| Montgomery 1995 | Block Lanczos for sparse `GF(2)` |
| Coppersmith 1994 | Block Wiedemann parallel algorithm |
| Bernstein et al. 2012 | Batch GCD with product trees |
| Schnorr 2021 | SVP-based factoring claim (disproven) |
| Ducas & van de Pol 2021 | Rebuttal of Schnorr's claim |
| Ajani & Bright 2024 | SAT + lattice hybrid |

## Integration Points with Inside-Out/Berggren Structure

1. **Berggren-SL₂ matrix space**: The 2x2 Berggren matrices `U_MN`, `A_MN`, `D_MN` are parabolic elements of SL₂(Z). Their group structure `|SL₂(F_p)| = p(p-1)(p+1)` provides three independent smoothness targets in one group.

2. **Möbius cascade as structured descent**: The transformations `f_U(z) = 1/(2-z)`, `f_A(z) = 1/(2+z)`, `f_D(z) = z/(1+2z)` define deterministic descent paths through the Berggren tree, potentially providing structured relation generation for the lattice combiner.

3. **CF-convergent resonance**: CFRAC-style relations from `p_k² - Nq_k² = ±r_k` with small r_k. The CF period matrix from `cf_matrix_cascade.py` encodes the full period structure.

4. **Cyclotomic cascade**: Generalizes p-1 and p+1 to all cyclotomic orders -- each order n gives a different splitting behavior mod p depending on whether n | (p-1) or n | (p+1).

5. **Squaring conductance**: Stage 4 of Resonance Cascade Factoring analyzes `x → x²` on Z/NZ as a CRT bottleneck -- fundamentally a 1D collision detection problem.

## Key Gaps and Research Opportunities

1. **Multi-dimensional CRT collision lane**: The theory of CRT collision lanes (Cheon et al. 2013) -- using the full exponent lattice structure rather than just pairwise products -- is not implemented.

2. **Kannan's embedding**: Converting CVP to SVP via augmented lattice could extend the 5-way batched GCD to full lattice structure.

3. **Production-grade LLL/BKZ**: The custom LLL in `lattice_factor.py` and `coppersmith.py` is simplified. Integrating fpylll or G6K would enable BKZ-reduced bases.

4. **SAT + lattice hybrid**: Ajani & Bright 2024 combines SAT solving with lattice reduction for partial-information problems -- promising for structured inputs.

5. **Berggren-guided smooth relation generation**: Instead of random search, using the Berggren tree structure to guide which relations to seek next.

## Verified Claims

- Schnorr 2021 SVP factoring claim has been empirically disproven for large moduli (Ducas & van de Pol 2021)
- LLL-based methods are polynomial-time but not sub-exponential for general factoring
- GNFS remains the fastest asymptotic algorithm for cryptographically-sized inputs
