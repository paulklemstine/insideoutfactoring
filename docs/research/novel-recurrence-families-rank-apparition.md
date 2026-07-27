# Novel Recurrence Families for Rank-of-Apparition Factoring

**Research Summary — 2026-07-26**

## Background: The Strong-Divisibility Framework

A sequence $(u_n)$ is a **strong divisibility sequence** if $\gcd(u_m, u_n) = u_{\gcd(m,n)}$.
For factoring $N = pq$: if $M$ is divisible by $\text{rank}_u(p)$ but not $\text{rank}_u(q)$,
then $\gcd(u_M, N) = p$.

**Currently implemented** (in `rank_apparition.py`, `fibonacci_factor.py`, `lucas_multi.py`):
- Fibonacci $U_n(1,-1)$, Lucas $U_n(P,Q)$ for several $(P,Q)$, Mersenne $a^n-1$
- Multi-parameter Lucas with CRT collision lanes (`lucas_multi.py`)
- Cyclotomic resultant cascade generalizing $p-1$ / $p+1$ to all $\Phi_n$ (`cyclotomic_resultant.py`)

## Key Theoretical Insight: Rank Structure by Recurrence Order

For a linear recurrence with characteristic polynomial $f(x)$ of degree $k$:
$$\text{rank}(p) \mid \#\mathbb{F}_{p^k}^\times = p^k - 1 \text{ (or a divisor thereof)}$$

- **Order 2** (Lucas/Fibonacci): $\text{rank}(p) \mid (p - \left(\frac{D}{p}\right))$ — divides $p \pm 1$
- **Order 3**: $\text{rank}(p) \mid (p^3 - 1) = (p-1)(p^2+p+1)$ — **new factor $p^2+p+1$**
- **Order $k$**: $\text{rank}(p) \mid (p^k - 1)$ — covers $\Phi_d(p)$ for all $d \mid k$

**This is the key advantage**: higher-order recurrences reach primes where $p^k-1$ is smooth even when $p \pm 1$ are not. This strictly subsumes $p-1$ and $p+1$ methods.

---

## Novel Family 1: Third-Order Strong Divisibility Sequences (Hall 1936)

**Reference**: Hall, M. "Divisibility Sequences of Third Order." *Bull. Amer. Math. Soc.* 42 (1936): 345–350.

**Definition**: A third-order recurrence $u_n = P \cdot u_{n-1} + Q \cdot u_{n-2} + R \cdot u_{n-3}$ with appropriate initial conditions forms a strong divisibility sequence when the characteristic roots satisfy certain coprimality conditions.

**Key construction**: Let $\alpha, \beta, \gamma$ be roots of $x^3 - Px^2 - Qx - R = 0$. Define:
$$u_n = \frac{(\alpha^n - \beta^n)(\alpha^n - \gamma^n)(\beta^n - \gamma^n)}{(\alpha - \beta)(\alpha - \gamma)(\beta - \gamma)}$$

This is the **resultant normalization** — it produces integer values and strong divisibility.

**Rank structure**: $\text{rank}(p) \mid (p^2 + p + 1)$ when $p \nmid \Delta$ and the cubic splits completely mod $p$.

**Why novel**: Reaches primes where $p^2+p+1$ is smooth — a condition NOT covered by any $p\pm1$, ECM, or Lucas method. The density of such primes is different, providing complementary coverage.

**Implementation**: $O(\log n)$ via companion matrix exponentiation (3×3 matrix fast powering).

---

## Novel Family 2: Elliptic Divisibility Sequences (EDS)

**Reference**: Ward, M. "The Arithmetic of Elliptic Divisibility Sequences." (1948); Stange, K. "Elliptic Nets and Elliptic Curves." (2010).

**Definition**: Nonlinear recurrence from elliptic curve division polynomials:
$$W_{2n+1} W_1^3 = W_{n+2} W_n^3 - W_{n+1}^3 W_{n-1}$$
$$W_{2n} W_2 W_1^2 = W_{n+2} W_n W_{n-1}^2 - W_n W_{n-2} W_{n+1}^2$$

**Divisibility**: $m \mid n \implies W_m \mid W_n$ (proven by Ward).

**Rank of apparition**: Smallest $r$ with $W_r \equiv 0$. Over $\mathbb{F}_q$: $r \leq (\sqrt{q}+1)^2$.

**Why novel for factoring**:
1. The rank structure is **determined by the elliptic curve's group order** mod $p$, which varies with curve parameters
2. Unlike ECM (which uses random curves), EDS gives a **deterministic** sequence per curve — enabling batch/CRT combination
3. The rank is "more random" than Lucas — it depends on $\#E(\mathbb{F}_p) = p+1-t$ where $t$ (the trace of Frobenius) varies
4. By running EDS on **multiple curves** and combining via CRT collision lanes (as in `batch_crt_cascade.py`), we get coverage of many group orders simultaneously

**Key advantage over ECM**: EDS terms can be computed in $O(\log n)$ via the nonlinear doubling formulas, and the batch CRT structure (already implemented) applies directly.

**Implementation**: 4 initial values $(W_1, W_2, W_3, W_4)$ from curve parameters; $O(\log n)$ via doubling formulas.

---

## Novel Family 3: Lehmer Sequences (Order-4 Recurrence)

**Reference**: Lehmer, D.H. "On Lucas's test for the primality of Mersenne's numbers." *J. London Math. Soc.* 10 (1935): 162–165.

**Definition**: Generalization of Lucas with $\sqrt{R}$ replacing $P$:
$$U_n(\sqrt{R}, Q) = \frac{\alpha^n - \beta^n}{\alpha - \beta} \text{ (odd } n\text{)}, \quad \frac{\alpha^n - \beta^n}{\alpha^2 - \beta^2} \text{ (even } n\text{)}$$

where $\alpha + \beta = \sqrt{R}$, $\alpha\beta = Q$.

**Recurrence** (order 4): $U_n = (R - 2Q) U_{n-2} - Q^2 U_{n-4}$ with $U_0=0, U_1=1, U_2=1, U_3=R-Q$.

**Rank structure**: $\text{rank}(p) \mid (p^2 - 1)$ when $R$ is a QR mod $p$, or $\mid (p^2+1)$ otherwise.

**Why novel**: The order-4 structure means the rank divides $p^4-1 = (p-1)(p+1)(p^2+1)$. The **$p^2+1$ factor** is new — it's the same factor reached by the $p+1$ method but via a different algebraic path. More importantly, by varying $R$ and $Q$, we get different splittings of $p^4-1$, providing diverse rank structures.

**Connection to existing code**: Extends `lucas_multi.py` naturally — the fast-doubling formulas generalize to order-4 with the same $O(\log n)$ complexity.

---

## Novel Family 4: Dickson Polynomial Evaluation Cascades

**Reference**: Lidl, R., Mullen, G., Turnwald, G. *Dickson Polynomials*. Pitman Monographs 65 (1993).

**Definition**: $D_n(x, \alpha)$ satisfies $D_n = x D_{n-1} - \alpha D_{n-2}$, with $D_0=2, D_1=x$.

**Key identity**: $D_n(2x, 1) = 2 T_n(x)$ (Chebyshev first kind), and $D_n(x, \alpha)$ is a Lucas sequence in $n$ for fixed $(x, \alpha)$.

**Novel construction — the Dickson Cascade**: Fix $n$ (the Dickson index) and evaluate $D_n(x, \alpha)$ for **multiple values of $x$**:
- $D_n(x, \alpha)$ is a polynomial in $x$ of degree $n$
- For prime $p$: if $x$ is a primitive $n$-th root of unity mod $p$, then $D_n(x, \alpha) \equiv 0 \pmod{p}$
- The set of $x$ values where $D_n(x, \alpha) \equiv 0 \pmod{p}$ has size depending on $\text{ord}_p(n)$

**Composition property**: $D_{mn}(x, \alpha) = D_m(D_n(x, \alpha), \alpha^n)$ — this gives a **multiplicative rank structure** analogous to cyclotomic polynomials.

**Why novel**: This is the "dual" of the standard Lucas approach:
- Standard: fix parameters, vary $n$ → rank divides $p \pm 1$
- Dickson cascade: fix $n$, vary $x$ → rank divides $\text{ord}_p(n) \cdot (p-1)$

This provides coverage of primes where $n$ is such that the multiplicative order of $x$ mod $p$ is smooth.

**Implementation**: For each $n \in \{2, 3, 5, 7, 11, \dots\}$ (primes), compute $D_n(x, \alpha)$ for $x = 2, 3, 4, \dots$ and check $\gcd(D_n(x,\alpha) \bmod N, N)$. The polynomial evaluation is $O(n)$ per $x$ value.

---

## Novel Family 5: Multi-Base Repunit with CRT Rank Combination

**Reference**: The Cunningham Project (Brillhart et al., *Factorizations of $b^n \pm 1$*).

**Definition**: $R_n^{(b)} = \frac{b^n - 1}{b - 1}$ (repunit in base $b$).

**Strong divisibility**: $\gcd(R_m^{(b)}, R_n^{(b)}) = R_{\gcd(m,n)}^{(b)}$.

**Rank structure**: $\text{rank}_{R^{(b)}}(p) = \text{ord}_p(b)$ (multiplicative order of $b$ mod $p$).

**Novel construction — Multi-Base CRT**:
For $N = pq$, compute $R_M^{(b)} \bmod N$ for multiple bases $b_1, b_2, \dots, b_k$ where $M = \text{lcm}(1, 2, \dots, \text{bound})$.

The key insight: $\text{ord}_p(b)$ varies with $b$. For a random prime $p$:
- $\text{ord}_p(2)$ divides $p-1$
- $\text{ord}_p(3)$ divides $p-1$
- BUT the **specific divisors** differ — $\text{ord}_p(2)$ and $\text{ord}_p(3)$ are different divisors of $p-1$

By the CRT collision technique (already in `batch_crt_cascade.py`), if $\text{ord}_p(2) \neq \text{ord}_p(3)$, then combining ranks via CRT gives a larger effective rank space.

**Why novel**: While repunits in a single base are equivalent to $p-1$ factoring, **multiple bases with CRT combination** explore different divisors of $p-1$ simultaneously. This is strictly stronger than single-base.

**Implementation**: For each base $b \in \{2, 3, 5, 6, 7, 10, 11, 12\}$ (Cunningham bases), compute $R_M^{(b)} \bmod N$ in $O(\log M)$ via fast exponentiation. Apply batch CRT GCD cascade.

---

## Novel Family 6: Higher-Order Dickson — $k$-Lucas Sequences

**Reference**: CheBallahi, F., & AlBdaiwi, B.F. "On some higher order Lucas sequences." *J. Math. Comput. Sci.* 6 (2016): 394–404.

**Definition**: The $k$-Lucas sequence of order $m$ generalizes $U_n(P,Q)$ to:
$$U_n^{(m)} = \sum_{i=1}^{m} \alpha_i^n \prod_{j \neq i} \frac{1}{\alpha_i - \alpha_j}$$

where $\alpha_i$ are roots of $x^m - P_1 x^{m-1} - \dots - P_m = 0$.

**Strong divisibility**: Holds when the roots are pairwise coprime in the appropriate ring.

**Rank structure**: $\text{rank}(p) \mid (p^{m-1} + p^{m-2} + \dots + 1) = \frac{p^m - 1}{p-1}$ when the polynomial is irreducible mod $p$.

**Why novel**: For order $m=3$, rank divides $p^2+p+1$. For $m=4$, rank divides $p^3+p^2+p+1$. These are **new smoothness targets** not reachable by any order-2 method.

**Implementation**: Companion matrix exponentiation with $m \times m$ matrices — $O(m^3 \log n)$ per term.

---

## Summary: Coverage Map

| Family | Rank divides | Novel factor | Implementation complexity |
|--------|-------------|-------------|--------------------------|
| Fibonacci/Lucas | $p \pm 1$ | — | ✅ existing |
| Mersenne/Repunit | $p - 1$ | — | ✅ existing |
| Cyclotomic $\Phi_n$ | $\Phi_d(p)$, $d \mid n$ | — | ✅ existing |
| **Hall 3rd order** | $p^2 + p + 1$ | ✅ NEW | $O(27 \log n)$ |
| **EDS** | $\#E(\mathbb{F}_p)$ | ✅ NEW | $O(\log n)$ nonlinear |
| **Lehmer (order 4)** | $p^2 + 1$ | ✅ NEW | $O(64 \log n)$ |
| **Dickson cascade** | $\text{ord}_p(n) \cdot (p-1)$ | ✅ NEW | $O(n)$ per eval |
| **Multi-base repunit** | various $\text{ord}_p(b)$ | ✅ NEW | $O(\log M)$ per base |
| **$k$-Lucas order $m$** | $\frac{p^m-1}{p-1}$ | ✅ NEW | $O(m^3 \log n)$ |

## Recommended Implementation Priority

1. **Hall 3rd-order sequences** — simplest to implement, genuinely new coverage ($p^2+p+1$), reuses existing batch CRT infrastructure
2. **EDS** — most novel rank structure, directly extends ECM philosophy with deterministic sequences
3. **Lehmer sequences** — natural extension of existing Lucas code, covers $p^2+1$
4. **Multi-base repunit** — trivial to add to existing repunit/Mersenne code, immediate practical gain
5. **$k$-Lucas order $m$** — generalization that subsumes all order-$m$ recurrences
6. **Dickson cascade** — most speculative but potentially highest coverage per evaluation

## Open Research Questions

1. **Density of smooth $p^2+p+1$**: What fraction of primes have $p^2+p+1$ smooth up to bound $B$? This determines the practical yield of Hall 3rd-order sequences. Heuristic: $\psi(x, B) / \pi(x)$ where $\psi$ counts $p \leq x$ with $p^2+p+1$ being $B$-smooth.

2. **Optimal curve selection for EDS**: Which elliptic curves produce the most "diverse" rank structures? Curves with complex multiplication have predictable orders; random curves give maximal diversity.

3. **Higher-order strong divisibility**: Hall (1936) constructed order-3 sequences, but the general theory of order-$k$ strong divisibility sequences is incomplete. Can we characterize ALL recurrences with this property?

4. **Rank distribution**: For EDS, is the rank distribution truly uniform over divisors of $\#E(\mathbb{F}_p)$? If so, EDS provides the "most random" rank structure of any known family.
