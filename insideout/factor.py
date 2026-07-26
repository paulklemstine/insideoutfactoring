"""Top-level factoring API.

Provides the main entry point for the Inside-Out factoring algorithm.
Tries multiple strategies in order of expected speed:

1. Perfect square detection (O(1))
2. CF convergent divisibility pre-check (O(log N))
3. Quick trial division for small factors
4. Brahmagupta-Fibonacci two-square method (for N ≡ 1 mod 4)
5. Fermat difference-of-squares (for close factors)
6. Fibonacci GCD factorization
7. Resonance Cascade (CF-convergent resonance + Möbius descent + squaring conductance)
8. Lucas-PPT (Williams p+1 via Berggren tree branch structure)
9. Spectral Cascade (CF squaring + SL₂ matrix order + QR discriminator + idempotent + near-square + walk)
10. Fibonacci-Pythagorean (smooth-rank Fibonacci + Pythagorean batched GCD)
11. Multi-Parameter Lucas (smooth-rank with multiple P parameters, decorrelated ranks)
12. CRT Collision (cross-product Lucas sequence collision detection)
13. Inside-Out Relation Generator (Berggren tree → smooth relations → congruence of squares)
14. SL₂ Group-Order Cascade (smooth group order in SL₂(Z/NZ))
15. Batch CRT Cascade (O(K²) multiplications + O(1) GCD for Lucas cross-products)
16. PPT Quadratic Sieve (PPT-structured smooth relation collection)
17. CF Matrix Cascade (CF period matrix + SL₂ group-order hybrid)
18. CF Cascade (pure CF convergent residue check)
19. PPT Form Cascade (binary quadratic forms from PPT parameters)
20. SQUFOF (Shanks' Square Form Factorization)
21. Class-Group Cascade (smooth-class-order in Cl(4N))
22. Class-Group SQUFOF (SQUFOF with PPT starting forms)
23. Cyclotomic Resultant (Φ_m(a) mod N for multiple orders m)
24. Cyclotomic Cascade (smooth-bound powering with Φ_m evaluation)
25. Discriminant Resonance (CRT divergence in quadratic discriminants)
26. Quadratic Resonance (smooth-bound powering with discriminant checks)
27. Hensel Cascade (Hensel lift + cyclotomic + QR checks)
28. CRT Lattice (multi-base CRT divergence detection)
30. Lattice-Combined (LLL reduction on smooth relations)
31. Hybrid Smooth (existing methods + lattice combination)
32. Graph-Order Cascade (multiplicative order graph + bridge detection)
33. Order Spectrum Analysis (order spectrum comparison)
34. Inside-Out (CF-steered best-first search + BFS)
32. Wavefront search
33. Full trial division
19. Inside-Out (CF-steered best-first search + BFS)
20. Wavefront search
21. Full trial division
"""
from __future__ import annotations

from math import isqrt

from .inside_out import inside_out_factor
from .wavefront import search_wavefront
from .cf_guide import cf_factor_check
from .brahmagupta import brahmagupta_fibonacci_factor, fermat_difference_of_squares
from .fibonacci_factor import fibonacci_gcd_factor, pisano_factor
from .resonance_cascade import resonance_cascade_factor
from .lucas_ppt import lucas_ppt_factor
from .projective_collision import chart_collision_factor as projective_chart_factor
from .orbit_smooth_relation import orbit_smooth_relation_factor
from .spectral_factor import spectral_cascade_factor
from .relation_generator import relation_factor
from .fibonacci_pythagorean import fibonacci_pythagorean_factor
from .lucas_multi import lucas_multi_factor, crt_collision_factor
from .sl2_group_order import sl2_group_order_factor, sl2_structured_factor
from .batch_crt_cascade import batch_crt_cascade_factor
from .ppt_quadratic_sieve import ppt_quadratic_sieve
from .cf_matrix_cascade import cf_matrix_cascade_factor, cf_cascade_factor
from .ppt_form_cascade import ppt_form_cascade_factor, squfof_factor
from .class_group_cascade import class_group_cascade_factor, class_group_squfof_factor
from .cyclotomic_resultant import cyclotomic_resultant_factor, cyclotomic_cascade_factor
from .resultant_cascade import discriminant_resonance_factor, quadratic_resonance_factor
from .hensel_cascade import hensel_cascade_factor, crt_lattice_factor
from .lattice_factor import lattice_factor, hybrid_smooth_factor
from .graph_order import graph_order_cascade_factor, order_spectrum_factor
from .coppersmith import coppersmith_factor
from .hybrid_cyclo_sl2 import hybrid_cyclo_sl2_factor


def factor(N: int) -> tuple[int, int] | None:
    """Factor an integer N into two factors p and q where N = p*q.

    Uses the Inside-Out factoring algorithm with wavefront search
    and trial division as fallback.

    Returns (p, q) with p < q if N is composite, None if N is prime.
    """
    result = factor_with_method(N)
    if result is None:
        return None
    return result[0]


def factor_with_method(N: int) -> tuple[tuple[int, int], str] | None:
    """Factor N and return the factors along with the method used.

    Method name is one of: "perfect_square", "cf_precheck",
    "brahmagupta", "fermat", "fibonacci", "resonance_cascade",
    "lucas_ppt", "spectral_cascade", "fib_pyth", "lucas_multi",
    "crt_collision", "sl2_group_order", "sl2_structured",
    "batch_crt", "ppt_sieve", "cf_matrix_cascade", "cf_cascade",
    "ppt_form", "squfof", "class_group", "class_squfof",
    "cyclotomic_resultant", "cyclotomic_cascade",
    "discriminant_resonance", "quadratic_resonance",
    "hensel_cascade", "crt_lattice",
    "lattice_factor", "hybrid_smooth",
    "graph_order", "order_spectrum",
    "coppersmith", "hybrid_cyclo_sl2", "relation_gen",
    "projective_chart", "orbit_relation",
    "inside_out", "wavefront", "trial_division",
    """
    if N < 4:
        return None

    # Handle even N
    if N % 2 == 0:
        if N == 2:
            return None
        return ((2, N // 2), "trial_division")

    # Perfect square detection: if N = p^2, then p is a factor
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return ((sqrt_N, sqrt_N), "perfect_square")

    # CF convergent divisibility pre-check
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        p, q = cf_result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "cf_precheck")

    # Quick trial division for small factors (safety net)
    for p in range(3, min(isqrt(N) + 1, 1000), 2):
        if N % p == 0:
            return ((p, N // p), "trial_division")

    # Strategy: Brahmagupta-Fibonacci two-square method
    # Effective for N ≡ 1 mod 4 (products of primes ≡ 1 mod 4)
    bf_result = brahmagupta_fibonacci_factor(N)
    if bf_result is not None:
        p, q = bf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "brahmagupta")

    # Strategy: Fermat difference-of-squares
    # Effective for close factors (p ~ q)
    fermat_result = fermat_difference_of_squares(N)
    if fermat_result is not None:
        p, q = fermat_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "fermat")

    # Strategy: Fibonacci GCD factorization
    # Effective for N where some prime factor p has a small entry point α(p)
    fib_result = fibonacci_gcd_factor(N, bound=5000)
    if fib_result is not None:
        p, q = fib_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "fibonacci")

    # Strategy: Lucas-PPT Factoring (Williams p+1 via PPT structure)
    # Uses Lucas sequences derived from Berggren tree branches
    # FAST: typically sub-millisecond when it succeeds
    lp_result = lucas_ppt_factor(N)
    if lp_result is not None:
        p, q = lp_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "lucas_ppt")

    # Strategy: Projective Chart Collision (chart compression + distinguished walks)
    # Places after lucas_ppt (fast sub-ms) and before resonance_cascade (slower)
    pc_result = projective_chart_factor(N, max_steps=50000, num_walks=16)
    if pc_result is not None:
        p, q = pc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "projective_chart")

    # Strategy: Resonance Cascade Factoring
    # Combines CF-convergent resonance, Möbius descent, and squaring conductance
    # SLOWER: typically 50-100ms when it succeeds
    rc_result = resonance_cascade_factor(N)
    if rc_result is not None:
        p, q = rc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "resonance_cascade")

    # Strategy: Orbit Smooth Relation (NFS-style: orbit norms → smooth relations → linear algebra)
    # Research experiment — placed after established methods; slow, success not guaranteed
    or_result = orbit_smooth_relation_factor(N, bound=30000, word_length=15)
    if or_result is not None:
        p, q = or_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "orbit_relation")

    # Strategy: Spectral Cascade (CF squaring + SL₂ matrix order + QR discriminator + idempotent + near-square + walk)
    # Combines CF-convergent squaring orbits, SL₂(Z/NZ) matrix powers,
    # QR discriminator, idempotent detection, near-square search, and CF-guided walk
    sc_result = spectral_cascade_factor(N)
    if sc_result is not None:
        p, q = sc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "spectral_cascade")

    # Strategy: Inside-Out Relation Generator (Berggren tree → smooth relations → congruence of squares)
    # Uses PPT structure to generate smooth relations, then combines via linear algebra
    rg_result = relation_factor(N)
    if rg_result is not None:
        p, q = rg_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "relation_gen")

    # Strategy: Fibonacci-Pythagorean Hybrid (smooth-rank Fibonacci + Pythagorean batched GCD)
    # Computes Q^M mod N via companion matrix, derives F_M, L_M, A_M, B_M, C_M
    # and batches all 5 into one GCD check. Fast when Z(p) is smooth.
    fp_result = fibonacci_pythagorean_factor(N, bound=10000)
    if fp_result is not None:
        p, q = fp_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "fib_pyth")

    # Strategy: Multi-Parameter Lucas (smooth-rank with multiple P parameters, decorrelated ranks)
    # Tries multiple Lucas parameters P, each giving independent rank of apparition
    lm_result = lucas_multi_factor(N, bound=5000, stage2_bound=1000, max_params=8)
    if lm_result is not None:
        p, q = lm_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "lucas_multi")

    # Strategy: CRT Collision (cross-product Lucas sequence collision detection)
    # Detects factors from projective state differences between independent Lucas sequences
    cc_result = crt_collision_factor(N, bound=3000, stage2_bound=500, max_params=6)
    if cc_result is not None:
        p, q = cc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "crt_collision")

    # Strategy: SL₂ Group-Order Cascade (smooth group order in SL₂(Z/NZ))
    # Analogous to ECM: uses 2×2 matrix groups. Group order p(p²-1) = p(p-1)(p+1)
    # gives three independent smoothness targets. Sub-exponential L_p[1/2].
    sl2_result = sl2_group_order_factor(N, bound=50000, curves=10)
    if sl2_result is not None:
        p, q = sl2_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "sl2_group_order")

    # Strategy: SL₂ with Berggren matrix starting points
    sl2s_result = sl2_structured_factor(N, bound=10000)
    if sl2s_result is not None:
        p, q = sl2s_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "sl2_structured")

    # Strategy: Batch CRT Cascade (O(K²) multiplications + O(1) GCD)
    # Extends CRT collision lane to batch cross-product detection
    bcc_result = batch_crt_cascade_factor(N, bound=5000, stage2_bound=1000, max_params=16, stages=2)
    if bcc_result is not None:
        p, q = bcc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "batch_crt")

    # Strategy: PPT Quadratic Sieve (PPT-structured smooth relation collection)
    # Uses PPT parameters (m,n) generating values m²+n², m²-n², 2mn for sieving
    ppts_result = ppt_quadratic_sieve(N, bound=1000, sieve_range=10000, max_relations=200)
    if ppts_result is not None:
        p, q = ppts_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "ppt_sieve")

    # Strategy: CF Matrix Cascade (CF period matrix + SL₂ group-order hybrid)
    # Combines CFRAC-style residue checks with smooth-group-order detection
    # using the CF period matrix of √N. Novel: uses the actual period matrix
    # rather than random SL₂ matrices.
    cfm_result = cf_matrix_cascade_factor(N, bound=50000, cf_steps=10000,
                                           smooth_bound=500, max_relations=200)
    if cfm_result is not None:
        p, q = cfm_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "cf_matrix_cascade")

    # Strategy: CF Cascade (pure CF convergent residue check)
    # Lightweight: just computes CF convergents and checks gcd(r_k, N)
    cfc_result = cf_cascade_factor(N, cf_steps=50000)
    if cfc_result is not None:
        p, q = cfc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "cf_cascade")

    # Strategy: PPT Form Cascade (binary quadratic forms from PPT parameters)
    # Uses PPT-derived forms (m²-n², 2mn, m²+n²) with discriminant and SQUFOF reduction
    ptf_result = ppt_form_cascade_factor(N, max_ppt=10000, squfof_steps=50000)
    if ptf_result is not None:
        p, q = ptf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "ppt_form")

    # Strategy: SQUFOF (Shanks' Square Form Factorization)
    # Standard binary quadratic form method for comparison
    sqf_result = squfof_factor(N, max_iterations=100000)
    if sqf_result is not None:
        p, q = sqf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "squfof")

    # Strategy: Class-Group Cascade (smooth-class-order in Cl(4N))
    # Analogous to ECM but using the class group of binary quadratic forms
    # instead of elliptic curves. PPT-derived and random starting forms.
    cg_result = class_group_cascade_factor(N, bound=50000, curves=15)
    if cg_result is not None:
        p, q = cg_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "class_group")

    # Strategy: Class-Group SQUFOF (SQUFOF with PPT starting forms)
    cgs_result = class_group_squfof_factor(N, max_iterations=100000)
    if cgs_result is not None:
        p, q = cgs_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "class_squfof")

    # Strategy: Cyclotomic Resultant (Φ_m(a) mod N for multiple orders m)
    cr_result = cyclotomic_resultant_factor(N, max_order=30, smooth_bound=50000)
    if cr_result is not None:
        p, q = cr_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "cyclotomic_resultant")

    # Strategy: Cyclotomic Cascade (smooth-bound powering with Φ_m evaluation)
    cc_result = cyclotomic_cascade_factor(N, bound=50000, base_points=10)
    if cc_result is not None:
        p, q = cc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "cyclotomic_cascade")

    # Strategy: Discriminant Resonance (CRT divergence in quadratic discriminants)
    dr_result = discriminant_resonance_factor(N, max_disc=1000, max_forms=100)
    if dr_result is not None:
        p, q = dr_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "discriminant_resonance")

    # Strategy: Quadratic Resonance (smooth-bound powering with discriminant checks)
    qr_result = quadratic_resonance_factor(N, bound=50000, bases=10)
    if qr_result is not None:
        p, q = qr_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "quadratic_resonance")

    # Strategy: Hensel Lifting Cascade (Hensel lift + cyclotomic + QR checks)
    hc_result = hensel_cascade_factor(N, bound=50000, max_lifts=10, base_points=10)
    if hc_result is not None:
        p, q = hc_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "hensel_cascade")

    # Strategy: CRT Lattice (multi-base CRT divergence detection)
    cl_result = crt_lattice_factor(N, bound=50000, lattice_dim=8)
    if cl_result is not None:
        p, q = cl_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "crt_lattice")

    # Strategy: Lattice-Combined Factoring (LLL reduction on smooth relations)
    lf_result = lattice_factor(N, bound=100000, target_relations=200)
    if lf_result is not None:
        p, q = lf_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "lattice_factor")

    # Strategy: Hybrid Smooth (existing methods + lattice combination)
    hs_result = hybrid_smooth_factor(N, bound=100000, target_relations=200)
    if hs_result is not None:
        p, q = hs_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "hybrid_smooth")

    # Strategy: Graph-Order Cascade (order graph analysis + bridge detection)
    go_result = graph_order_cascade_factor(N, bound=50000, max_exp=50, bases=8)
    if go_result is not None:
        p, q = go_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "graph_order")

    # Strategy: Order Spectrum Analysis
    os_result = order_spectrum_factor(N, bound=50000, spectrum_size=30)
    if os_result is not None:
        p, q = os_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "order_spectrum")

    # Strategy: Coppersmith's Method (polynomial-time for close factors)
    # Use sqrt(N) as approximation to the smaller factor
    cop_result = coppersmith_factor(N, X=isqrt(N))
    if cop_result is not None:
        p, q = cop_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "coppersmith")

    # Strategy: Hybrid Cyclo-SL2 Cascade (combines both strongest methods)
    hyb_result = hybrid_cyclo_sl2_factor(N, time_budget_ms=5000)
    if hyb_result is not None:
        p, q = hyb_result
        if p * q == N and 1 < p < N and 1 < q < N:
            return ((min(p, q), max(p, q)), "hybrid_cyclo_sl2")

    # Strategy: Inside-Out (CF-steered best-first search + BFS)
    result = inside_out_factor(N, max_iterations=50000)
    if result is not None:
        p, q = result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "inside_out")

    # Strategy: Wavefront search
    result = search_wavefront(N, max_radius=500)
    if result is not None:
        p, q = result
        if p * q == N and p > 1 and q > 1:
            return ((min(p, q), max(p, q)), "wavefront")

    # Fallback: Full trial division
    limit = isqrt(N) + 1
    for p in range(3, limit, 2):
        if N % p == 0:
            return ((p, N // p), "trial_division")

    return None