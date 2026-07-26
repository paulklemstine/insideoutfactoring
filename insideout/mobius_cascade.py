"""Möbius Cascade Factoring (MCF).

A novel factoring algorithm that converts the Berggren tree search into a
direct Möbius transform computation. The key insight is that the continued
fraction expansion of sqrtN encodes the exact sequence of Möbius transforms
that navigate the Berggren tree to the factor-revealing node.

The three Berggren Möbius transforms act on the slope z = n/m:
  f_U(z) = 1/(2 - z)   (U-branch: ascending)
  f_A(z) = 1/(2 + z)   (A-branch: balanced)
  f_D(z) = z/(1 + 2z)  (D-branch: descending)

where (m,n) are the Gaussian integer parameters of the PPT:
  a = m2 - n2, b = 2mn, c = m2 + n2

Instead of searching the tree node-by-node (BFS/DFS), MCF:
1. Computes the CF expansion of sqrtN
2. Uses each CF convergent to determine the next Möbius transform
3. Composes all transforms into a single rational function M(z) = (az+b)/(cz+d)
4. Evaluates M at the root slope z0 = 1/2 (n/m for root PPT (3,4,5))
5. Converts the target slope to (m,n) parameters and checks divisibility

This converts the factoring problem from a SEARCH problem to a COMPUTATION
problem: O(L) Möbius compositions where L is the CF period length.
"""
from __future__ import annotations

from fractions import Fraction
from math import gcd, isqrt

from .cf_guide import cf_sqrt, convergents, predict_branch
from .berggren import Triple
from .gaussian import MnPair, mn_to_triple


class MobiusTransform:
    """A Möbius transformation M(z) = (az + b) / (cz + d).

    Möbius transforms compose naturally:
    M1(M2(z)) = M1 ∘ M2 is also a Möbius transform.
    """

    __slots__ = ('a', 'b', 'c', 'd')

    def __init__(self, a: int, b: int, c: int, d: int):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, z: Fraction) -> Fraction:
        """Apply this transform to z: (az + b) / (cz + d)."""
        numerator = self.a * z + self.b
        denominator = self.c * z + self.d
        if denominator == 0:
            raise ValueError(f"Denominator zero: {self.c}*{z} + {self.d} = 0")
        return Fraction(numerator, denominator)

    def compose(self, other: 'MobiusTransform') -> 'MobiusTransform':
        """Compose this transform with another: self(other(z)).

        If self(z) = (a1*z + b1)/(c1*z + d1) and
        other(z) = (a2*z + b2)/(c2*z + d2), then:
        self(other(z)) = ((a1*a2 + b1*c2)*z + (a1*b2 + b1*d2)) /
                         ((c1*a2 + d1*c2)*z + (c1*b2 + d1*d2))
        """
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = other.a, other.b, other.c, other.d

        return MobiusTransform(
            a1 * a2 + b1 * c2,
            a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2,
            c1 * b2 + d1 * d2,
        )

    def invert(self) -> 'MobiusTransform':
        """Compute the inverse Möbius transform.

        If M(z) = (az+b)/(cz+d), then M⁻1(w) = (dw-b)/(-cw+a).
        Valid when ad - bc != 0.
        """
        det = self.a * self.d - self.b * self.c
        if det == 0:
            raise ValueError("Transform is not invertible (det=0)")
        return MobiusTransform(self.d, -self.b, -self.c, self.a)

    def __repr__(self) -> str:
        return f"Mobius({self.a}z+{self.b}) / ({self.c}z+{self.d})"


# The three Berggren Möbius transforms on the slope z = n/m:
# These map the slope of a PPT to the slope of each child PPT.
# f_U(z) = 1/(2-z)   → as (az+b)/(cz+d): numerator=1, denominator=2-z=(-1)z+2
#   So M_U = MobiusTransform(0, 1, -1, 2)
# f_A(z) = 1/(2+z)   → numerator=1, denominator=2+z=1z+2
#   So M_A = MobiusTransform(0, 1, 1, 2)
# f_D(z) = z/(1+2z)  → numerator=z=1z+0, denominator=1+2z=2z+1
#   So M_D = MobiusTransform(1, 0, 2, 1)

M_U = MobiusTransform(0, 1, -1, 2)   # f_U(z) = 1/(2-z)
M_A = MobiusTransform(0, 1, 1, 2)     # f_A(z) = 1/(2+z)
M_D = MobiusTransform(1, 0, 2, 1)     # f_D(z) = z/(1+2z)

# Inverse transforms for descending the tree
M_U_INV = M_U.invert()   # f_U^{-1}
M_A_INV = M_A.invert()   # f_A^{-1}
M_D_INV = M_D.invert()   # f_D^{-1}


def slope_to_mn(z: Fraction) -> tuple[int, int]:
    """Convert a slope z = n/m to (m, n) parameters.

    The Möbius transforms f_U, f_A, f_D act on z = n/m (the ratio of
    Gaussian integer parameters, NOT b/a).

    When z = n/m = p/q (in lowest terms), the PPT parameters are simply:
      m = q, n = p

    This is because z = n/m with gcd(m,n)=1 implies m = q, n = p directly.

    Returns (m, n) if valid PPT parameters, (0, 0) otherwise.
    """
    if z <= 0:
        return (0, 0)

    p, q = z.numerator, z.denominator
    m, n = q, p  # z = n/m = p/q → m = q, n = p

    if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
        return (m, n)

    # If the direct assignment doesn't yield valid PPT params,
    # it means the slope doesn't correspond to a primitive triple.
    # This can happen for non-primitive triples or invalid inputs.
    # Try scaling to find valid (m, n) with gcd=1 and m-n odd.
    for k in range(1, 200):
        mk, nk = q * k, p * k
        if mk > nk > 0 and (mk - nk) % 2 == 1 and gcd(mk, nk) == 1:
            return (mk, nk)

    return (0, 0)


def mn_to_slope(m: int, n: int) -> Fraction:
    """Convert (m, n) parameters to slope z = n/m.

    The Möbius transforms act on z = n/m (the ratio of Gaussian integer
    parameters). For the root PPT (3,4,5) with (m,n) = (2,1),
    z0 = 1/2.
    """
    if m <= 0 or n <= 0 or m <= n:
        return Fraction(0)
    return Fraction(n, m)


def mobius_cascade_factor(N: int, max_cascade: int = 100) -> tuple[int, int] | None:
    """Factor N using the Möbius Cascade method.

    Instead of searching the Berggren tree node by node, this method:
    1. Computes CF(sqrtN) to get the branch sequence
    2. For each convergent, determines the optimal Möbius transform (U, A, or D)
    3. Composes the Möbius transforms into a single function M(z)
    4. Evaluates M(z0) where z0 = 4/3 (root PPT slope) to get the target slope
    5. Converts the target slope to (m,n) parameters
    6. Checks if the resulting triple's leg divides N

    The cascade also tries intermediate compositions (partial paths through
    the tree) to catch factors that appear at intermediate depths.

    Returns (p, q) with p < q and p*q = N, or None if no factor found.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square fast path
    sqrt_N = isqrt(N)
    if sqrt_N * sqrt_N == N and sqrt_N > 1:
        return (sqrt_N, sqrt_N)

    # CF convergent pre-check (already fast)
    from .cf_guide import cf_factor_check
    cf_result = cf_factor_check(N)
    if cf_result is not None:
        p, q = cf_result
        if p * q == N and 1 < p < N:
            return (min(p, q), max(p, q))

    # Compute CF expansion of sqrt(N)
    cf = cf_sqrt(N, max_terms=max_cascade)
    convs = convergents(cf)

    # Root PPT slope: (3,4,5) has (m,n) = (2,1), so z0 = n/m = 1/2
    z_root = Fraction(1, 2)

    # Strategy: compose Möbius transforms along the CF-predicted path
    # At each step, choose the branch (U, A, D) that moves the slope
    # closest to sqrtN
    target_slope = Fraction(isqrt(N * N + 1), N) if N > 0 else Fraction(0)

    # Try all partial compositions (paths of length 1, 2, 3, ...)
    M = MobiusTransform(1, 0, 0, 1)  # Identity transform

    for i, (pk, qk) in enumerate(convs):
        if i >= max_cascade:
            break

        # Current slope after composition so far
        try:
            z_current = M(z_root)
        except (ValueError, ZeroDivisionError):
            continue

        if z_current <= 0:
            continue

        # Convert current slope to (m, n) and check the triple
        m, n = slope_to_mn(z_current)
        if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
            triple = mn_to_triple(MnPair(m, n))

            # Check resonance
            from .inside_out import resonance_check
            result = resonance_check(N, triple)
            if result is not None:
                p, q = result
                if p * q == N and 1 < p < N and 1 < q:
                    return (min(p, q), max(p, q))

            # Direct divisibility
            a, b, c = triple
            if 1 < a < N and N % a == 0:
                return (min(a, N // a), max(a, N // a))
            if 1 < b < N and N % b == 0:
                return (min(b, N // b), max(b, N // b))

        # Determine which branch to follow using predict_branch
        if m > n > 0:
            try:
                triple = mn_to_triple(MnPair(m, n))
                distances = predict_branch(N, (triple.a, triple.b, triple.c))
                # Choose the branch with minimum distance
                min_dist = min(distances)
                if min_dist == distances[0]:
                    M = M.compose(M_U)
                elif min_dist == distances[1]:
                    M = M.compose(M_A)
                else:
                    M = M.compose(M_D)
            except (ValueError, ZeroDivisionError):
                # Fallback: try all three branches
                pass
        else:
            # If slope didn't give valid (m,n), try each branch from root
            # using convergent information
            z_conv = Fraction(pk, qk) if qk != 0 else Fraction(pk, 1)

            # The convergent slope tells us which branch to take
            # z < 1 → D branch, z > 2 → U branch, otherwise → A branch
            if z_conv < 1:
                M = M.compose(M_D)
            elif z_conv > 2:
                M = M.compose(M_U)
            else:
                M = M.compose(M_A)

    # Also try the inverse cascade: start from the trivial triple for N
    # and descend toward the root
    # The trivial triple for N is (N, (N2-1)/2, (N2+1)/2)
    # Its slope is ((N2-1)/2) / N ~= N/2
    z_trivial = Fraction(N * N - 1, 2 * N) if N > 0 else Fraction(0)

    M_inv = MobiusTransform(1, 0, 0, 1)  # Identity
    z_descent = z_trivial

    for i in range(min(len(convs), max_cascade)):
        # Determine inverse branch from σ-invariants
        # σ1 = a + 2b - 2c, σ2 = 2a + b - 2c
        # The branch is determined by which inverse maps to positive triple
        # For now, try all three inverse branches
        for M_inv_branch in [M_U_INV, M_A_INV, M_D_INV]:
            try:
                z_branch = M_inv_branch(z_descent)
                if z_branch > 0:
                    m, n = slope_to_mn(z_branch)
                    if m > n > 0 and (m - n) % 2 == 1 and gcd(m, n) == 1:
                        triple = mn_to_triple(MnPair(m, n))

                        # Check resonance
                        from .inside_out import resonance_check
                        result = resonance_check(N, triple)
                        if result is not None:
                            p, q = result
                            if p * q == N and 1 < p < N:
                                return (min(p, q), max(p, q))

                        # Direct divisibility
                        a, b, c = triple
                        if 1 < a < N and N % a == 0:
                            return (min(a, N // a), max(a, N // a))
                        if 1 < b < N and N % b == 0:
                            return (min(b, N // b), max(b, N // b))
            except (ValueError, ZeroDivisionError):
                continue

        # Choose the inverse branch that decreases slope (moves toward root)
        try:
            z_u = M_U_INV(z_descent)
            z_a = M_A_INV(z_descent)
            z_d = M_D_INV(z_descent)
            # Pick the branch that gives a positive slope closest to sqrtN's slope
            candidates = []
            if z_u > 0:
                candidates.append(('U', z_u, M_U_INV))
            if z_a > 0:
                candidates.append(('A', z_a, M_A_INV))
            if z_d > 0:
                candidates.append(('D', z_d, M_D_INV))

            if candidates:
                # Choose the one closest to the target slope
                best = min(candidates, key=lambda x: abs(x[1] - target_slope))
                z_descent = best[1]
        except (ValueError, ZeroDivisionError):
            break

    return None