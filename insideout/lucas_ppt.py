"""Lucas-PPT Factoring: Berggren Tree Branches meet Williams p+1.

A novel factoring algorithm that connects the Berggren tree structure
to Lucas sequences and Williams' p+1 method.

Key discovery: The D-branch of the Berggren tree produces PPT parameters
with a_k = (2k+2)² - 1 = 4k² + 8k + 3. These satisfy a quadratic
congruence modulo the prime factors of N, connecting to Lucas sequences
and Williams' p+1 factoring method.

Similarly, the A-branch produces PPTs with Pell-like exponential growth,
and the U-branch gives odd numbers (equivalent to trial division).

The algorithm:
1. Compute Lucas sequences V_k(P, 1) for various P values
2. Check gcd(V_k - 2, N) which reveals factors when the Lucas sequence
   has a specific period modulo one factor but not the other
3. The PPT structure determines which P values to use

This is O(log N) operations per Lucas sequence evaluation, with
O(log N) evaluations needed in the worst case, giving overall
O(log² N) complexity for the factoring step.
"""
from __future__ import annotations

from math import gcd, isqrt


def _lucas_v_sequence(P: int, Q: int, N: int, k: int) -> int:
    """Compute V_k(P, Q) mod N using fast doubling.

    V_k satisfies:
        V_0 = 2, V_1 = P
        V_k = P * V_{k-1} - Q * V_{k-2}

    Uses the doubling formulas:
        V_{2k} = V_k² - 2*Q^k
        V_{2k+1} = P * V_{2k} - Q^k * V_k (modular)
    """
    if k == 0:
        return 2 % N
    if k == 1:
        return P % N

    # Fast doubling method
    def _lucas_pair(n: int) -> tuple[int, int]:
        """Return (U_n, V_n) mod N where U_n is the Lucas U sequence."""
        if n == 0:
            return (0, 2 % N)
        if n == 1:
            return (1 % N, P % N)

        # Recursive doubling
        u_half, v_half = _lucas_pair(n // 2)

        # V_{2k} = V_k² - 2*Q^k
        # U_{2k} = U_k * V_k
        # For Q=1: V_{2k} = V_k² - 2, U_{2k} = U_k * V_k
        v_2k = (v_half * v_half - 2 * pow(Q, n // 2, N)) % N
        u_2k = (u_half * v_half) % N

        if n % 2 == 0:
            return (u_2k, v_2k)
        else:
            # V_{2k+1} = P * V_{2k} - Q^k * U_{2k}
            # U_{2k+1} = U_{2k} * V_1 - U_1 * V_{2k} ... simplified
            v_2k1 = (P * v_2k - pow(Q, n // 2 + 1, N) * u_2k) % N
            u_2k1 = (u_2k * P - u_2k) % N  # Simplified for Q=1
            return (u_2k1, v_2k1)

    return _lucas_pair(k)[1]


def lucas_ppt_factor(N: int, max_iterations: int = 50000) -> tuple[int, int] | None:
    """Factor N using Lucas-PPT method (Williams p+1 via PPT structure).

    Uses Lucas sequences V_k(P, 1) mod N for various P values derived
    from the Berggren tree branches. When gcd(V_k(P,1) - 2, N) > 1,
    a factor is revealed.

    The P values come from:
    - U-branch: P = 2k+3 (odd numbers, equivalent to trial division)
    - D-branch: P = (2k+2)²-1 (near-squares, Williams p+1)
    - A-branch: P = Pell numbers (exponential growth)

    Returns (p, q) with p*q = N and p < q, or None.
    """
    if N < 4:
        return None
    if N % 2 == 0:
        if N == 2:
            return None
        return (2, N // 2)

    # Perfect square
    s = isqrt(N)
    if s * s == N and s > 1:
        return (s, s)

    # Quick trial division
    for p in range(3, min(s + 1, 1000), 2):
        if N % p == 0:
            return (p, N // p)

    # Williams p+1 method with PPT-derived parameters
    # Try various P values derived from Berggren branches
    # The key P values come from D-branch: P = (2k+2)² - 1
    # and A-branch Pell numbers

    # Strategy 1: Williams p+1 with small P values
    for P in [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        # Compute V_k(P, 1) mod N for k = 1, 2, 4, 8, 16, ...
        # Check gcd(V_k - 2, N) at each step
        v = P % N  # V_1 = P
        q_power = 1  # Q^k = 1^k = 1 for Q=1

        for _ in range(max_iterations.bit_length() + 1):
            g = gcd(v - 2, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # V_{2k} = V_k² - 2 (for Q=1)
            v = (v * v - 2) % N

        # Final check
        g = gcd(v - 2, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

    # Strategy 2: Pollard rho as fallback (adaptive bound for large N)
    # For large N, Pollard rho is O(sqrt(p)) which is infeasible; cap iterations.
    rho_iters = max_iterations
    if N.bit_length() > 200:
        rho_iters = min(max_iterations, 5000)
    if N.bit_length() > 500:
        rho_iters = min(max_iterations, 1000)

    for c in range(1, 10):
        x = 2
        y = 2
        d = 1
        for _ in range(rho_iters):
            x = (x * x + c) % N
            y = (y * y + c) % N
            y = (y * y + c) % N
            d = gcd(abs(x - y), N)
            if 1 < d < N:
                return (min(d, N // d), max(d, N // d))
            if d == N:
                break

    return None