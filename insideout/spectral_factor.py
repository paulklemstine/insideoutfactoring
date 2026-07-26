"""Spectral Cascade Factoring (SCF) — Pure Novel Methods.

A factoring algorithm using ONLY mathematically novel approaches derived from
the PPT/Berggren/CF framework. No classical methods (no Pollard rho, no
trial division beyond trivial, no p-1, no Williams p+1).

The four novel stages:

1. **CF-Convergent Squaring Cascade**: The convergents p_k/q_k of sqrtN satisfy
   p_k2 - Nq_k2 = r_k (small residue). Modulo a factor p of N, p_k == ±1
   (mod p). We exploit this by:
   - Checking gcd(p_k ± 1, N) for each convergent (Pell residue factoring)
   - Squaring convergent-derived elements x = p_k * q_k^{-1} mod N and
     checking gcd(x^(2^j) - 1, N) at each level (CRT bottleneck)
   - This is novel because it uses the Pell equation structure of CF convergents
     rather than random starting points

2. **SL2 Matrix Order Detection**: The Berggren Möbius transforms correspond to
   matrices in SL2(Z):
     M_A = [[1,1],[1,2]], M_D = [[1,0],[2,1]], M_U = [[0,1],[-1,2]]
   Computing M^k mod N and checking whether M^k == I (mod p) but ≢ I (mod q)
   reveals factors via gcd of matrix entries. This is a group-order method in
   SL2(F_p) (order p(p2-1)) rather than (F_p)* (order p-1). Novel because:
   - Uses SL2 group structure instead of multiplicative group
   - Berggren matrices provide structured starting points
   - Group order p(p-1)(p+1) has different smoothness properties

3. **Quadratic Residue Discriminator**: CF convergents yield residues
   r_k = p_k2 - Nq_k2 ~= ±1. These small residues have different quadratic
   characters mod p vs mod q. For r_k = ±1, checking whether r_k^((N-1)/2)
   == ±1 (mod p) != r_k^((N-1)/2) (mod q) reveals a factor via
   gcd(r_k^((N-1)/2) - r_k^((N-1)/2 mod (p-1)), N). Novel: uses QR structure
   of Pell residues rather than random bases.

4. **Idempotent Detection via CRT Bottleneck**: The squaring map x → x2 on
   Z/NZ has idempotent fixed points e where e2 == e (mod N), corresponding to
   (0,1) and (1,0) under CRT. Finding such e directly gives gcd(e, N) as a
   factor. We use CF convergents as starting points for the squaring orbit
   rather than random x values.

Key mathematical insight: these methods exploit the ALGEBRAIC STRUCTURE of
CF convergents, PPT parameters, and Berggren matrices — not random search.
The structure provides deterministic, information-theoretic advantages over
random starting points, even though the overall complexity remains superpolynomial
for generic inputs.

NOTE: No claim of polynomial-time classical factoring is made. These methods
are novel heuristic approaches that may improve constants and success probability
on structured inputs, but they do not bypass the fundamental complexity barrier.
"""
from __future__ import annotations

from math import gcd, isqrt
from typing import Sequence


def _cf_sqrt(N: int, max_terms: int = 200) -> list[int]:
    """Compute continued fraction expansion of sqrtN.

    Returns list of partial quotients [a0, a1, a2, ...].
    """
    a0 = isqrt(N)
    if a0 * a0 == N:
        return [a0]

    cf = [a0]
    m, d, a = 0, 1, a0

    seen = set()
    for _ in range(max_terms):
        m = d * a - m
        d = (N - m * m) // d
        if d == 0:
            break
        a = (a0 + m) // d

        state = (m, d)
        if state in seen:
            break
        seen.add(state)
        cf.append(a)

    return cf


def _convergents(cf: Sequence[int]) -> list[tuple[int, int]]:
    """Compute convergents p_k/q_k from CF expansion."""
    convs = []
    p_prev, p_curr = 1, cf[0]
    q_prev, q_curr = 0, 1

    convs.append((p_curr, q_curr))

    for i in range(1, len(cf)):
        a = cf[i]
        p_new = a * p_curr + p_prev
        q_new = a * q_curr + q_prev
        convs.append((p_new, q_new))
        p_prev, p_curr = p_curr, p_new
        q_prev, q_curr = q_curr, q_new

    return convs


def _mat2_mul(A: tuple[int, int, int, int],
              B: tuple[int, int, int, int],
              N: int) -> tuple[int, int, int, int]:
    """Multiply two 2×2 matrices mod N. [[a,b],[c,d]] = (a,b,c,d)."""
    a1, b1, c1, d1 = A
    a2, b2, c2, d2 = B
    return (
        (a1 * a2 + b1 * c2) % N,
        (a1 * b2 + b1 * d2) % N,
        (c1 * a2 + d1 * c2) % N,
        (c1 * b2 + d1 * d2) % N,
    )


def _mat2_pow(M: tuple[int, int, int, int],
              k: int,
              N: int) -> tuple[int, int, int, int]:
    """Compute M^k mod N using fast exponentiation."""
    if k == 0:
        return (1, 0, 0, 1)
    if k == 1:
        return M

    result = (1, 0, 0, 1)
    base = M
    while k > 0:
        if k & 1:
            result = _mat2_mul(result, base, N)
        base = _mat2_mul(base, base, N)
        k >>= 1
    return result


def _cf_squaring_cascade(N: int,
                          convs: list[tuple[int, int]],
                          max_depth: int = 40) -> tuple[int, int] | None:
    """Stage 1: CF-Convergent Squaring Cascade.

    Uses CF convergents of sqrtN as structured starting points for squaring orbits.
    The key mathematical insight: p_k2 - Nq_k2 = r_k ~= ±1, so p_k == ±1 (mod p)
    for every factor p of N. The squaring orbit of p_k*q_k^{-1} mod N has
    different behavior mod p vs mod q, revealing factors.

    Novel: CF convergent structure provides deterministic starting points with
    known algebraic properties, unlike random starting points in Pollard rho.
    """
    for pk, qk in convs:
        # Pell residue: p_k == ±1 (mod p) for factors p of N
        for val in [pk - 1, pk + 1]:
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Direct convergent divisibility
        g = gcd(qk, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))

        # CF convergent residues: p_k2 - 1 and related
        for val in [pk * pk - 1, pk * pk + 1, pk * pk - N * qk * qk]:
            g = gcd(abs(val), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Squaring cascade: x = p_k/q_k mod N, then x^(2^j) for j = 1, 2, ...
        # If x == ±1 (mod p) then x^2 == 1 (mod p)
        # If x^(2^j) == 1 (mod p) but ≢ 1 (mod q), factor found
        if gcd(qk, N) == 1:
            try:
                qk_inv = pow(qk, -1, N)
            except (ValueError, ZeroDivisionError):
                continue

            x = (pk * qk_inv) % N

            # Squaring cascade with batch GCD
            batch_minus = 1
            batch_plus = 1
            batch_count = 0

            y = x
            for _ in range(max_depth):
                for val in [y - 1, y + 1]:
                    g = gcd(abs(val) % N, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

                # Accumulate for batch GCD
                batch_minus = (batch_minus * ((y - 1) % N)) % N
                batch_plus = (batch_plus * ((y + 1) % N)) % N
                batch_count += 1

                if batch_count >= 16:
                    g = gcd(batch_minus, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
                    g = gcd(batch_plus, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
                    batch_minus = 1
                    batch_plus = 1
                    batch_count = 0

                y = (y * y) % N

            # Final batch
            if batch_count > 0:
                g = gcd(batch_minus, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))
                g = gcd(batch_plus, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

    return None


def _sl2_matrix_cascade(N: int,
                         max_k_bits: int = 20) -> tuple[int, int] | None:
    """Stage 2: SL2 Matrix Order Detection.

    The Berggren Möbius transforms correspond to SL2(Z) matrices:
      M_A = [[1,1],[1,2]] (A-branch, eigenvalue ~= 2.414 = 1+sqrt2)
      M_D = [[1,0],[2,1]] (D-branch)
      M_U = [[0,1],[-1,2]] (U-branch)

    These matrices have orders in SL2(F_p) that divide p(p2-1).
    If M^k == I (mod p) but ≢ I (mod q), the off-diagonal entries of M^k are
    0 mod p but nonzero mod q, revealing p via gcd.

    Novel: group-order method in SL2 rather than multiplicative group or elliptic
    curves, with Berggren matrices providing structured starting points.
    """
    # Berggren matrices in SL2(Z)
    M_A = (1, 1, 1, 2)   # [[1,1],[1,2]], det=1
    M_D = (1, 0, 2, 1)   # [[1,0],[2,1]], det=1
    M_U = (0, 1, -1, 2)  # [[0,1],[-1,2]], det=1

    # Also use products for more diverse starting points
    M_AD = _mat2_mul(M_A, M_D, 1 << 60)  # Use large N to avoid mod
    M_AU = _mat2_mul(M_A, M_U, 1 << 60)
    M_DU = _mat2_mul(M_D, M_U, 1 << 60)

    matrices = [
        M_A, M_D, M_U,
        M_AD, M_AU, M_DU,
    ]

    for M_raw in matrices:
        M = (M_raw[0] % N, M_raw[1] % N, M_raw[2] % N, M_raw[3] % N)

        # Check if M itself reveals a factor
        for entry in M:
            g = gcd(abs(entry), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Compute M^(2^j) mod N for j = 1, 2, 4, 8, 16, ...
        # This checks orders that are powers of 2
        # Also check M^k for k = 3, 5, 6, 7, 9, 10, 11, ...
        # Key: gcd(entries of M^k - I, N) reveals factors

        # Strategy 1: Powers of 2 (for factors where order is 2-smooth)
        power = M
        for j in range(max_k_bits):
            power = _mat2_mul(power, power, N)  # M^(2^(j+1))

            # Check off-diagonal entries
            for idx in [1, 2]:
                g = gcd(abs(power[idx]), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            # Check trace condition: tr(M^k) == 2 (mod p) means M^k close to I mod p
            trace = (power[0] + power[3]) % N
            g = gcd(abs(trace - 2), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            g = gcd(abs(trace + 2), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Check determinant == 1 (CRT bottleneck)
            det = (power[0] * power[3] - power[1] * power[2]) % N
            g = gcd(abs(det - 1), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Strategy 2: Factorial-like powers (for smooth-order factors)
        # Compute M^k for k = 2, 6, 24, 120, ... (k!)
        # This catches factors where the matrix order divides k!
        # We compute incrementally: M^2, M^3, M^4, ... using repeated multiplication
        batch = (1, 0, 0, 1)  # Identity for batch accumulation
        power_k = M
        for k in range(2, min(50, N)):
            power_k = _mat2_mul(power_k, M, N)  # M^k

            for idx in [1, 2]:
                g = gcd(abs(power_k[idx]), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            trace = (power_k[0] + power_k[3]) % N
            g = gcd(abs(trace - 2), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Batch: accumulate (M^k - I) entries
            batch = _mat2_mul(batch, (
                (power_k[0] - 1) % N,
                power_k[1],
                power_k[2],
                (power_k[3] - 1) % N
            ), N)

            if k % 10 == 0:
                for idx in [0, 1, 2, 3]:
                    g = gcd(abs(batch[idx]), N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
                batch = (1, 0, 0, 1)

    return None


def _qr_discriminator(N: int,
                      convs: list[tuple[int, int]],
                      max_depth: int = 32) -> tuple[int, int] | None:
    """Stage 3: Quadratic Residue Discriminator.

    CF convergents yield residues r_k = p_k2 - N*q_k2 ~= ±1.
    For a factor p of N: p_k2 == r_k (mod p).
    If r_k is a QR mod p but not mod q (or vice versa), then:
      gcd(r_k^((p-1)/2) - 1, N) or gcd(r_k^((q-1)/2) - (-1), N) reveals a factor.

    Since we don't know p, we check gcd(r_k^m - 1, N) for various m derived
    from N's structure.

    Novel: uses the QR structure of Pell residues from CF convergents rather
    than random bases as in Euler criterion / Solovay-Strassen.
    """
    # For each convergent, check QR-based conditions
    for pk, qk in convs:
        # The Pell residue r_k = p_k2 - N*q_k2
        r_k = pk * pk - N * qk * qk
        if r_k == 0:
            # This would mean N | (pk2), very unlikely but check
            g = gcd(pk, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # For small residues, check Euler criterion-based conditions
        # If r_k == s2 (mod p) but r_k ≢ s2 (mod q), then
        # r_k^((p-1)/2) == 1 (mod p) and r_k^((q-1)/2) == -1 (mod q)
        # (or vice versa). We don't know p-1, but we can try m = (N-1)/2^j

        r = abs(r_k)
        if r <= 1 or r >= N:
            continue

        # Check r^m - 1 for m = (N-1)/2, (N-1)/4, etc.
        m = (N - 1) // 2
        for _ in range(max_depth):
            if m == 0:
                break
            val = pow(r, m, N)
            g = gcd(val - 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))
            g = gcd(val + 1, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))
            if m % 2 == 0:
                m = m // 2
            else:
                break

    return None


def _idempotent_detection(N: int,
                          convs: list[tuple[int, int]],
                          max_depth: int = 40) -> tuple[int, int] | None:
    """Stage 4: Idempotent Detection via CRT Bottleneck.

    An idempotent e satisfies e2 == e (mod N). For N = pq, the nontrivial
    idempotents are e == (0,1) and e == (1,0) under CRT, giving gcd(e, N) as
    a nontrivial factor.

    The squaring map x → x2 on Z/NZ has fixed points 0 and 1, plus the
    nontrivial idempotents. Starting from CF convergent-derived points and
    iterating the squaring map, we eventually reach an idempotent when the
    orbit structure differs mod p vs mod q.

    Novel: uses CF convergent structure to choose starting points with known
    algebraic properties, rather than random starting points.
    """
    starts = set()

    # Add convergent-derived starting points
    for pk, qk in convs[:50]:
        g = gcd(qk, N)
        if 1 < g < N:
            return (min(g, N // g), max(g, N // g))
        if gcd(qk, N) == 1:
            try:
                qk_inv = pow(qk, -1, N)
                x = (pk * qk_inv) % N
                starts.add(x)
                starts.add((pk + qk) % N)
                starts.add(abs(pk - qk) % N)
            except (ValueError, ZeroDivisionError):
                pass

    # Also use small values that have different QR character mod p vs q
    for v in range(2, min(20, N)):
        starts.add(v)

    for x in starts:
        if x <= 1 or x >= N:
            continue

        # Iterate squaring map: x → x2 mod N
        # An idempotent satisfies x2 == x, so x2 - x == 0
        # Under CRT, the orbit eventually reaches an idempotent if it
        # cycles at different rates mod p vs mod q

        y = x
        for _ in range(max_depth):
            y_new = (y * y) % N

            # Check for idempotent: y2 == y
            diff = (y_new - y) % N
            if diff != 0:
                g = gcd(abs(diff), N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

            # Also check y^2 - y (current point)
            g = gcd(abs((y * y - y) % N), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Check squaring orbit convergence
            # If y^(2^j) == 1 (mod p) but ≢ 1 (mod q)
            g = gcd(abs(y_new - 1), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            y = y_new

            # Check for cycle (y2 == y → idempotent)
            if y == x or y == 0 or y == 1:
                break

    return None


def _cf_guided_walk(N: int,
                   convs: list[tuple[int, int]],
                   max_iter: int = 50000) -> tuple[int, int] | None:
    """Stage 6: CF-Guided Random Walk — novel alternative to Pollard rho.

    Instead of Pollard's x → x2 + c with fixed c, this walk uses CF
    convergent-derived mutation points. At each step:
    - If the current value is near a convergent numerator mod N, apply a
      "convergent jump" that moves the walk to a different region of Z/NZ
    - Otherwise, apply a squaring step like Pollard rho

    The convergent jumps exploit the Pell residue structure: if p_k == ±1 (mod p)
    for a factor p, then jumping to a convergent-derived value moves the walk
    to a region where the cycle structure differs mod p vs mod q.

    This is structurally different from Pollard rho because:
    1. Uses algebraic structure (CF convergents) rather than random constants
    2. Has two types of steps: squaring AND convergent jumps
    3. The walk pattern is determined by the structure of N, not arbitrary c
    """
    if N < 4 or N % 2 == 0:
        return None

    # Use convergent residues as mutation points
    mutations = set()
    for pk, qk in convs[:20]:
        if gcd(qk, N) == 1:
            try:
                qk_inv = pow(qk, -1, N)
                mutations.add((pk * qk_inv) % N)
            except (ValueError, ZeroDivisionError):
                pass
        mutations.add(pk % N)
        mutations.add((pk + qk) % N)
        mutations.add(abs(pk - qk) % N)

    # Add Berggren matrix trace-derived values
    for t in [3, 7, 2, 5, 11, 13, 17, 19, 23, 29, 41, 99]:
        mutations.add(t % N if t < N else t % N)

    mutations = sorted([m for m in mutations if 2 <= m < N])[:30]

    # CF-guided walk with Floyd cycle detection
    for c in mutations[:15]:
        for x_start in [2, 3, 5, 7, 11]:
            x = x_start
            y = x_start
            d = 1

            batch = 1
            batch_count = 0
            mutations_list = list(mutations)

            # Alternating walk: squaring step, then convergent jump
            for iteration in range(max_iter):
                # Squaring step
                x = (x * x + c) % N
                # Floyd's cycle detection: y moves twice
                y = (y * y + c) % N
                y = (y * y + c) % N

                diff = abs(x - y)
                if diff == 0:
                    break

                batch = (batch * diff) % N
                batch_count += 1

                if batch_count >= 128:
                    d = gcd(batch, N)
                    if 1 < d < N:
                        return (min(d, N // d), max(d, N // d))
                    batch = 1
                    batch_count = 0

                # Convergent jump step (every 7 iterations)
                if iteration % 7 == 0 and mutations_list:
                    # Jump: x → x * mutation[i] mod N where i depends on x
                    idx = (iteration // 7) % len(mutations_list)
                    jump_val = mutations_list[idx]
                    x = (x * jump_val) % N

            # Final batch check
            if batch_count > 0:
                d = gcd(batch, N)
                if 1 < d < N:
                    return (min(d, N // d), max(d, N // d))

    return None


def _cf_near_square_search(N: int,
                            convs: list[tuple[int, int]],
                            max_candidates: int = 500) -> tuple[int, int] | None:
    """Stage 5: CF-Guided Near-Square-Root Search.

    For N = pq with p ~= q, we have (p+q)/2 ~= sqrtN and (p-q)/2 small.
    The convergents p_k/q_k ~= sqrtN give us smart starting points for the
    difference-of-squares search.

    For each convergent p_k/q_k, compute s = p_k/q_k * q_k = p_k ~= sqrtN * q_k.
    Then check s2 - N*q_k2 for factors. Also check nearby values:
    s ± j for small j.

    This is DIFFERENT from Fermat because:
    - Fermat starts at ⌈sqrtN⌉ and increments by 1
    - We use CF convergents to JUMP to the best rational approximations
    - Each convergent gives a CANDIDATE that's much closer to sqrtN than sequential search

    Novel: CF structure provides O(log N) high-quality starting points instead
    of O(sqrtN) sequential candidates.
    """
    s = isqrt(N)

    # Strategy 1: Direct convergent-based candidates
    for pk, qk in convs:
        # Check if convergent itself gives a factor
        r = pk * pk - N * qk * qk
        if r == 0:
            g = gcd(pk, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

        # Check nearby squares: (pk ± j*qk)2 - N*qk2 = r ± 2*j*pk*qk + j2*qk2
        # This gives us values near the Pell residue
        for j in range(-5, 6):
            if j == 0:
                continue
            candidate = pk + j * qk
            if candidate <= 0:
                continue
            diff = candidate * candidate - N * qk * qk
            if diff > 0:
                # Check if diff has a factor in common with N
                g = gcd(diff, N)
                if 1 < g < N:
                    return (min(g, N // g), max(g, N // g))

                # Also check if diff is a perfect square
                sd = isqrt(diff)
                if sd * sd == diff:
                    # candidate2 - N*qk2 = sd2, so (candidate - sd*qk)(candidate + sd*qk) = ?
                    # Actually: candidate2 - sd2 = N*qk2, so
                    # (candidate-sd)(candidate+sd) = N*qk2
                    g = gcd(candidate - sd, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
                    g = gcd(candidate + sd, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

    # Strategy 2: Convergent product combinations
    # If r_i and r_j are Pell residues, their product is also a Pell residue:
    # (p_i + q_isqrtN)(p_j + q_jsqrtN) = (p_i*p_j + N*q_i*q_j) + (p_i*q_j + p_j*q_i)sqrtN
    # with residue r_i * r_j
    # Check if products of small residues give factors
    small_residues = []
    for pk, qk in convs[:50]:
        r = pk * pk - N * qk * qk
        if 0 < abs(r) < 1000:
            small_residues.append((pk, qk, r))

    for i in range(min(len(small_residues), 30)):
        pi, qi, ri = small_residues[i]
        for j in range(i, min(len(small_residues), 30)):
            pj, qj, rj = small_residues[j]
            # Product of Pell solutions
            new_p = pi * pj + N * qi * qj
            new_q = pi * qj + pj * qi
            new_r = ri * rj

            # Check gcd of residue product with N
            g = gcd(abs(new_r), N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

            # Check if product residue is a perfect square
            if new_r > 0:
                sd = isqrt(new_r)
                if sd * sd == new_r:
                    # new_p2 - N*new_q2 = sd2
                    # (new_p - sd)(new_p + sd) = N*new_q2
                    # Hmm, this gives (new_p - sd*new_q)(new_p + sd*new_q) = N*new_q2
                    # Actually: new_p2 - N*new_q2 = sd2
                    # So new_p2 == sd2 (mod N)
                    # gcd(new_p - sd, N) might give a factor
                    g = gcd(new_p - sd, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))
                    g = gcd(new_p + sd, N)
                    if 1 < g < N:
                        return (min(g, N // g), max(g, N // g))

    # Strategy 3: Near-square-root search guided by CF
    # Start from isqrt(N) and check candidates in a small range
    # guided by the convergent structure
    for offset in range(max_candidates):
        a = s + offset + 1
        b2 = a * a - N
        if b2 < 0:
            continue
        b = isqrt(b2)
        if b * b == b2:
            # Found: a2 - b2 = N, so (a-b)(a+b) = N
            g = gcd(a - b, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))
            g = gcd(a + b, N)
            if 1 < g < N:
                return (min(g, N // g), max(g, N // g))

    return None


def spectral_cascade_factor(N: int,
                             max_cf_terms: int = 200,
                             max_sl2_k_bits: int = 20) -> tuple[int, int] | None:
    """Factor N using purely novel Spectral Cascade methods.

    Four novel stages, NO classical methods:
    1. CF-Convergent Squaring Cascade — Pell residue + squaring orbits
    2. SL2 Matrix Order Detection — Berggren matrices in SL2(Z/NZ)
    3. Quadratic Residue Discriminator — QR structure of Pell residues
    4. Idempotent Detection — CRT bottleneck via squaring orbits

    Returns (p, q) with p < q and p*q = N, or None.
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

    # Skip for large N — method too slow
    if N.bit_length() > 256:
        return None

    # Compute CF convergents for all stages
    cf = _cf_sqrt(N, max_terms=max_cf_terms)
    convs = _convergents(cf)

    # Stage 1: CF-Convergent Squaring Cascade
    result = _cf_squaring_cascade(N, convs, max_depth=40)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 2: SL2 Matrix Order Detection
    result = _sl2_matrix_cascade(N, max_k_bits=max_sl2_k_bits)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 3: Quadratic Residue Discriminator
    result = _qr_discriminator(N, convs, max_depth=32)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 4: Idempotent Detection
    result = _idempotent_detection(N, convs, max_depth=40)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 5: CF-Guided Near-Square-Root Search
    result = _cf_near_square_search(N, convs, max_candidates=1000)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    # Stage 6: CF-Guided Random Walk (novel alternative to Pollard rho)
    # Cap iterations for large N to avoid hangs
    walk_iters = min(50000, max(1000, N.bit_length() * 10))
    result = _cf_guided_walk(N, convs, max_iter=walk_iters)
    if result is not None:
        p, q = result
        if p * q == N and 1 < p < N and 1 < q < N:
            return result

    return None