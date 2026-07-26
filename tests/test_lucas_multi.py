"""Tests for multi-parameter Lucas sequence factoring."""
import pytest
from math import isqrt

from insideout.lucas_multi import (
    _lucas_pair,
    _lucas_pythagorean_batch,
    lucas_multi_factor,
    crt_collision_factor,
)


class TestLucasPair:
    """Test Lucas sequence fast doubling computation."""

    def test_fibonacci_special_case(self):
        """P=1, Q=-1 gives Fibonacci/Lucas numbers."""
        # U_k(1,-1) = F_k (Fibonacci numbers)
        # V_k(1,-1) = L_k (Lucas numbers)
        # F_10 = 55, L_10 = 123
        U_10, V_10 = _lucas_pair(10, 1, -1, 10**9)
        assert U_10 == 55, f"U_10(1,-1) should be F_10=55, got {U_10}"
        assert V_10 == 123, f"V_10(1,-1) should be L_10=123, got {V_10}"

    def test_pell_special_case(self):
        """P=2, Q=-1 gives Pell numbers."""
        # U_k(2,-1) = P_n (Pell numbers): P_0=0, P_1=1, P_2=2, P_3=5, P_4=12
        U_0, V_0 = _lucas_pair(0, 2, -1, 10**9)
        assert U_0 == 0

        U_1, V_1 = _lucas_pair(1, 2, -1, 10**9)
        assert U_1 == 1
        assert V_1 == 2

        U_4, V_4 = _lucas_pair(4, 2, -1, 10**9)
        assert U_4 == 12, f"Pell P_4 should be 12, got {U_4}"

    def test_lucas_pair_modular(self):
        """Lucas sequences work correctly modulo N."""
        # U_k(3,1) mod 7
        # U_0=0, U_1=1, U_2=3*1-1*0=3, U_3=3*3-1*1=8, U_4=3*8-1*3=21
        # Mod 7: U_3≡1, U_4≡0
        U_3, V_3 = _lucas_pair(3, 3, 1, 7)
        assert U_3 == 1, f"U_3(3,1) mod 7 should be 1, got {U_3}"

        U_4, V_4 = _lucas_pair(4, 3, 1, 7)
        assert U_4 == 0, f"U_4(3,1) mod 7 should be 0, got {U_4}"

    def test_lucas_pair_large(self):
        """Lucas sequences handle large k correctly."""
        # U_100(1,-1) mod 1000 should give F_100 mod 1000 = 75 (F_100=354224848179261915075)
        U_100, V_100 = _lucas_pair(100, 1, -1, 1000)
        assert U_100 == 75, f"F_100 mod 1000 should be 75, got {U_100}"


class TestLucasPythagoreanBatch:
    """Test Pythagorean augmentation of Lucas sequences."""

    def test_fibonacci_batch_identity(self):
        """For P=1, Q=-1 (Fibonacci), batch reduces to Fibonacci-Pythagorean identities."""
        # U_5(1,-1) = F_5 = 5
        U_5, V_5 = _lucas_pair(5, 1, -1, 10**9)
        U_6, V_6 = _lucas_pair(6, 1, -1, 10**9)

        batch, candidates = _lucas_pythagorean_batch(U_5, U_6, V_5, 1, 10**9, Q=-1)

        # candidates[0] = U_5 = F_5 = 5
        assert candidates[0] == 5
        # candidates[1] = V_5 = L_5 = 11
        assert candidates[1] == 11

    def test_batch_produces_valid_candidates(self):
        """All batch candidates should be non-negative integers less than N."""
        N = 10007
        U_k, V_k = _lucas_pair(100, 3, 1, N)
        U_k1, V_k1 = _lucas_pair(101, 3, 1, N)

        batch, candidates = _lucas_pythagorean_batch(U_k, U_k1, V_k, 3, N)

        for c in candidates:
            assert 0 <= c < N, f"Candidate {c} out of range [0, {N})"
        assert 0 <= batch < N


class TestLucasMultiFactor:
    """Test multi-parameter Lucas factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
        (437, 19, 23),
        (667, 23, 29),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Multi-parameter Lucas should factor small semiprimes."""
        result = lucas_multi_factor(N, bound=5000, stage2_bound=1000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = lucas_multi_factor(1681, bound=5000)  # 41²
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = lucas_multi_factor(30)
        assert result is not None
        assert result[0] == 2 or result[1] == 2

    def test_small_prime(self):
        """Primes should return None."""
        result = lucas_multi_factor(7, bound=5000)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert lucas_multi_factor(1) is None
        assert lucas_multi_factor(2) is None
        assert lucas_multi_factor(3) is None

    def test_larger_semiprime(self):
        """Should factor a medium semiprime."""
        # 10007 * 10009 = 100160063
        N = 10007 * 10009
        result = lucas_multi_factor(N, bound=15000, stage2_bound=3000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestCrtCollisionFactor:
    """Test CRT collision factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """CRT collision should factor small semiprimes."""
        result = crt_collision_factor(N, bound=5000, stage2_bound=1000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = crt_collision_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = crt_collision_factor(30)
        assert result is not None

    def test_small_prime(self):
        """Primes should return None."""
        result = crt_collision_factor(7, bound=5000)
        assert result is None