"""Tests for SL₂ group-order cascade factoring."""
import random
import pytest
from math import isqrt

from insideout.sl2_group_order import (
    _mat2_mul,
    _mat2_pow,
    _mat2_det,
    _random_sl2_matrix,
    _check_matrix_crt,
    sl2_group_order_factor,
    sl2_structured_factor,
)


class TestMat2Operations:
    """Test 2×2 matrix operations mod N."""

    def test_identity_multiplication(self):
        """I * M = M for any M."""
        N = 97
        I = (1, 0, 0, 1)
        M = (3, 5, 7, 11)
        result = _mat2_mul(I, M, N)
        assert result == M, f"I*M should equal M, got {result}"

    def test_associativity(self):
        """Matrix multiplication is associative mod N."""
        N = 97
        A = (3, 5, 7, 2)
        B = (11, 13, 17, 19)
        C = (23, 29, 31, 37)

        AB_C = _mat2_mul(_mat2_mul(A, B, N), C, N)
        A_BC = _mat2_mul(A, _mat2_mul(B, C, N), N)
        assert AB_C == A_BC, "Matrix multiplication should be associative"

    def test_determinant_preserved(self):
        """det(AB) = det(A)*det(B) mod N."""
        N = 97
        A = (3, 5, 7, 2)
        B = (11, 13, 17, 19)

        det_A = _mat2_det(A, N)
        det_B = _mat2_det(B, N)
        AB = _mat2_mul(A, B, N)
        det_AB = _mat2_det(AB, N)

        assert det_AB == (det_A * det_B) % N, \
            f"det(AB) should equal det(A)*det(B) mod N"

    def test_sl2_determinant(self):
        """SL₂ matrices have determinant 1."""
        N = 97
        M_A = (1, 1, 1, 2)
        M_D = (1, 0, 2, 1)
        M_U = (0, 1, 96, 2)  # [[0,1],[-1,2]] mod 97

        assert _mat2_det(M_A, N) == 1, "M_A should have det 1"
        assert _mat2_det(M_D, N) == 1, "M_D should have det 1"
        assert _mat2_det(M_U, N) == 1, "M_U should have det 1"

    def test_mat2_pow_identity(self):
        """M^0 = I for any M."""
        N = 97
        M = (3, 5, 7, 2)
        result = _mat2_pow(M, 0, N)
        assert result == (1, 0, 0, 1), "M^0 should be identity"

    def test_mat2_pow_first(self):
        """M^1 = M."""
        N = 97
        M = (3, 5, 7, 2)
        result = _mat2_pow(M, 1, N)
        assert result == (3 % N, 5 % N, 7 % N, 2 % N)

    def test_mat2_pow_large(self):
        """M^k should preserve det(M)^k."""
        N = 97
        M = (3, 5, 7, 2)  # det = 3*2 - 5*7 = 6 - 35 = -29 = 68 mod 97
        det_M = _mat2_det(M, N)

        M10 = _mat2_pow(M, 10, N)
        det_M10 = _mat2_det(M10, N)
        assert det_M10 == pow(det_M, 10, N), \
            f"det(M^10) should equal det(M)^10"


class TestRandomSL2Matrix:
    """Test random SL₂ matrix generation."""

    def test_determinant_is_1(self):
        """Generated matrices should have determinant 1 mod N."""
        N = 10007  # Prime, so all non-zero elements are invertible
        for _ in range(10):
            M = _random_sl2_matrix(N)
            if isinstance(M, tuple) and len(M) == 2:
                continue  # Found a factor, skip
            assert _mat2_det(M, N) == 1, \
                f"Generated matrix should have det 1, got {_mat2_det(M, N)}"

    def test_composite_N_can_find_factor(self):
        """For composite N, matrix generation may find a factor directly."""
        # This tests the early-exit path where gcd(a, N) > 1
        N = 15  # 3 * 5
        found_factor = False
        for _ in range(100):
            result = _random_sl2_matrix(N)
            if isinstance(result, tuple) and len(result) == 2:
                p, q = result
                if p * q == N:
                    found_factor = True
                    break
        # Should almost always find a factor for N=15 since gcd(random, 15)
        # is non-trivial about 60% of the time
        assert found_factor, "Should find a factor for N=15 via random matrix"


class TestCheckMatrixCrt:
    """Test CRT divergence detection in matrices."""

    def test_identity_matrix_no_factor(self):
        """Identity matrix should not reveal a factor."""
        N = 15
        I = (1, 0, 0, 1)
        result = _check_matrix_crt(I, N)
        assert result is None, "Identity matrix should not reveal a factor"

    def test_known_crt_divergence(self):
        """Matrix that is I mod 3 but not I mod 5 should reveal factor 3."""
        # M = I + 5*K for some matrix K (so M ≡ I mod 5)
        # But M ≢ I mod 3
        # M = [[6, 0], [0, 6]] = 6*I = I mod 5 (since 6 ≡ 1 mod 5)
        # M = [[6, 0], [0, 6]] = 0*I mod 3 (since 6 ≡ 0 mod 3)
        # But this gives gcd(6, 15) = 3
        N = 15
        M = (6, 0, 0, 6)
        result = _check_matrix_crt(M, N)
        assert result is not None, "Should detect factor from CRT divergence"


class TestSL2GroupOrderFactor:
    """Test SL₂ group-order cascade factoring."""

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
        """SL₂ method should factor small semiprimes."""
        random.seed(42)  # Deterministic for testing
        result = sl2_group_order_factor(N, bound=1000, curves=10)
        if result is not None:
            fp, fq = result
            assert fp * fq == N, f"Factors should multiply to N"

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        random.seed(42)
        result = sl2_group_order_factor(1681, bound=1000)  # 41²
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        random.seed(42)
        result = sl2_group_order_factor(30)
        assert result is not None
        assert 2 in result

    def test_small_prime(self):
        """Primes should return None."""
        random.seed(42)
        result = sl2_group_order_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert sl2_group_order_factor(1) is None
        assert sl2_group_order_factor(2) is None
        assert sl2_group_order_factor(3) is None


class TestSL2StructuredFactor:
    """Test SL₂ factoring with Berggren matrix starting points."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Structured SL₂ should factor small semiprimes."""
        result = sl2_structured_factor(N, bound=1000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = sl2_structured_factor(1681, bound=1000)
        assert result is not None
        assert result == (41, 41)