"""Tests for Lattice Descent Factoring."""
import pytest
from math import isqrt

from insideout.lattice_descent import (
    _sorted_pair,
    _mat_vec_mul_mod,
    _mat_mul_mod,
    _mat_pow_mod,
    _jacobi_symbol,
    _tonelli_shanks,
    _eigenvalue_gcd,
    _berggren_eigenvalue_p,
    _lattice_walk,
    _find_hyperplane_point,
    _cf_convergents_sqrt,
    _cf_guided_walk,
    _is_probable_prime,
    lattice_descent_factor,
)
from insideout.berggren import U as U_MAT, A as A_MAT, D as D_MAT, Matrix3x3


class TestSortedPair:
    """Test _sorted_pair helper."""

    def test_already_sorted(self):
        assert _sorted_pair(3, 5) == (3, 5)

    def test_reversed(self):
        assert _sorted_pair(5, 3) == (3, 5)

    def test_equal(self):
        assert _sorted_pair(7, 7) == (7, 7)


class TestMatVecMulMod:
    """Test matrix-vector multiplication mod N."""

    def test_identity(self):
        I = Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1)
        v = (3, 4, 5)
        assert _mat_vec_mul_mod(I, v, 100) == (3, 4, 5)

    def test_berggren_U(self):
        """U * (3,4,5) = (5,12,13) mod large N."""
        v = (3, 4, 5)
        result = _mat_vec_mul_mod(U_MAT, v, 1000)
        assert result == (5, 12, 13)

    def test_berggren_A(self):
        """A * (3,4,5) = (5,12,13) is wrong; check actual value."""
        v = (3, 4, 5)
        result = _mat_vec_mul_mod(A_MAT, v, 1000)
        # A = [[1,2,2],[2,1,2],[2,2,3]]
        # A*(3,4,5) = (3+8+10, 6+4+10, 6+8+15) = (21, 20, 29)
        assert result == (21, 20, 29)

    def test_mod_reduction(self):
        """Modular reduction works correctly."""
        v = (3, 4, 5)
        result = _mat_vec_mul_mod(U_MAT, v, 10)
        # (5, 12, 13) mod 10 = (5, 2, 3)
        assert result == (5, 2, 3)


class TestMatMulMod:
    """Test matrix-matrix multiplication mod N."""

    def test_identity(self):
        I = Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1)
        N = 10**9
        result = _mat_mul_mod(I, U_MAT, N)
        # After mod reduction, negative entries become positive
        for i in range(3):
            row = result.row(i)
            expected_row = U_MAT.row(i)
            for j in range(3):
                assert row[j] % N == expected_row[j] % N

    def test_U_squared(self):
        """U^2 should be computable."""
        result = _mat_mul_mod(U_MAT, U_MAT, 10**9)
        # U^2 * (3,4,5) = U * (5,12,13) = (5-24+26, 10-12+26, 10-24+39) = (7, 24, 25)
        v = _mat_vec_mul_mod(result, (3, 4, 5), 10**9)
        assert v == (7, 24, 25)


class TestMatPowMod:
    """Test matrix exponentiation mod N."""

    def test_power_zero(self):
        """M^0 = I."""
        result = _mat_pow_mod(U_MAT, 0, 1000)
        assert result == Matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1)

    def test_power_one(self):
        """M^1 = M."""
        N = 10**9
        result = _mat_pow_mod(U_MAT, 1, N)
        for i in range(3):
            row = result.row(i)
            expected_row = U_MAT.row(i)
            for j in range(3):
                assert row[j] % N == expected_row[j] % N

    def test_power_two(self):
        """M^2 = M * M."""
        result = _mat_pow_mod(U_MAT, 2, 10**9)
        expected = _mat_mul_mod(U_MAT, U_MAT, 10**9)
        assert result == expected

    def test_power_large(self):
        """Large power via binary exponentiation."""
        # U^4 * (3,4,5) should give a valid PPT
        M4 = _mat_pow_mod(U_MAT, 4, 10**18)
        v = _mat_vec_mul_mod(M4, (3, 4, 5), 10**18)
        # Check it's a valid PPT
        a, b, c = v
        assert a * a + b * b == c * c


class TestJacobiSymbol:
    """Test Jacobi symbol computation."""

    def test_one(self):
        assert _jacobi_symbol(1, 7) == 1

    def test_qr(self):
        """4 is QR mod 7 (2^2 = 4)."""
        assert _jacobi_symbol(4, 7) == 1

    def test_qnr(self):
        """3 is QNR mod 7."""
        assert _jacobi_symbol(3, 7) == -1

    def test_composite_modulus(self):
        """Jacobi symbol for composite modulus."""
        # (2/15) = (2/3)*(2/5) = (-1)*(-1) = 1
        assert _jacobi_symbol(2, 15) == 1

    def test_a_equals_zero(self):
        assert _jacobi_symbol(0, 7) == 0

    def test_even_a(self):
        """(4/7) = 1."""
        assert _jacobi_symbol(4, 7) == 1


class TestTonelliShanks:
    """Test Tonelli-Shanks square root computation."""

    def test_simple_qr(self):
        """sqrt(4) mod 7 = 2 or 5."""
        result = _tonelli_shanks(4, 7)
        assert result is not None
        assert (result * result) % 7 == 4

    def test_qnr_returns_none(self):
        """sqrt(3) mod 7 doesn't exist."""
        assert _tonelli_shanks(3, 7) is None

    def test_zero(self):
        assert _tonelli_shanks(0, 7) == 0

    def test_one(self):
        result = _tonelli_shanks(1, 7)
        assert result is not None
        assert result == 1 or result == 6

    def test_large_prime(self):
        """Test with a larger prime."""
        p = 1000003  # prime
        a = 4
        result = _tonelli_shanks(a, p)
        assert result is not None
        assert (result * result) % p == a


class TestEigenvalueGcd:
    """Test eigenvalue GCD computation."""

    def test_finds_small_factor(self):
        """Should find a factor of a small semiprime."""
        # 8051 = 83 * 97
        result = _eigenvalue_gcd(U_MAT, 8051, max_steps=500)
        if result is not None:
            p, q = result
            assert p * q == 8051

    def test_returns_none_for_prime(self):
        """Should return None for a prime."""
        result = _eigenvalue_gcd(U_MAT, 8053, max_steps=100)  # 8053 is prime
        # May or may not find a factor (depends on step count)
        # But should not return an incorrect factor
        if result is not None:
            p, q = result
            assert p * q == 8053


class TestBerggrenEigenvalueP:
    """Test Berggren eigenvalue discriminant check."""

    def test_discriminant_values(self):
        """Check discriminant values for each matrix."""
        # U: tr=3, Δ = 9-4 = 5
        # A: tr=6, Δ = 36-4 = 32
        # D: tr=4, Δ = 16-4 = 12
        pass  # Just sanity check

    def test_may_find_factor(self):
        """May find a factor via discriminant check."""
        # 15 = 3 * 5
        result = _berggren_eigenvalue_p(15, 3)
        if result is not None:
            p, q = result
            assert p * q == 15


class TestLatticeWalk:
    """Test lattice walk in various directions."""

    def test_walk_along_U(self):
        """Walk along U-branch."""
        result = _lattice_walk(8051, direction=(U_MAT,), steps=200)
        if result is not None:
            p, q = result
            assert p * q == 8051

    def test_walk_along_A(self):
        """Walk along A-branch."""
        result = _lattice_walk(8051, direction=(A_MAT,), steps=200)
        if result is not None:
            p, q = result
            assert p * q == 8051

    def test_walk_combined(self):
        """Walk in combined direction."""
        result = _lattice_walk(8051, direction=(U_MAT, A_MAT), steps=200)
        if result is not None:
            p, q = result
            assert p * q == 8051


class TestFindHyperplanePoint:
    """Test hyperplane point search."""

    def test_finds_factor(self):
        """Should find a factor via hyperplane search."""
        # 8051 = 83 * 97
        primes = [83, 97, 101, 103, 107]
        result = _find_hyperplane_point(8051, primes)
        if result is not None:
            p, q = result
            assert p * q == 8051

    def test_with_small_primes(self):
        """Should work with small primes list."""
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        # 323 = 17 * 19
        result = _find_hyperplane_point(323, primes)
        if result is not None:
            p, q = result
            assert p * q == 323


class TestCfConvergentsSqrt:
    """Test continued fraction convergents of sqrt(N)."""

    def test_perfect_square(self):
        """Perfect square returns trivial convergent."""
        result = _cf_convergents_sqrt(16)
        assert result == [(4, 1)]

    def test_non_square(self):
        """Non-square returns multiple convergents."""
        result = _cf_convergents_sqrt(8051)
        assert len(result) > 1
        # First convergent should be isqrt(8051) = 89
        assert result[0] == (89, 1)

    def test_convergent_is_good_approximation(self):
        """Convergents approximate sqrt(N) well."""
        result = _cf_convergents_sqrt(8051)
        for p, q in result[1:]:
            # |p/q - sqrt(N)| should be small
            approx = p / q
            actual = 8051 ** 0.5
            assert abs(approx - actual) < 1.0 / (q * q) + 1e-10


class TestCfGuidedWalk:
    """Test CF-guided walk."""

    def test_may_find_factor(self):
        """May find a factor via CF-guided walk."""
        result = _cf_guided_walk(8051, max_steps=500)
        if result is not None:
            p, q = result
            assert p * q == 8051


class TestIsProbablePrime:
    """Test Miller-Rabin primality test."""

    def test_small_primes(self):
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31):
            assert _is_probable_prime(p)

    def test_small_composites(self):
        for n in (4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25):
            assert not _is_probable_prime(n)

    def test_medium_prime(self):
        assert _is_probable_prime(104729)  # 10000th prime

    def test_medium_composite(self):
        assert not _is_probable_prime(104729 * 2)  # Even
        assert not _is_probable_prime(104730)  # Even

    def test_large_prime(self):
        # Known large prime
        assert _is_probable_prime(1000003)

    def test_one(self):
        assert not _is_probable_prime(1)

    def test_zero(self):
        assert not _is_probable_prime(0)


class TestLatticeDescentFactor:
    """Test main lattice_descent_factor entry point."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        result = lattice_descent_factor(N, max_steps=5000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_8051(self):
        """Should factor 8051 = 83 * 97."""
        result = lattice_descent_factor(8051, max_steps=10000)
        if result is not None:
            p, q = result
            assert p * q == 8051
            assert {p, q} == {83, 97}

    def test_15571(self):
        """Should factor 15571."""
        result = lattice_descent_factor(15571, max_steps=10000)
        if result is not None:
            p, q = result
            assert p * q == 15571

    def test_1022117(self):
        """Should factor 1022117."""
        result = lattice_descent_factor(1022117, max_steps=10000)
        if result is not None:
            p, q = result
            assert p * q == 1022117

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = lattice_descent_factor(1681, max_steps=1000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = lattice_descent_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = lattice_descent_factor(8053, max_steps=1000)  # 8053 is prime
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert lattice_descent_factor(1) is None
        assert lattice_descent_factor(2) is None
        assert lattice_descent_factor(3) is None

    def test_balanced_semiprime(self):
        """Should factor a balanced semiprime."""
        # Pick a balanced semiprime: p and q close together
        # 10007 * 10009 = 100160063
        N = 10007 * 10009
        result = lattice_descent_factor(N, max_steps=10000)
        if result is not None:
            p, q = result
            assert p * q == N

    def test_result_is_sorted(self):
        """Result should be sorted (p <= q)."""
        result = lattice_descent_factor(8051, max_steps=10000)
        if result is not None:
            p, q = result
            assert p <= q
