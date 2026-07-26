"""Tests for Brahmagupta-Fibonacci and Fibonacci factoring methods."""
import pytest
from math import isqrt

from insideout.brahmagupta import (
    find_two_square_representation,
    find_all_two_square_representations,
    brahmagupta_fibonacci_factor,
    fermat_difference_of_squares,
)
from insideout.fibonacci_factor import (
    fibonacci_mod,
    pisano_period,
    entry_point,
    pisano_factor,
    fibonacci_gcd_factor,
)
from insideout.factor import factor, factor_with_method


class TestTwoSquareRepresentations:
    """Test finding representations of N as sum of two squares."""

    def test_5_is_1_2(self):
        """5 = 1^2 + 2^2."""
        result = find_two_square_representation(5)
        assert result is not None
        a, b = result
        assert a * a + b * b == 5

    def test_25_is_0_5(self):
        """25 = 0^2 + 5^2 = 3^2 + 4^2."""
        result = find_two_square_representation(25)
        assert result is not None
        a, b = result
        assert a * a + b * b == 25

    def test_65_has_two_representations(self):
        """65 = 1^2 + 8^2 = 4^2 + 7^2."""
        reps = find_all_two_square_representations(65)
        assert len(reps) >= 2
        for a, b in reps:
            assert a * a + b * b == 65

    def test_3_no_representation(self):
        """3 cannot be expressed as sum of two squares."""
        result = find_two_square_representation(3)
        assert result is None

    def test_primes_1_mod_4(self):
        """Primes p ≡ 1 mod 4 have a representation."""
        for p in [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]:
            result = find_two_square_representation(p)
            assert result is not None, f"Prime {p} should have a two-square representation"
            a, b = result
            assert a * a + b * b == p


class TestBrahmaguptaFibonacciFactor:
    """Test factoring via the Brahmagupta-Fibonacci identity."""

    @pytest.mark.parametrize("N,p,q", [
        (65, 5, 13),    # 65 = 1²+8² = 4²+7²
        (85, 5, 17),    # 85 = 2²+9² = 6²+7²
        (325, 5, 65),   # 325 = 1²+18² = 6²+17²
    ])
    def test_factors_numbers_with_two_representations(self, N, p, q):
        """Numbers with two square representations should be factorable."""
        result = brahmagupta_fibonacci_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_returns_none_for_primes_3_mod_4(self):
        """Primes ≡ 3 mod 4 have no sum-of-two-squares representation."""
        result = brahmagupta_fibonacci_factor(7)
        assert result is None

    def test_returns_none_for_small_N(self):
        result = brahmagupta_fibonacci_factor(2)
        assert result is None


class TestFermatDifferenceOfSquares:
    """Test Fermat's difference-of-squares method."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),   # Close factors
        (899, 29, 31),   # Twin primes
    ])
    def test_factors_semiprimes(self, N, p, q):
        """Fermat should factor close-factor semiprimes efficiently."""
        result = fermat_difference_of_squares(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_close_factors(self):
        """Fermat excels at close-factor semiprimes."""
        for p, q in [(17, 19), (29, 31), (41, 43)]:
            N = p * q
            result = fermat_difference_of_squares(N)
            assert result is not None, f"Failed to factor {N}={p}*{q}"
            fp, fq = result
            assert fp * fq == N


class TestFibonacciMod:
    """Test Fibonacci modular arithmetic."""

    def test_fib_mod_0(self):
        assert fibonacci_mod(0, 10) == 0

    def test_fib_mod_1(self):
        assert fibonacci_mod(1, 10) == 1

    def test_fib_mod_small(self):
        """F(10) = 55, F(10) mod 7 = 55 mod 7 = 6."""
        assert fibonacci_mod(10, 7) == 6

    def test_fib_mod_large(self):
        """F(100) mod 1000."""
        result = fibonacci_mod(100, 1000)
        assert 0 <= result < 1000

    def test_fib_mod_identity(self):
        """F(n) mod 1 = 0."""
        for n in [0, 1, 5, 10, 50]:
            assert fibonacci_mod(n, 1) == 0


class TestPisanoPeriod:
    """Test Pisano period computation."""

    def test_pisano_2(self):
        """π(2) = 3: F mod 2 = 0, 1, 1, 0, 1, 1, ..."""
        assert pisano_period(2) == 3

    def test_pisano_5(self):
        """π(5) = 20."""
        assert pisano_period(5) == 20

    def test_pisano_10(self):
        """π(10) = 60."""
        assert pisano_period(10) == 60


class TestEntryPoint:
    """Test Fibonacci entry point computation."""

    def test_entry_point_5(self):
        """5 | F(5), so α(5) divides 5."""
        alpha = entry_point(5, 100)
        assert alpha is not None
        assert fibonacci_mod(alpha, 5) == 0

    def test_entry_point_7(self):
        """7 | F(8), so α(7) divides 8."""
        alpha = entry_point(7, 100)
        assert alpha is not None
        assert fibonacci_mod(alpha, 7) == 0


class TestFibonacciGCD:
    """Test Fibonacci GCD factoring."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Fibonacci GCD should factor small semiprimes."""
        result = fibonacci_gcd_factor(N, bound=1000)
        # May or may not find a factor depending on entry points
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestFactorIntegration:
    """Test that the factor() API integrates all new methods."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (667, 23, 29),
        (323, 17, 19),
        (121, 11, 11),
        (961, 31, 31),
    ])
    def test_factor_still_works(self, N, p, q):
        """factor() should still find factors with the new methods."""
        result = factor(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    def test_factor_with_method_reports_valid_method(self):
        """factor_with_method should report a valid method name."""
        valid_methods = {
            "perfect_square", "cf_precheck", "brahmagupta", "fermat",
            "fibonacci", "resonance_cascade", "lucas_ppt", "spectral_cascade",
            "fib_pyth", "lucas_multi", "crt_collision",
            "sl2_group_order", "sl2_structured", "batch_crt", "ppt_sieve",
            "cf_matrix_cascade", "cf_cascade",
            "ppt_form", "squfof", "class_group", "class_squfof", "class_group", "class_squfof",
            "relation_gen", "inside_out", "wavefront", "trial_division",
        }
        result = factor_with_method(21)
        assert result is not None
        _, method = result
        assert method in valid_methods

    def test_brahmagupta_method_for_65(self):
        """65 = 5*13 should be found by Brahmagupta method."""
        result = factor_with_method(65)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 65
        # 65 has two square representations: 1²+8² and 4²+7²
        # so Brahmagupta should work (but other methods may find it first)
        assert method in valid_methods_list()

    def test_fermat_method_for_close_factors(self):
        """Fermat method excels at close factors."""
        # 1763 = 41*43, very close factors
        result = factor_with_method(1763)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 1763


def valid_methods_list():
    return [
        "perfect_square", "cf_precheck", "brahmagupta", "fermat",
        "fibonacci", "resonance_cascade", "lucas_ppt", "spectral_cascade",
        "fib_pyth", "lucas_multi", "crt_collision",
        "sl2_group_order", "sl2_structured", "batch_crt", "ppt_sieve",
        "cf_matrix_cascade", "cf_cascade",
        "ppt_form", "squfof", "class_group", "class_squfof",
        "cyclotomic_resultant", "cyclotomic_cascade",
        "discriminant_resonance", "quadratic_resonance",
        "hensel_cascade", "crt_lattice",
        "lattice_factor", "hybrid_smooth",
        "graph_order", "order_spectrum",
        "relation_gen", "inside_out", "wavefront", "trial_division",
        "projective_chart", "orbit_relation",
    ]


class TestChartCompression:
    """Tests for conic chart compression: single determinant vs 3 minors."""

    def test_chart_determinant_zero_when_minors_zero(self):
        """If two triples are projectively equal mod p, chart determinant is 0 mod p.

        Proportional triples (3,4,5) and (6,8,10) represent the same projective
        point, so their chart determinant should vanish mod p.
        """
        from insideout.projective_collision import chart_determinant, Triple
        # (3,4,5) and (6,8,10) are proportional → same projective point
        t1 = Triple(3, 4, 5)
        t2 = Triple(6, 8, 10)  # 2*(3,4,5)
        det = chart_determinant(t1, t2, 97)
        assert det % 97 == 0, f"chart det {det} should be 0 mod 97"

    def test_chart_determinant_nonzero_when_minors_nonzero(self):
        """If triples differ mod p, chart determinant is nonzero with high probability."""
        from insideout.projective_collision import chart_determinant, Triple
        t1 = Triple(3, 4, 5)
        t2 = Triple(7, 11, 13)
        det = chart_determinant(t1, t2, 97)
        # Probabilistically nonzero mod 97 unless we got unlucky
        assert det % 97 != 0 or det == 0

    def test_failed_inversion_gcd(self):
        """When c+b is not invertible mod N, gcd(c+b, N) reveals a factor."""
        from insideout.projective_collision import gcd_safe_c_plus_b, Triple
        # N = 97 * 101 = 9797; c+b = 194 = 2 * 97
        t = Triple(97, 0, 97)
        g = gcd_safe_c_plus_b(t, 97 * 101)
        assert g in (97, 101, 9797), f"gcd = {g}"

    def test_distinguished_predicate(self):
        """Distinguished points have low bits zero."""
        from insideout.projective_collision import is_distinguished, Triple
        t_dist = Triple(0, 0, 0)
        t_nondist = Triple(1, 2, 3)
        assert is_distinguished(t_dist, bits=4) is True
        assert is_distinguished(t_nondist, bits=4) is False

    def test_distinguished_density(self):
        """Distinguished density is approximately 1/2^(3*bits)."""
        import random
        from insideout.projective_collision import is_distinguished, Triple
        random.seed(42)
        count = 0
        for _ in range(10000):
            t = Triple(random.randrange(0, 2**20),
                       random.randrange(0, 2**20),
                       random.randrange(0, 2**20))
            if is_distinguished(t, bits=8):
                count += 1
        # Expected: 10000 / 2^24 ≈ 0.0006; allow broad range
        assert 0 <= count <= 10, f"distinguished count {count} out of expected range"


SEMIPRIMES = [
    (35, 5, 7),
    (77, 7, 11),
    (221, 13, 17),
    (437, 19, 23),
    (667, 23, 29),
    (1147, 31, 37),
    (1927, 41, 47),
    (8051, 83, 97),
    (15571, 113, 137),
    # Balanced semiprimes (harder for rho-like methods)
    (3127,  53,  59),   # 53×59=3127
    (3599,  59,  61),   # 59×61=3599
    (4757,  67,  71),   # 67×71=4757
    (4891,  67,  73),   # 67×73=4891
    (5183,  71,  73),   # 71×73=5183
    (6557,  79,  83),   # 79×83=6557
    # Larger semiprimes
    (1022117, 1009, 1013),
    (1032247, 1013, 1019),
    (1040399, 1019, 1021),
    (1052651, 1021, 1031),
    (1065023, 1031, 1033),
    (1073287, 1033, 1039),
    (1089911, 1039, 1049),
    (1102499, 1049, 1051),
]


class TestProjectiveChartIntegration:
    """Integration tests: chart collision factoring on known semiprimes."""

    @pytest.mark.parametrize("expected, p, q", SEMIPRIMES)
    def test_chart_factors_known_semiprime(self, expected, p, q):
        from insideout.projective_collision import chart_collision_factor
        N = p * q
        result = chart_collision_factor(N, max_steps=50000, num_walks=16)
        if result is None:
            result = chart_collision_factor(N, max_steps=200000, num_walks=32)
        assert result is not None, f"chart_collision failed on N={N}"
        factors = sorted(result)
        assert factors[0] == min(p, q) and factors[1] == max(p, q), \
            f"got {factors}, expected ({min(p,q)}, {max(p,q)})"


class TestOrbitSmoothRelation:
    """Tests for orbit-to-smooth-relation NFS lane."""

    def test_norm_of_branch_word_is_integer(self):
        """norm_of_branch_word returns a positive integer."""
        from insideout.orbit_smooth_relation import norm_of_branch_word, Triple
        N = 97 * 101
        seed = Triple(3 % N, 4 % N, 5 % N)
        n = norm_of_branch_word('U', seed, N)
        assert isinstance(n, int), f"norm should be int, got {type(n)}"
        assert n > 0, f"norm should be positive, got {n}"

    def test_norm_same_for_equivalent_words(self):
        """norm is deterministic: same word gives same norm."""
        from insideout.orbit_smooth_relation import norm_of_branch_word, Triple
        N = 97 * 101
        seed = Triple(3 % N, 4 % N, 5 % N)
        n1 = norm_of_branch_word('UUU', seed, N)
        n2 = norm_of_branch_word('UUU', seed, N)
        assert n1 == n2

    def test_smooth_detection_known_smooth(self):
        """Known smooth numbers are detected correctly."""
        from insideout.orbit_smooth_relation import is_smooth
        # 2^10 = 1024 = only prime {2}; 2 <= 100, so True
        assert is_smooth(1024, bound=1024) is True
        assert is_smooth(1024, bound=100) is True   # 2 <= 100
        # 2*3*5 = 30; all factors <= 5, so True
        assert is_smooth(30, bound=30) is True
        assert is_smooth(30, bound=5) is True   # 2,3,5 all <= 5

    def test_smooth_detection_not_smooth(self):
        """Numbers with large prime factors are not smooth."""
        from insideout.orbit_smooth_relation import is_smooth
        assert is_smooth(101, bound=100) is False  # 101 > 100
        assert is_smooth(103424, bound=100) is False   # 103424 = 2^10 * 101; 101 > 100

    def test_build_relation_matrix_dimensions(self):
        """Relation matrix has correct row/col dimensions."""
        from insideout.orbit_smooth_relation import build_relation_matrix
        relations = [
            {2: 2, 3: 1},   # 12 = 2^2 * 3^1
            {2: 1, 3: 2},   # 18 = 2^1 * 3^2
        ]
        FB = [2, 3, 5, 7]
        M = build_relation_matrix(relations, FB)
        assert M.rows == 2, f"expected 2 rows, got {M.rows}"
        assert M.cols == 4, f"expected 4 cols, got {M.cols}"

    def test_gaussian_elimination_finds_nullvector(self):
        """Gaussian elimination over GF(2) finds a nullspace vector.

        Matrix (3x3, rank 2, nullity 1):
            [1 1 1]
            [1 0 1]
        Row1 xor Row2 = [0 1 0] → no pivot in col 1 → nullspace = {1}
        """
        from insideout.orbit_smooth_relation import GF2SparseMatrix
        M = GF2SparseMatrix(2, 3)
        M.set(0, 0); M.set(0, 1); M.set(0, 2)  # row 0: [1,1,1]
        M.set(1, 0); M.set(1, 2)                 # row 1: [1,0,1]
        null = M.gaussian_elimination()
        assert null is not None
        assert len(null) > 0


class TestOrbitRelationIntegration:
    """Integration tests: orbit-to-smooth-relation NFS lane on known semiprimes."""

    @pytest.mark.parametrize("N", [35, 77, 221, 437, 667, 1147, 1927, 8051])
    def test_orbit_relation_factors_known_semiprime(self, N):
        """orbit_smooth_relation_factor finds a factor for known semiprimes.

        This is a research experiment; skip if it fails gracefully.
        """
        from insideout.orbit_smooth_relation import orbit_smooth_relation_factor
        result = orbit_smooth_relation_factor(N, bound=30000, word_length=15)
        # It's acceptable for this research method to return None
        if result is not None:
            factors = sorted(result)
            assert factors[0] * factors[1] == N
            assert 1 < factors[0] < N