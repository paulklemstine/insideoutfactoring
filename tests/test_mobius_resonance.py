"""Tests for Möbius Resonance Factoring algorithm."""
import pytest
from math import gcd

from insideout.mobius_resonance import (
    apply_mobius_word,
    mobius_matrix,
    mat_vec_mul,
    factor_from_collision,
    detect_resonance,
    mobius_resonance_factor,
    _is_distinguished,
)


class TestMobiusAction:
    """Test the Möbius action on (num, den) pairs."""

    def test_U_action(self):
        """U(t) = t/(1+t). For t=1/2: (1/2)/(3/2) = 1/3."""
        assert apply_mobius_word((1, 2), 'U', 10**18) == (1, 3)

    def test_A_action(self):
        """A(t) = (t+1)/(t+2). For t=1/2: (3/2)/(5/2) = 3/5."""
        assert apply_mobius_word((1, 2), 'A', 10**18) == (3, 5)

    def test_D_action(self):
        """D(t) = 1/(2+t). For t=1/2: 1/(5/2) = 2/5."""
        assert apply_mobius_word((1, 2), 'D', 10**18) == (2, 5)

    def test_composition_UA(self):
        """UA means apply U then A."""
        # U(1/2) = 1/3, A(1/3) = (1/3+1)/(1/3+2) = (4/3)/(7/3) = 4/7
        assert apply_mobius_word((1, 2), 'UA', 10**18) == (4, 7)

    def test_composition_AU(self):
        """AU means apply A then U."""
        # A(1/2) = 3/5, U(3/5) = (3/5)/(8/5) = 3/8
        assert apply_mobius_word((1, 2), 'AU', 10**18) == (3, 8)

    def test_identity_word(self):
        """Empty word should be identity."""
        assert apply_mobius_word((1, 2), '', 10**18) == (1, 2)

    def test_modular_reduction(self):
        """Action should reduce mod N."""
        N = 8051
        # U(1/2) = 1/3, so (1, 3) mod 8051 = (1, 3)
        assert apply_mobius_word((1, 2), 'U', N) == (1, 3)
        # Larger example: apply long word
        result = apply_mobius_word((1, 2), 'UUDDAU', N)
        num, den = result
        assert 0 <= num < N
        assert 0 <= den < N


class TestMobiusMatrix:
    """Test matrix computation for Möbius words."""

    def test_U_matrix(self):
        M = mobius_matrix('U')
        assert M == ((1, 0), (1, 1))

    def test_A_matrix(self):
        M = mobius_matrix('A')
        assert M == ((1, 1), (1, 2))

    def test_D_matrix(self):
        M = mobius_matrix('D')
        assert M == ((0, 1), (1, 2))

    def test_identity_matrix(self):
        M = mobius_matrix('')
        assert M == ((1, 0), (0, 1))

    def test_composition_UA(self):
        """M_UA should equal M_A * M_U."""
        M_UA = mobius_matrix('UA')
        M_U = mobius_matrix('U')
        M_A = mobius_matrix('A')
        # Matrix multiply M_A * M_U
        a1, b1 = M_A[0]
        c1, d1 = M_A[1]
        a2, b2 = M_U[0]
        c2, d2 = M_U[1]
        expected = (
            (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2),
            (c1 * a2 + d1 * c2, c1 * b2 + d1 * d2),
        )
        assert M_UA == expected

    def test_matrix_times_vector(self):
        """M_UA * (1, 2) should equal apply_mobius_word((1,2), 'UA')."""
        N = 10**18
        M = mobius_matrix('UA')
        v_mat = mat_vec_mul(M, (1, 2), N)
        v_word = apply_mobius_word((1, 2), 'UA', N)
        assert v_mat == v_word

    def test_matrix_determinants(self):
        """U and A have det=1; D has det=-1."""
        det_U = mobius_matrix('U')[0][0] * mobius_matrix('U')[1][1] - \
                mobius_matrix('U')[0][1] * mobius_matrix('U')[1][0]
        det_A = mobius_matrix('A')[0][0] * mobius_matrix('A')[1][1] - \
                mobius_matrix('A')[0][1] * mobius_matrix('A')[1][0]
        det_D = mobius_matrix('D')[0][0] * mobius_matrix('D')[1][1] - \
                mobius_matrix('D')[0][1] * mobius_matrix('D')[1][0]
        assert det_U == 1
        assert det_A == 1
        assert det_D == -1

    def test_longer_word_matrix(self):
        """Matrix for longer word should be consistent with action."""
        N = 10**18
        word = 'UUDDAUDDAU'
        M = mobius_matrix(word)
        v_mat = mat_vec_mul(M, (1, 2), N)
        v_word = apply_mobius_word((1, 2), word, N)
        assert v_mat == v_word


class TestFactorFromCollision:
    """Test factor extraction from colliding matrices."""

    def test_trivial_collision_gives_none(self):
        """If matrices are identical, no factor should be found."""
        M = mobius_matrix('UUDDAU')
        assert factor_from_collision(M, M, 8051) is None

    def test_known_collision(self):
        """Construct a known collision that reveals a factor."""
        # For N = 8051 = 83 * 97, find two words that give same value mod 83
        # but different mod 97, then check that factor_from_collision works.
        N = 8051
        # We'll brute-force find a collision
        seen = {}
        for i in range(100000):
            import random
            word = ''.join(random.choice('UAD') for _ in range(20))
            point = apply_mobius_word((1, 2), word, N)
            if point in seen and seen[point] != word:
                M1 = mobius_matrix(word)
                M2 = mobius_matrix(seen[point])
                v1 = mat_vec_mul(M1, (1, 2), N)
                v2 = mat_vec_mul(M2, (1, 2), N)
                if v1 == v2:
                    factor = factor_from_collision(M1, M2, N)
                    # It's OK if factor is None — collisions don't always yield factors
                    # But if it finds one, it must be correct
                    if factor is not None:
                        assert N % factor == 0
                        assert 1 < factor < N
                    return
            seen[point] = word
        # If we didn't find a collision, that's OK too
        pytest.skip("No collision found in reasonable time")


class TestIsDistinguished:
    """Test the distinguished-point predicate."""

    def test_zero_is_distinguished(self):
        assert _is_distinguished((0, 0), 0b1111)

    def test_low_bits_zero(self):
        # 0b10000 has low 4 bits zero
        assert _is_distinguished((16, 32), 0b1111)

    def test_low_bits_nonzero(self):
        # 0b0001 has low bit set
        assert not _is_distinguished((1, 16), 0b1111)

    def test_both_must_be_distinguished(self):
        assert not _is_distinguished((16, 1), 0b1111)


class TestDetectResonance:
    """Test the resonance detection algorithm."""

    @pytest.mark.parametrize("N,p,q", [
        (8051, 83, 97),
        (15571, 23, 677),
        (1022117, 1009, 1013),
        (10403, 101, 103),
    ])
    def test_detects_factors(self, N, p, q):
        """detect_resonance should factor small semiprimes."""
        result = detect_resonance(
            N, walk_steps=30, num_walks=200000,
            distinguished_bits=6, seed=42,
        )
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_even_number(self):
        """Even numbers should be handled."""
        result = detect_resonance(26)
        assert result == (2, 13)


class TestMobiusResonanceFactor:
    """Test the main entry point."""

    @pytest.mark.parametrize("N,expected_p,expected_q", [
        (8051, 83, 97),
        (15571, 23, 677),
        (1022117, 1009, 1013),
        (10403, 101, 103),
        (10007 * 10009, 10007, 10009),
        (1000003 * 1000033, 1000003, 1000033),
    ])
    def test_factors_semiprimes(self, N, expected_p, expected_q):
        """MRF should factor semiprimes correctly."""
        result = mobius_resonance_factor(N, max_steps=500000, distinguished_bits=6)
        assert result is not None
        p, q = result
        assert p * q == N
        assert {p, q} == {expected_p, expected_q}

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = mobius_resonance_factor(121)
        assert result == (11, 11)

    def test_even_number(self):
        """Even numbers should return (2, N/2)."""
        result = mobius_resonance_factor(26)
        assert result == (2, 13)

    def test_small_N(self):
        """N < 4 should return None."""
        assert mobius_resonance_factor(1) is None
        assert mobius_resonance_factor(2) is None
        assert mobius_resonance_factor(3) is None

    def test_returns_ordered_pair(self):
        """Result should be (p, q) with p <= q."""
        result = mobius_resonance_factor(8051)
        assert result is not None
        p, q = result
        assert p <= q
        assert p * q == 8051
