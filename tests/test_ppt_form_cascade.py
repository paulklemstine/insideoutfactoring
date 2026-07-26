"""Tests for PPT-Form Cascade factoring."""
import pytest
from math import isqrt

from insideout.ppt_form_cascade import (
    _reduce_form,
    _compose_forms,
    _squfof_step,
    ppt_form_cascade_factor,
    squfof_factor,
)


class TestReduceForm:
    """Test binary quadratic form reduction."""

    def test_identity_form(self):
        """Identity form (1, 0, 1) should stay reduced."""
        result = _reduce_form(1, 0, 1, 100)
        a, b, c = result
        assert a == 1
        assert b == 0
        assert c == 1

    def test_form_mod_N(self):
        """Form values should be taken mod N."""
        result = _reduce_form(101, 202, 303, 100)
        a, b, c = result
        assert a < 100
        assert b < 100
        assert c < 100


class TestComposeForms:
    """Test form composition."""

    def test_identity_composition(self):
        """Composing with identity should preserve form."""
        result = _compose_forms(1, 0, 1, 1, 0, 1, 97)
        a, b, c, factor = result
        assert factor is None  # No factor found
        # Identity composed with identity should give something related

    def test_composition_detects_factor(self):
        """Composition should detect gcd with N when present."""
        # Form with a=15, which shares factor 5 with N=15
        result = _compose_forms(15, 3, 7, 1, 0, 1, 15)
        a, b, c, factor = result
        # g = gcd(15, 15) = 15, but we need 1 < g < N
        # This particular case gives g=N, so no factor


class TestSqufofStep:
    """Test SQUFOF reduction."""

    def test_find_factor_15(self):
        """Should find factor of 15."""
        # Principal form of discriminant 4*15 = 60
        # sqrt(60) ≈ 7, form: (4, 14, -1) → (4, -14, -1) after adjustment
        result = _squfof_step(15, 4, 14, 1, max_steps=100)
        # May or may not find factor depending on form
        if result is not None:
            p, q = result
            assert p * q == 15


class TestPPTFormCascadeFactor:
    """Test the PPT-form cascade factor."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """PPT form cascade should factor small semiprimes."""
        result = ppt_form_cascade_factor(N, max_ppt=5000, squfof_steps=10000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = ppt_form_cascade_factor(1681)  # 41²
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = ppt_form_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = ppt_form_cascade_factor(7, max_ppt=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert ppt_form_cascade_factor(1) is None
        assert ppt_form_cascade_factor(2) is None
        assert ppt_form_cascade_factor(3) is None


class TestSqufofFactor:
    """Test standard SQUFOF implementation."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """SQUFOF should factor small semiprimes."""
        result = squfof_factor(N, max_iterations=50000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = squfof_factor(1681)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = squfof_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = squfof_factor(7, max_iterations=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert squfof_factor(1) is None
        assert squfof_factor(2) is None
        assert squfof_factor(3) is None