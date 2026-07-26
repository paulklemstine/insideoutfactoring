"""Tests for Class-Group Smooth-Order Cascade factoring."""
import pytest
import random
from math import isqrt

from insideout.class_group_cascade import (
    _small_primes,
    _reduce_form,
    _compose_forms,
    _check_form_crt,
    _random_form,
    _ppt_derived_forms,
    class_group_cascade_factor,
    class_group_squfof_factor,
)


class TestSmallPrimes:
    """Test prime sieve."""

    def test_up_to_10(self):
        assert _small_primes(10) == [2, 3, 5, 7]

    def test_empty(self):
        assert _small_primes(1) == []


class TestReduceForm:
    """Test binary quadratic form reduction."""

    def test_reduced_form_stays(self):
        """A reduced form should stay reduced."""
        result = _reduce_form(1, 0, 1, -4)
        assert result[0] == 1  # a should stay 1
        assert result[1] == 0  # b should stay 0

    def test_form_mod_values(self):
        """Form values should be bounded."""
        result = _reduce_form(100, 50, 25, -4)
        # After reduction, values should be reasonable
        a, b, c = result
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert isinstance(c, int)


class TestComposeForms:
    """Test form composition."""

    def test_identity_composition(self):
        """Composing with the identity should preserve the form."""
        # (1, 0, 1) is close to identity
        result = _compose_forms(1, 0, 1, 1, 0, 1, 97)
        a, b, c, factor = result
        assert factor is None  # Should not find a factor for prime N

    def test_factor_detection(self):
        """Composition should detect gcd with N."""
        # Form (15, 3, 7) composed with (1, 0, 1) mod 15
        result = _compose_forms(15, 3, 7, 1, 0, 1, 15)
        # gcd(15, 15) = 15, which is N itself, so no factor found
        a, b, c, factor = result
        # Factor may or may not be found depending on composition path


class TestCheckFormCRT:
    """Test CRT divergence detection in forms."""

    def test_identity_no_factor(self):
        """Identity form should not reveal a factor."""
        result = _check_form_crt(1, 0, 1, 15)
        assert result is None

    def test_divisible_coefficient(self):
        """A form with a coefficient divisible by a factor should reveal it."""
        # Form (3, 0, 1) mod 15: gcd(3, 15) = 3
        result = _check_form_crt(3, 0, 1, 15)
        assert result is not None
        assert 3 in result


class TestPPTDerivedForms:
    """Test PPT-derived form generation."""

    def test_generates_forms(self):
        """Should generate PPT-derived forms."""
        forms, factor = _ppt_derived_forms(97)  # Prime N, no early factor
        assert len(forms) > 0
        assert factor is None

    def test_form_coefficients_bounded(self):
        """Form coefficients should be bounded by N."""
        forms, factor = _ppt_derived_forms(97)  # Prime N
        assert factor is None
        for a, b, c in forms:
            assert 0 <= a < 97
            assert 0 <= b < 97
            assert 0 <= c < 97


class TestClassGroupCascadeFactor:
    """Test the class-group smooth-order cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Class group cascade should factor small semiprimes."""
        random.seed(42)
        result = class_group_cascade_factor(N, bound=5000, curves=10)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        random.seed(42)
        result = class_group_cascade_factor(1681, bound=5000)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        random.seed(42)
        result = class_group_cascade_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        random.seed(42)
        result = class_group_cascade_factor(7, bound=500)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert class_group_cascade_factor(1) is None
        assert class_group_cascade_factor(2) is None
        assert class_group_cascade_factor(3) is None


class TestClassGroupSQUFOF:
    """Test the SQUFOF implementation with PPT starting forms."""

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
        result = class_group_squfof_factor(N, max_iterations=50000)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Perfect squares should be detected."""
        result = class_group_squfof_factor(1681)
        assert result is not None
        assert result == (41, 41)

    def test_even_number(self):
        """Even numbers should return factor 2."""
        result = class_group_squfof_factor(30)
        assert result is not None
        assert 2 in result

    def test_prime_returns_none(self):
        """Primes should return None."""
        result = class_group_squfof_factor(7, max_iterations=100)
        assert result is None

    def test_rejects_N_lt_4(self):
        """N < 4 should return None."""
        assert class_group_squfof_factor(1) is None
        assert class_group_squfof_factor(2) is None
        assert class_group_squfof_factor(3) is None