"""Tests for the top-level factor API."""
import pytest
from insideout.factor import factor, factor_with_method


class TestFactor:
    """Integration tests covering known semiprimes."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (667, 23, 29),
    ])
    def test_known_semiprimes(self, N, p, q):
        result = factor(N)
        assert result is not None, f"Failed to factor {N}"
        assert result[0] * result[1] == N
        assert result == (p, q) or result == (q, p)

    def test_rejects_prime(self):
        result = factor(7)
        assert result is None

    def test_rejects_one(self):
        result = factor(1)
        assert result is None

    def test_even_numbers(self):
        result = factor(6)
        assert result is not None
        assert result[0] * result[1] == 6

    def test_larger_semiprime(self):
        """Test a larger semiprime: 10403 = 101 * 103."""
        result = factor(10403)
        if result is not None:
            assert result[0] * result[1] == 10403

    def test_N_below_4_returns_none(self):
        """N < 4 cannot be expressed as product of two integers > 1."""
        for N in (0, 1, 2, 3):
            assert factor(N) is None, f"factor({N}) should be None"

    def test_even_N_returns_two_factor(self):
        """Even N should always return (2, N//2)."""
        result = factor(10)
        assert result is not None
        assert result[0] * result[1] == 10
        assert result == (2, 5)

    def test_result_ordered_p_lt_q(self):
        """Every result should have p < q."""
        for N in (15, 21, 35, 77):
            result = factor(N)
            assert result is not None
            p, q = result
            assert p < q, f"factor({N}) = ({p}, {q}), expected p < q"


class TestFactorWithMethod:
    def test_returns_method(self):
        result = factor_with_method(15)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 15
        assert method in ("inside_out", "wavefront", "trial_division")

    def test_even_returns_trial_division(self):
        result = factor_with_method(6)
        assert result is not None
        factors, method = result
        assert factors == (2, 3)
        assert method == "trial_division"

    def test_prime_returns_none(self):
        result = factor_with_method(7)
        assert result is None

    def test_N_below_4_returns_none(self):
        for N in (0, 1, 2, 3):
            assert factor_with_method(N) is None, f"factor_with_method({N}) should be None"

    def test_method_is_valid_string(self):
        """Method name must be one of the three known strategies."""
        valid = {"inside_out", "wavefront", "trial_division"}
        result = factor_with_method(21)
        assert result is not None
        _, method = result
        assert method in valid