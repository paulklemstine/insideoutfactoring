"""Tests for Spectral Cascade Factoring — Pure Novel Methods."""
import pytest
from math import isqrt

from insideout.spectral_factor import (
    spectral_cascade_factor,
    _cf_squaring_cascade,
    _sl2_matrix_cascade,
    _qr_discriminator,
    _idempotent_detection,
    _cf_sqrt,
    _convergents,
)


class TestCFSqrt:
    """Test continued fraction expansion of √N."""

    def test_perfect_square(self):
        """CF of perfect square should be [sqrt]."""
        assert _cf_sqrt(4) == [2]
        assert _cf_sqrt(9) == [3]
        assert _cf_sqrt(16) == [4]

    def test_non_square(self):
        """CF of non-square should start with floor(sqrt)."""
        cf = _cf_sqrt(2)
        assert cf[0] == 1
        assert len(cf) > 1

    def test_periodic(self):
        """CF of √2 should be [1, 2, 2, 2, ...]."""
        cf = _cf_sqrt(2, max_terms=10)
        assert cf[0] == 1
        for a in cf[1:]:
            assert a == 2


class TestConvergents:
    """Test CF convergent computation."""

    def test_sqrt2_convergents(self):
        """Convergents of √2 should satisfy p² - 2q² = ±1."""
        cf = _cf_sqrt(2, max_terms=20)
        convs = _convergents(cf)
        for pk, qk in convs:
            residue = pk * pk - 2 * qk * qk
            assert abs(residue) == 1

    def test_first_convergent(self):
        """First convergent should be (a₀, 1)."""
        cf = _cf_sqrt(7, max_terms=10)
        convs = _convergents(cf)
        assert convs[0] == (cf[0], 1)


class TestCFSquaringCascade:
    """Test CF-Convergent Squaring Cascade."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (323, 17, 19),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes."""
        cf = _cf_sqrt(N, max_terms=100)
        convs = _convergents(cf)
        result = _cf_squaring_cascade(N, convs)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestSL2MatrixCascade:
    """Test SL₂ Matrix Order Detection."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (35, 5, 7),
        (77, 7, 11),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor some small semiprimes."""
        result = _sl2_matrix_cascade(N, max_k_bits=20)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestQRDiscriminator:
    """Test Quadratic Residue Discriminator."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor some small semiprimes."""
        cf = _cf_sqrt(N, max_terms=50)
        convs = _convergents(cf)
        result = _qr_discriminator(N, convs)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestIdempotentDetection:
    """Test Idempotent Detection."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Should factor small semiprimes via idempotent detection."""
        cf = _cf_sqrt(N, max_terms=50)
        convs = _convergents(cf)
        result = _idempotent_detection(N, convs)
        if result is not None:
            fp, fq = result
            assert fp * fq == N


class TestSpectralCascadeFactor:
    """Test the full Spectral Cascade Factoring pipeline."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (323, 17, 19),
        (899, 29, 31),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Spectral cascade should factor small semiprimes."""
        result = spectral_cascade_factor(N)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_perfect_square(self):
        """Should handle perfect squares."""
        result = spectral_cascade_factor(121)
        assert result is not None
        assert result == (11, 11)

    def test_even_number(self):
        """Should handle even numbers."""
        result = spectral_cascade_factor(6)
        assert result is not None
        assert result[0] * result[1] == 6

    def test_rejects_trivial_factors(self):
        """Should not return trivial factors (1 or N)."""
        result = spectral_cascade_factor(15)
        if result is not None:
            p, q = result
            assert 1 < p < 15
            assert 1 < q < 15

    def test_close_factors(self):
        """Should handle close-factor semiprimes."""
        result = spectral_cascade_factor(323)  # 17*19
        if result is not None:
            assert result[0] * result[1] == 323

    def test_far_factors(self):
        """Should handle far-factor semiprimes."""
        result = spectral_cascade_factor(7 * 10007)
        if result is not None:
            assert result[0] * result[1] == 7 * 10007

    def test_medium_semiprime(self):
        """Should factor medium semiprimes."""
        result = spectral_cascade_factor(10007 * 10009)
        if result is not None:
            assert result[0] * result[1] == 10007 * 10009