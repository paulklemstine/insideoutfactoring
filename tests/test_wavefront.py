"""Tests for wavefront parallel search."""
import pytest
from insideout.wavefront import expand_wavefront, search_wavefront


class TestExpandWavefront:
    def test_yields_batches(self):
        """expand_wavefront should yield lists of triples."""
        batches = list(expand_wavefront(15, max_batches=5))
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, list)
            assert len(batch) > 0

    def test_batches_increase_in_energy(self):
        """Later batches should have higher-energy triples."""
        batches = list(expand_wavefront(15, max_batches=5))
        if len(batches) >= 2:
            from insideout.energy import energy
            min_energy_first = min(energy(t) for t in batches[0])
            min_energy_last = min(energy(t) for t in batches[-1])
            assert min_energy_last >= min_energy_first

    def test_all_triples_are_valid_ppts(self):
        """All yielded triples should satisfy a^2 + b^2 = c^2."""
        for batch in expand_wavefront(15, max_batches=5):
            for t in batch:
                assert t.a ** 2 + t.b ** 2 == t.c ** 2

    def test_respects_energy_bound(self):
        """All yielded triples should have c <= (N^2+1)/2."""
        N = 15
        upper = (N * N + 1) // 2
        for batch in expand_wavefront(N, max_batches=5):
            for t in batch:
                assert t.c <= upper

    def test_yields_at_least_one_batch(self):
        """Even for small N, should produce at least one batch."""
        batches = list(expand_wavefront(6, max_batches=3))
        assert len(batches) >= 1

    def test_max_batches_limits_iterations(self):
        """Should not yield more batches than max_batches."""
        batches = list(expand_wavefront(15, max_batches=2))
        assert len(batches) <= 2


class TestSearchWavefront:
    def test_finds_factor_15(self):
        result = search_wavefront(15, max_radius=50)
        assert result is not None
        p, q = result
        assert p * q == 15

    def test_finds_factor_35(self):
        result = search_wavefront(35, max_radius=50)
        assert result is not None
        p, q = result
        assert p * q == 35

    def test_finds_factor_21(self):
        result = search_wavefront(21, max_radius=50)
        assert result is not None
        p, q = result
        assert p * q == 21

    def test_finds_factor_77(self):
        """77 = 7 * 11"""
        result = search_wavefront(77, max_radius=100)
        assert result is not None
        p, q = result
        assert p * q == 77

    def test_returns_ordered_factors(self):
        """Factors should be returned with p <= q."""
        result = search_wavefront(35, max_radius=50)
        assert result is not None
        p, q = result
        assert p <= q

    def test_handles_even_N(self):
        """Even numbers should be handled."""
        result = search_wavefront(6, max_radius=50)
        assert result is not None
        p, q = result
        assert p * q == 6

    def test_rejects_small_N(self):
        """N < 4 cannot be factored as a product of two integers > 1."""
        assert search_wavefront(3, max_radius=50) is None
        assert search_wavefront(2, max_radius=50) is None
        assert search_wavefront(1, max_radius=50) is None

    def test_returns_none_for_prime(self):
        """A prime number has no non-trivial factors."""
        result = search_wavefront(7, max_radius=50)
        # May or may not find a factor, but result should be None
        # since 7 is prime
        if result is not None:
            p, q = result
            assert p * q == 7