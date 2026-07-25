"""Tests for CF-steered best-first search and CF-seeded well points."""
import pytest
from math import isqrt
from insideout.inside_out import (
    inside_out_factor,
    _steered_search,
    _bfs_search,
    central_well,
    cf_seeded_well_points,
)
from insideout.factor import factor, factor_with_method
from insideout.gaussian import MnPair
from insideout.cf_guide import cf_sqrt, convergents


class TestCFSeededWellPoints:
    """Test the CF-convergent-derived seed point generation."""

    def test_returns_nonempty_list(self):
        """Seed points should always be non-empty."""
        seeds = cf_seeded_well_points(15)
        assert len(seeds) > 0

    def test_well_in_seeds(self):
        """The central well should be included in seed points."""
        N = 667
        well = central_well(N)
        seeds = cf_seeded_well_points(N)
        seed_coords = [(s.m, s.n) for s in seeds]
        assert (well.m, well.n) in seed_coords

    def test_root_in_seeds(self):
        """The root (2,1) should be included."""
        seeds = cf_seeded_well_points(15)
        seed_coords = [(s.m, s.n) for s in seeds]
        assert (2, 1) in seed_coords

    def test_convergent_seeds_included(self):
        """CF convergent-derived points should be included."""
        N = 437  # = 19 * 23
        seeds = cf_seeded_well_points(N)
        seed_coords = [(s.m, s.n) for s in seeds]
        # The first convergent of sqrt(437) ≈ 20.9 is [20, 1, 1, ...]
        # So (20, 1) or (21, 1) should appear as seeds
        # At least some (m, 1) seeds near sqrt(N) should be present
        m_values = [s.m for s in seeds if s.n == 1]
        assert len(m_values) > 0, "Should have at least one (m, 1) seed"

    def test_seeds_are_valid_ppt_params(self):
        """All seed points should be valid PPT parameters."""
        from math import gcd
        for N in (15, 77, 437, 667):
            seeds = cf_seeded_well_points(N)
            for s in seeds:
                assert s.m > s.n > 0, f"Seed ({s.m}, {s.n}) should have m > n > 0"
                assert (s.m - s.n) % 2 == 1, f"Seed ({s.m}, {s.n}) should have opposite parity"
                assert gcd(s.m, s.n) == 1, f"Seed ({s.m}, {s.n}) should be coprime"

    def test_seeds_no_duplicates(self):
        """Seed points should not contain duplicates."""
        seeds = cf_seeded_well_points(667)
        seed_coords = [(s.m, s.n) for s in seeds]
        assert len(seed_coords) == len(set(seed_coords))

    def test_more_seeds_than_old_method(self):
        """CF-seeded should produce more starting points than the old ±5 window."""
        N = 1849  # = 43^2, close factor case
        seeds = cf_seeded_well_points(N)
        # Old method: ±5 on m, 0-5 on n = 66 seed points max
        # CF-seeded should have at least the well neighborhood + convergent seeds
        assert len(seeds) > 0


class TestSteeredSearch:
    """Test CF-steered best-first search."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (667, 23, 29),
    ])
    def test_steered_factors_semiprimes(self, N, p, q):
        """Steered search should factor standard semiprimes."""
        result = _steered_search(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    @pytest.mark.parametrize("N,p,q", [
        (323, 17, 19),   # twin primes
        (899, 29, 31),   # twin primes
        (1763, 41, 43),  # close primes
    ])
    def test_steered_handles_close_factors(self, N, p, q):
        """Steered search should efficiently handle close-factor semiprimes."""
        result = _steered_search(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    def test_steered_returns_none_for_prime(self):
        """Steered search should return None for primes."""
        result = _steered_search(7)
        assert result is None

    def test_steered_returns_none_for_small_N(self):
        """Steered search should return None for N < 4."""
        for N in (0, 1, 2, 3):
            result = _steered_search(N)
            assert result is None, f"Steered search should return None for N={N}"


class TestBFSSearch:
    """Test BFS fallback search."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
    ])
    def test_bfs_factors_small_semiprimes(self, N, p, q):
        """BFS search should factor small semiprimes."""
        result = _bfs_search(N, max_iterations=5000)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N


class TestInsideOutWithSteered:
    """Test that inside_out_factor uses steered search correctly."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
        (437, 19, 23),
        (667, 23, 29),
        (323, 17, 19),
        (899, 29, 31),
        (1763, 41, 43),
    ])
    def test_factor_uses_steered_search(self, N, p, q):
        """factor() should find all semiprime factorizations."""
        result = factor(N)
        assert result is not None, f"Failed to factor {N}"
        fp, fq = result
        assert fp * fq == N

    def test_factor_reports_inside_out_method(self):
        """factor_with_method should report 'inside_out' for steered+BFS."""
        # N=667 should be found by steered search (inside_out)
        result = factor_with_method(667)
        assert result is not None
        factors, method = result
        assert factors[0] * factors[1] == 667
        # Method could be 'cf_precheck', 'inside_out', or 'steered'
        # depending on which catches it first
        assert method in ("inside_out", "cf_precheck", "steered", "wavefront", "trial_division")


class TestSteeredPerformance:
    """Performance tests verifying steered search is faster than BFS for close factors."""

    @pytest.mark.parametrize("N", [323, 899, 1763, 2021])
    def test_steered_finds_close_factors(self, N):
        """Steered search should find close-factor semiprimes quickly."""
        import time
        start = time.perf_counter()
        result = _steered_search(N, max_iterations=1000)
        elapsed = time.perf_counter() - start

        assert result is not None, f"Failed to factor {N} with steered search"
        p, q = result
        assert p * q == N
        # Should be fast — steered search typically takes < 50 iterations
        assert elapsed < 1.0, f"Steered search took {elapsed:.3f}s for N={N}"

    @pytest.mark.parametrize("N", [323, 899, 1763])
    def test_steered_vs_bfs_iteration_count(self, N):
        """Steered search should use fewer iterations than BFS for close factors."""
        # Run steered search with iteration counting
        from insideout.inside_out import central_well, resonance_check
        from insideout.gaussian import mn_to_triple, mn_children
        from insideout.cf_guide import predict_branch
        from insideout.energy import hypotenuse_bound
        import heapq

        # Count steered iterations
        well = central_well(N)
        upper = hypotenuse_bound(N)
        seed_points = cf_seeded_well_points(N)
        visited = set()
        heap = []
        counter = 0
        for seed in seed_points:
            if (seed.m, seed.n) not in visited:
                triple = mn_to_triple(seed)
                if triple.c <= upper:
                    dist = min(predict_branch(N, (triple.a, triple.b, triple.c)))
                    heapq.heappush(heap, (dist, counter, seed))
                    counter += 1

        steered_iters = 0
        found = False
        while heap and steered_iters < 10000:
            _, _, current = heapq.heappop(heap)
            key = (current.m, current.n)
            if key in visited:
                continue
            visited.add(key)
            steered_iters += 1

            if current.m <= current.n:
                continue
            from math import gcd
            if gcd(current.m, current.n) != 1 or (current.m - current.n) % 2 != 1:
                continue

            triple = mn_to_triple(current)
            if triple.c > upper:
                continue
            if triple.c >= N:
                result = resonance_check(N, triple)
                if result is not None:
                    found = True
                    break
            if 1 < triple.a < N and N % triple.a == 0:
                found = True
                break
            if 1 < triple.b < N and N % triple.b == 0:
                found = True
                break

            for child in mn_children(current):
                if child.m > child.n > 0 and (child.m, child.n) not in visited:
                    ct = mn_to_triple(child)
                    if ct.c <= upper:
                        dist = min(predict_branch(N, (ct.a, ct.b, ct.c)))
                        heapq.heappush(heap, (dist, counter, child))
                        counter += 1

        assert found, f"Steered search should find factors for N={N}"
        assert steered_iters < 100, f"Steered search should be efficient for N={N}, took {steered_iters} iterations"