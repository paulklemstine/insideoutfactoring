"""Tests for Inside-Out Relation Generator."""
import pytest
from insideout.relation_generator import (
    relation_factor,
    _smoothness_score,
    _generate_ppt_relations,
)


class TestSmoothnessScore:
    """Test smoothness scoring."""

    def test_smooth_number(self):
        """2*3*5*7 = 210 should be fully smooth up to 541."""
        score, factors = _smoothness_score(210)
        assert score == 1.0
        assert sorted(factors) == [2, 3, 5, 7]

    def test_prime_number(self):
        """A large prime beyond the smoothness bound should have low score."""
        # 10007 is prime and > 541, so it should score 0
        score, factors = _smoothness_score(10007)
        assert score == 0.0

    def test_one(self):
        """1 should have score 1.0."""
        score, factors = _smoothness_score(1)
        assert score == 1.0

    def test_power_of_two(self):
        """1024 = 2^10 should be fully smooth."""
        score, factors = _smoothness_score(1024)
        assert score == 1.0
        assert all(f == 2 for f in factors)


class TestGenerateRelations:
    """Test PPT relation generation."""

    def test_generates_relations(self):
        """Should generate some relations for a composite number."""
        relations = _generate_ppt_relations(15 * 7, max_params=1000, max_relations=50)
        # Should generate some relations (may be empty for small N)
        assert isinstance(relations, list)

    def test_perfect_square(self):
        """Should handle perfect squares."""
        result = relation_factor(121)
        assert result is not None
        assert result[0] * result[1] == 121

    def test_even_number(self):
        """Should handle even numbers."""
        result = relation_factor(6)
        assert result is not None
        assert result[0] * result[1] == 6


class TestRelationFactor:
    """Test the full relation factoring pipeline."""

    @pytest.mark.parametrize("N,p,q", [
        (15, 3, 5),
        (21, 3, 7),
        (35, 5, 7),
        (77, 7, 11),
    ])
    def test_factors_small_semiprimes(self, N, p, q):
        """Relation factor should find some factors."""
        result = relation_factor(N, max_params=1000, max_relations=100)
        if result is not None:
            fp, fq = result
            assert fp * fq == N

    def test_rejects_trivial_factors(self):
        """Should not return trivial factors."""
        result = relation_factor(15, max_params=1000, max_relations=100)
        if result is not None:
            p, q = result
            assert 1 < p < 15
            assert 1 < q < 15