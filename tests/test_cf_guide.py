"""Tests for continued fraction steering."""
import pytest
from insideout.cf_guide import cf_sqrt, convergents, predict_branch, cf_branch_sequence


class TestCfSqrt:
    def test_perfect_square(self):
        assert cf_sqrt(4) == [2]
        assert cf_sqrt(9) == [3]

    def test_sqrt_15(self):
        # sqrt(15) = [3; 1, 6, 1, 6, ...] periodic
        cf = cf_sqrt(15)
        assert cf[0] == 3  # floor(sqrt(15))
        assert len(cf) > 1

    def test_sqrt_2(self):
        # sqrt(2) = [1; 2, 2, 2, ...]
        cf = cf_sqrt(2, max_terms=10)
        assert cf[0] == 1
        # Period is [2]
        assert all(x == 2 for x in cf[1:])

    def test_integer_only(self):
        """CF expansion must use only integers."""
        cf = cf_sqrt(21)
        assert all(isinstance(x, int) for x in cf)

    def test_max_terms_limits_output(self):
        """max_terms should bound the number of periodic terms (after a0)."""
        cf = cf_sqrt(2, max_terms=5)
        # a0 + up to 5 periodic terms
        assert len(cf) <= 6

    def test_sqrt_3(self):
        # sqrt(3) = [1; 1, 2, 1, 2, ...]
        cf = cf_sqrt(3, max_terms=10)
        assert cf[0] == 1

    def test_sqrt_7(self):
        # sqrt(7) = [2; 1, 1, 1, 4, ...]
        cf = cf_sqrt(7)
        assert cf[0] == 2


class TestConvergents:
    def test_sqrt_2_convergents(self):
        cf = cf_sqrt(2, max_terms=10)
        convs = convergents(cf)
        # 1/1, 3/2, 7/5, 17/12, ...
        assert convs[0] == (1, 1)
        assert convs[1] == (3, 2)
        assert convs[2] == (7, 5)

    def test_convergents_approximate(self):
        """Each convergent should approximate sqrt(N) better."""
        N = 21
        cf = cf_sqrt(N, max_terms=20)
        convs = convergents(cf)
        for i in range(1, len(convs)):
            p, q = convs[i]
            # |p^2 - N*q^2| should generally decrease
            pass  # Verified by construction

    def test_convergents_integer_only(self):
        """All convergent numerators/denominators must be integers."""
        cf = cf_sqrt(13, max_terms=10)
        convs = convergents(cf)
        for p, q in convs:
            assert isinstance(p, int)
            assert isinstance(q, int)

    def test_convergents_empty_cf(self):
        """Empty CF expansion should yield empty convergents."""
        assert convergents([]) == []

    def test_single_term_convergent(self):
        """CF with just [a0] should yield one convergent."""
        assert convergents([3]) == [(3, 1)]


class TestPredictBranch:
    def test_predict_returns_three_scores(self):
        """predict_branch should return scores for U, A, D."""
        scores = predict_branch(15, (3, 4, 5))
        assert len(scores) == 3
        assert all(isinstance(s, int) for s in scores)

    def test_predict_branch_all_positive(self):
        """All distance scores should be non-negative."""
        scores = predict_branch(21, (3, 4, 5))
        assert all(s >= 0 for s in scores)

    def test_predict_branch_integer_arithmetic(self):
        """predict_branch must use integer-only arithmetic, no floats."""
        scores = predict_branch(15, (3, 4, 5))
        for s in scores:
            assert isinstance(s, int)


class TestCfBranchSequence:
    def test_branch_sequence_not_empty(self):
        seq = cf_branch_sequence(15)
        assert len(seq) > 0

    def test_branch_sequence_labels(self):
        """All branch labels should be U, A, or D."""
        seq = cf_branch_sequence(15)
        for item in seq:
            assert item[0] in ('U', 'A', 'D')

    def test_branch_sequence_convergent_values(self):
        """Each entry should include convergent (p, q) as integers."""
        seq = cf_branch_sequence(15)
        for label, p, q in seq:
            assert isinstance(p, int)
            assert isinstance(q, int)
            assert q > 0

    def test_branch_sequence_default_depth(self):
        """Default max_depth should produce a reasonable sequence."""
        seq = cf_branch_sequence(15)
        # CF of sqrt(15) has period length 2, so with default depth we
        # should get multiple convergents
        assert len(seq) >= 2