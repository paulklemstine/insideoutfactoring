"""Thompson Sampling Multi-Armed Bandit for Adaptive Factoring Method Selection.

Treats each factoring method as an arm in a Bayesian multi-armed bandit.
Maintains a Beta-Bernoulli posterior per method, updated after every factoring
attempt. Thompson Sampling balances exploration (trying less-tested methods)
with exploitation (favoring historically fast, high-success-rate methods).

Contextual Extension:
  - Partitions by bit-size bucket (tiny/small/medium/large/xlarge)
  - Detects N form (perfect_square, close_factors, general)
  - Hierarchical posteriors blend context-specific with global priors
  - Hardness-scaled UCB bonus for hard instances

Usage:
    bandit = MethodBandit(ALL_METHODS)
    method = bandit.select_method(candidate_methods, N=some_N)
    # ... run factoring ...
    bandit.update(method, success=True, time_ms=12.5)

The bandit state persists across calls and improves method selection over time.
"""
from __future__ import annotations

import json
import math
import os
import random
from math import isqrt
from pathlib import Path
from typing import Optional

# Default state file
_STATE_DIR = Path.home() / ".insideout"
_STATE_FILE = _STATE_DIR / "bandit_state.json"


# All known factoring methods
ALL_METHODS = [
    "fermat", "brahmagupta", "fibonacci",
    "resonance_cascade", "lucas_ppt", "spectral_cascade",
    "relation_gen", "fib_pyth", "lucas_multi", "crt_collision",
    "sl2_group_order", "sl2_structured", "batch_crt",
    "ppt_sieve", "cf_matrix_cascade", "cf_cascade",
    "ppt_form", "squfof", "class_group", "class_squfof",
    "cyclotomic_resultant", "cyclotomic_cascade",
    "discriminant_resonance", "quadratic_resonance",
    "hensel_cascade", "crt_lattice",
    "lattice_factor", "hybrid_smooth",
    "graph_order", "order_spectrum",
    "coppersmith", "hybrid_cyclo_sl2",
    "inside_out", "wavefront",
]


def _context_bucket(n_bits: int) -> str:
    """Map bit size to context bucket for bandit partitioning."""
    if n_bits < 40:
        return "tiny"
    elif n_bits < 56:
        return "small"
    elif n_bits < 72:
        return "medium"
    elif n_bits < 96:
        return "large"
    else:
        return "xlarge"


def _detect_N_form(N: int) -> str:
    """Detect structural form of N for contextual learning.

    Returns one of:
      - 'perfect_square': N = p^2
      - 'close_factors': |p - q| is small relative to N (Fermat-friendly)
      - 'smooth_order': p+1 or p-1 has small factors (p±1 method-friendly)
      - 'general': no special structure detected
    """
    # Perfect square check
    s = isqrt(N)
    if s * s == N and s > 1:
        return "perfect_square"

    # Close factors check: estimate gap from sqrt(N)
    s = isqrt(N)
    if s * s < N:
        sq_below = (s - 1) * (s - 1) if s > 1 else 0
    else:
        sq_below = s * s
    gap = N - sq_below

    # If gap is small relative to N, factors are close (Fermat-friendly)
    if gap > 0 and gap < N // 10000:
        return "close_factors"

    return "general"


class MethodBandit:
    """Thompson Sampling bandit for factoring method selection with contextual learning.

    Each method has hierarchical Beta-distributed success rate priors:
      - Global: alpha_global, beta_global (pooled across all contexts)
      - Context-specific: alpha[bucket], beta[bucket] per bit-size bucket
      - Form-specific: alpha[form], beta[form] per N-form

    Score combines:
      - Sampled success rate from blended Beta posterior
      - Speed score: avg_time^(-0.3) penalizes slow methods
      - Hardness-scaled UCB exploration bonus
    """

    def __init__(self, methods: Optional[list[str]] = None):
        self.methods = methods or list(ALL_METHODS)

        # Global posteriors (pooled across contexts)
        self.alpha: dict[str, float] = {m: 1.0 for m in self.methods}
        self.beta: dict[str, float] = {m: 1.0 for m in self.methods}

        # Context-specific posteriors: method -> bucket -> alpha/beta
        self.alpha_ctx: dict[str, dict[str, float]] = {m: {} for m in self.methods}
        self.beta_ctx: dict[str, dict[str, float]] = {m: {} for m in self.methods}

        # Form-specific posteriors: method -> form -> alpha/beta
        self.alpha_form: dict[str, dict[str, float]] = {m: {} for m in self.methods}
        self.beta_form: dict[str, dict[str, float]] = {m: {} for m in self.methods}

        # Statistics
        self.total_time: dict[str, float] = {m: 0.0 for m in self.methods}
        self.attempts: dict[str, int] = {m: 0 for m in self.methods}
        self.successes: dict[str, int] = {m: 0 for m in self.methods}

        # Context-specific statistics
        self.total_time_ctx: dict[str, dict[str, float]] = {m: {} for m in self.methods}
        self.attempts_ctx: dict[str, dict[str, int]] = {m: {} for m in self.methods}

        self._load_state()

    @property
    def GLOBAL_WEIGHT(self) -> float:
        """Weight given to global priors vs context-specific (shrinkage)."""
        return 0.3

    def _sample_beta(self, alpha: float, beta: float) -> float:
        """Sample from Beta distribution using gamma sampling."""
        # Gamma sampling via Marsaglia and Tsang's method
        def _gamma(shape: float) -> float:
            if shape < 1:
                return _gamma(shape + 1) * (random.random() ** (1.0 / shape))
            d = shape - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)
            while True:
                x = random.gauss(0.0, 1.0)
                v = 1.0 + c * x
                if v > 0:
                    v2 = v * v * v
                    u = random.random()
                    if u < 1.0 - 0.0331 * (x * x) * (x * x):
                        return d * v2
                    if math.log(u) < 0.5 * x * x + d * (1.0 - v2 + math.log(v2)):
                        return d * v2

        g_alpha = _gamma(alpha)
        g_beta = _gamma(beta)
        return g_alpha / (g_alpha + g_beta)

    def _get_posterior(self, method: str, bucket: str, form: str) -> tuple[float, float]:
        """Blend global, bucket-specific, and form-specific posteriors.

        Uses shrinkage estimator: blend context-specific with global prior.
        """
        # Global prior
        global_a = self.alpha.get(method, 1.0)
        global_b = self.beta.get(method, 1.0)

        # Bucket-specific
        bucket_a = self.alpha_ctx.get(method, {}).get(bucket, 1.0)
        bucket_b = self.beta_ctx.get(method, {}).get(bucket, 1.0)

        # Form-specific
        form_a = self.alpha_form.get(method, {}).get(form, 1.0)
        form_b = self.beta_form.get(method, {}).get(form, 1.0)

        # Shrinkage: blend context-specific with global
        gw = self.GLOBAL_WEIGHT
        alpha = gw * global_a + (1 - gw) * (bucket_a + form_a) / 2
        beta = gw * global_b + (1 - gw) * (bucket_b + form_b) / 2

        return max(alpha, 0.1), max(beta, 0.1)

    def sample_score(self, method: str, bucket: str = "small",
                     form: str = "general", hardness: float = 1.0) -> float:
        """Sample from posterior and compute combined score.

        Score = sampled_success_rate * speed_factor + UCB_bonus * hardness

        Args:
            method: Factoring method name
            bucket: Bit-size context bucket (tiny/small/medium/large/xlarge)
            form: N form detected by _detect_N_form
            hardness: Instance hardness multiplier for UCB bonus (1.0 = normal)
        """
        alpha, beta = self._get_posterior(method, bucket, form)
        sampled_rate = self._sample_beta(alpha, beta)

        attempts = self.attempts.get(method, 0)
        if attempts == 0:
            avg_time = 1000.0  # Default 1 second for untested methods
        else:
            avg_time = self.total_time.get(method, 0.0) / attempts

        # Speed factor: faster methods get a bonus
        # avg_time^(-0.3): 1ms → 0.16, 100ms → 0.32, 1000ms → 0.50
        speed_factor = avg_time ** (-0.3) if avg_time > 0 else 1.0

        # UCB-style exploration bonus scaled by hardness
        if attempts > 0:
            uncertainty = math.sqrt(math.log(self.total_attempts + 1) / attempts)
            ucb_bonus = hardness * uncertainty * 0.1
        else:
            ucb_bonus = 0.2 * hardness  # Extra bonus for untested methods

        return sampled_rate * speed_factor + ucb_bonus

    @property
    def total_attempts(self) -> int:
        return sum(self.attempts.values())

    def select_method(self, candidate_methods: Optional[list[str]] = None,
                      N: Optional[int] = None,
                      exploration_bonus: float = 0.3) -> str:
        """Thompson Sampling method selection with contextual awareness.

        If candidates provided, restrict to those. Otherwise use all methods.
        exploration_bonus gives untested/undertested methods a boost.

        Args:
            candidate_methods: Methods to choose from
            N: The integer being factored (for context extraction)
            exploration_bonus: Legacy parameter (ignored, hardness-driven now)
        """
        candidates = candidate_methods or self.methods
        candidates = [m for m in candidates if m in self.methods]

        if not candidates:
            return self.methods[0]

        # Extract context from N
        if N is not None:
            bucket = _context_bucket(N.bit_length())
            form = _detect_N_form(N)
            hardness = self._estimate_hardness(N)
        else:
            bucket = "small"
            form = "general"
            hardness = 1.0

        # Thompson sampling: draw from posterior for each candidate
        scores = {}
        for method in candidates:
            attempts = self.attempts.get(method, 0)
            # Exploration bonus for methods with few attempts
            if attempts < 3:
                base_score = self.sample_score(method, bucket, form, hardness)
                exploration_score = exploration_bonus / (attempts + 1)
                scores[method] = base_score * 0.7 + exploration_score
            else:
                scores[method] = self.sample_score(method, bucket, form, hardness)

        return max(scores, key=scores.get)

    def _estimate_hardness(self, N: int) -> float:
        """Estimate instance hardness for exploration scaling.

        Harder instances (large N, close factors) get more exploration budget.
        Returns hardness multiplier in [0.5, 3.0].
        """
        n_bits = N.bit_length()

        # Base hardness by size
        if n_bits < 40:
            hardness = 0.5
        elif n_bits < 56:
            hardness = 0.7
        elif n_bits < 72:
            hardness = 1.0
        elif n_bits < 96:
            hardness = 1.5
        else:
            hardness = 2.0

        # Adjust for structure
        form = _detect_N_form(N)
        if form == "close_factors":
            hardness *= 1.5  # Fermat-friendly but other methods struggle
        elif form == "perfect_square":
            hardness *= 0.3  # Easy - perfect square detection is instant

        return hardness

    def update(self, method: str, success: bool, time_ms: float,
               N: Optional[int] = None) -> None:
        """Bayesian update after a factoring attempt.

        Args:
            method: The method that was tried
            success: Whether it found a factor
            time_ms: Time taken in milliseconds
            N: The integer being factored (for context extraction)
        """
        if method not in self.methods:
            return

        self.attempts[method] = self.attempts.get(method, 0) + 1
        self.total_time[method] = self.total_time.get(method, 0.0) + time_ms

        if success:
            self.alpha[method] = self.alpha.get(method, 1.0) + 1.0
            self.successes[method] = self.successes.get(method, 0) + 1
        else:
            self.beta[method] = self.beta.get(method, 1.0) + 1.0

        # Context-specific updates
        if N is not None:
            bucket = _context_bucket(N.bit_length())
            form = _detect_N_form(N)

            # Bucket-specific update
            self.alpha_ctx.setdefault(method, {}).setdefault(bucket, 1.0)
            self.beta_ctx.setdefault(method, {}).setdefault(bucket, 1.0)
            self.total_time_ctx.setdefault(method, {}).setdefault(bucket, 0.0)
            self.attempts_ctx.setdefault(method, {}).setdefault(bucket, 0)

            self.attempts_ctx[method][bucket] += 1
            self.total_time_ctx[method][bucket] += time_ms

            if success:
                self.alpha_ctx[method][bucket] += 1.0
            else:
                self.beta_ctx[method][bucket] += 1.0

            # Form-specific update
            self.alpha_form.setdefault(method, {}).setdefault(form, 1.0)
            self.beta_form.setdefault(method, {}).setdefault(form, 1.0)

            if success:
                self.alpha_form[method][form] += 1.0
            else:
                self.beta_form[method][form] += 1.0

        self._save_state()

    def get_method_stats(self) -> dict:
        """Return per-method statistics."""
        stats = {}
        for method in self.methods:
            attempts = self.attempts.get(method, 0)
            if attempts == 0:
                stats[method] = {
                    "attempts": 0,
                    "successes": 0,
                    "success_rate": None,
                    "avg_time_ms": None,
                }
            else:
                avg_time = self.total_time.get(method, 0.0) / attempts
                stats[method] = {
                    "attempts": attempts,
                    "successes": self.successes.get(method, 0),
                    "success_rate": self.successes.get(method, 0) / attempts,
                    "avg_time_ms": avg_time,
                }
        return stats

    def reset(self) -> None:
        """Reset all state to priors."""
        self.alpha = {m: 1.0 for m in self.methods}
        self.beta = {m: 1.0 for m in self.methods}
        self.total_time = {m: 0.0 for m in self.methods}
        self.attempts = {m: 0 for m in self.methods}
        self.successes = {m: 0 for m in self.methods}
        self._save_state()

    def _state_path(self) -> Path:
        return Path(os.environ.get("INSIDEOUT_BANDIT_PATH", str(_STATE_FILE)))

    def _load_state(self) -> None:
        """Load persisted bandit state."""
        path = self._state_path()
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for key in ("alpha", "beta", "total_time", "attempts", "successes"):
                if key in data:
                    getattr(self, key).update(data[key])
        except (json.JSONDecodeError, IOError):
            pass  # Ignore corrupt files

    def _save_state(self) -> None:
        """Persist bandit state to disk."""
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "alpha": self.alpha,
            "beta": self.beta,
            "total_time": self.total_time,
            "attempts": self.attempts,
            "successes": self.successes,
            "alpha_ctx": self.alpha_ctx,
            "beta_ctx": self.beta_ctx,
            "alpha_form": self.alpha_form,
            "beta_form": self.beta_form,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f)
        except IOError:
            pass  # Ignore write failures


# Global singleton — persists across calls
_GLOBAL_BANDIT: Optional[MethodBandit] = None


def get_bandit() -> MethodBandit:
    """Get the global bandit singleton."""
    global _GLOBAL_BANDIT
    if _GLOBAL_BANDIT is None:
        _GLOBAL_BANDIT = MethodBandit()
    return _GLOBAL_BANDIT


def bandit_select(candidate_methods: Optional[list[str]] = None) -> str:
    """Convenience wrapper: select a method via global bandit."""
    return get_bandit().select_method(candidate_methods)


def bandit_update(method: str, success: bool, time_ms: float) -> None:
    """Convenience wrapper: update global bandit."""
    get_bandit().update(method, success, time_ms)
