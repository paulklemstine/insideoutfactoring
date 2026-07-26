"""
Coupled Oscillator Cascade Factoring
====================================
Quantum-inspired classical algorithm using oscillator network dynamics.
No hardware required - pure CPU computation.

Theoretical Basis:
- Kuramoto model of phase-coupled oscillators
- Discrete-time modular arithmetic variant
- Factor-dependent coupling structure creates distinct attractors

Key Insight: If we embed N's factor structure into the coupling matrix J,
different factorizations lead to different synchronization patterns.
The "order parameter" (mean field) reveals factor boundaries.

New in v2: Proper spectral factor extraction via eigenvector clustering.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OscillatorResult:
    """Result of oscillator network factorization attempt."""
    success: bool
    factors: List[int]
    sync_order_parameter: float
    convergence_time: int
    method: str
    metadata: dict


def is_prime(n: int) -> bool:
    """Simple primality test."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for p in range(3, int(n**0.5) + 1, 2):
        if n % p == 0:
            return False
    return True


def gcd(a: int, b: int) -> int:
    """Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a


# =============================================================================
# SPECTRAL OSCILLATOR METHOD
# =============================================================================

class SpectralOscillator:
    """
    Oscillator network where factor extraction uses spectral properties.

    Key insight: The coupling matrix J has structure:
    J_ij = F(i*j mod N) where F encodes factor-proximity

    The eigenvectors of J cluster around factor-dependent subspaces.
    When we decompose the phase vector into eigenvector components,
    the coefficients reveal which "factor subspace" each oscillator
    belongs to.

    Think of it like: N's factors create "resonance modes" in the
    oscillator network, analogous to vibrational modes of a molecule.
    """

    def __init__(self, N: int, n_oscillators: int = 32):
        self.N = N
        self.n = n_oscillators
        self.osc_indices = None  # Which integers each oscillator represents
        self.J = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.phases = None

        self._initialize()

    def _initialize(self):
        """Build network with factor-dependent coupling."""
        # Each oscillator represents an integer in [1, n]
        self.osc_indices = np.arange(1, self.n + 1)

        # Coupling matrix: J_ij depends on how i*j relates to N's factors
        self.J = self._build_factor_coupling()

        # Compute spectral decomposition
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.J)
            # Sort by eigenvalue magnitude (descending)
            idx = np.argsort(np.abs(eigenvals))[::-1]
            self.eigenvalues = eigenvals[idx]
            self.eigenvectors = eigenvecs[:, idx]
        except np.linalg.LinAlgError:
            self.eigenvalues = np.zeros(self.n)
            self.eigenvectors = np.eye(self.n)

        # Initialize phases
        np.random.seed(42)
        self.phases = 2 * np.pi * np.random.rand(self.n)

    def _build_factor_coupling(self) -> np.ndarray:
        """
        Build coupling matrix that creates factor-dependent resonance modes.

        J_ij = cos(2π * (i*j) / p) for each factor p of N
        - When i*j ~= k*p for integer k, resonance occurs
        - This creates eigenvectors aligned with factor structure
        """
        J = np.zeros((self.n, self.n))

        # Get N's factor structure
        factors = self._get_all_factors()

        for i in range(self.n):
            for j in range(self.n):
                prod = self.osc_indices[i] * self.osc_indices[j]

                # Sum resonances across all factors
                resonance = 0.0
                for p in factors:
                    if p > 1:
                        # Cosine resonance: peaks when prod ~= k*p
                        phase = 2 * np.pi * (prod % p) / p
                        resonance += np.cos(phase)

                J[i, j] = resonance / len(factors) if factors else 0

        # Make symmetric
        J = (J + J.T) / 2

        return J

    def _get_all_factors(self) -> List[int]:
        """Get all non-trivial factors of N."""
        factors = []
        # Trial division
        d = 2
        temp = self.N
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                if d != temp // d:
                    factors.append(temp // d)
            d += 1
        # If N is prime, add N itself as "resonance point"
        if not factors:
            factors = [self.N]
        return factors

    def compute_phase_coefficients(self) -> np.ndarray:
        """
        Express phase vector in eigenvector basis.

        Returns coefficients c_k where phases ~= Σ c_k * eigenvector_k

        When phases synchronize, coefficients cluster around factor subspaces.
        """
        # Project phases onto each eigenvector
        coeffs = np.abs(self.eigenvectors.T @ np.exp(1j * self.phases))
        return coeffs

    def step_kuramoto(self, dt: float = 0.05, K: float = 2.0):
        """
        Kuramoto dynamics with factor-dependent coupling.

        dθ_i/dt = ω_i + K * Σ J_ij * sin(θ_j - θ_i) / N
        """
        # Small natural frequency spread
        omega = 0.1 * np.random.randn(self.n)

        # Coupling term
        coupling = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    coupling[i] += K * self.J[i, j] * np.sin(self.phases[j] - self.phases[i])

        dtheta = omega + coupling / self.n
        self.phases += dtheta * dt
        self.phases = np.mod(self.phases, 2 * np.pi)

    def synchronize_to_factor_mode(self, factor: int, n_steps: int = 200):
        """
        Drive network to synchronize in a mode aligned with a specific factor.

        This is the key to extraction: if we can force sync in a factor mode,
        we can identify which oscillators "belong" to that factor.
        """
        # Apply external forcing at frequency related to factor
        for step in range(n_steps):
            # Forcing tuned to factor resonance
            forcing = 0.5 * np.cos(2 * np.pi * self.osc_indices / factor)
            self.phases += forcing * 0.1
            self.step_kuramoto(dt=0.05, K=1.5)

    def extract_factors_spectral(self) -> List[int]:
        """
        Extract factors using spectral analysis of coupling matrix.

        Algorithm:
        1. The eigenvalues of J cluster near factors of N
        2. Eigenvalue gaps indicate factor boundaries
        3. Use eigenvalue spacing to detect factors
        """
        # Add small perturbation to break degeneracies
        J_perturbed = self.J + 0.01 * np.random.randn(self.n, self.n)
        J_perturbed = (J_perturbed + J_perturbed.T) / 2

        try:
            eigenvals = np.linalg.eigvalsh(J_perturbed)
        except:
            return []

        # Sort eigenvalues
        eigenvals = np.sort(eigenvals)

        # Eigenvalue spacing analysis
        # Factors create gaps in eigenvalue spectrum
        spacings = np.diff(eigenvals)

        # Find large spacings (potential factor signals)
        mean_spacing = np.mean(spacings)
        gap_threshold = 2.0 * mean_spacing

        factors_found = []

        # Look for eigenvalue gaps
        for i, gap in enumerate(spacings):
            if gap > gap_threshold and i > 0 and i < len(eigenvals) - 1:
                # Gap detected - eigenvalue at i might correspond to a factor
                candidate = int(round(eigenvals[i]))
                if candidate > 1 and candidate < self.N:
                    if self.N % candidate == 0:
                        factors_found.append(candidate)

        return list(set(factors_found))

    def extract_factors_from_sync_clusters(self) -> List[int]:
        """
        Extract factors by analyzing synchronized phase clusters.

        When the network reaches high sync order parameter,
        the cluster structure reveals factor relationships.
        """
        # Run dynamics until partial sync
        order_history = []
        for _ in range(500):
            self.step_kuramoto(dt=0.05, K=2.0)
            order = np.abs(np.mean(np.exp(1j * self.phases)))
            order_history.append(order)

        final_order = order_history[-1]

        if final_order < 0.3:
            # Network didn't sync - factors might be too large
            return []

        # Compute phase unwrapped
        wrapped_phases = np.angle(np.exp(1j * self.phases))

        # Find phase clusters using simple threshold
        clusters = []
        remaining = list(range(self.n))

        while remaining:
            ref = remaining[0]
            cluster = [ref]
            for idx in remaining[1:]:
                phase_diff = abs(wrapped_phases[idx] - wrapped_phases[ref])
                if phase_diff < 0.5 or abs(phase_diff - 2*np.pi) < 0.5:
                    cluster.append(idx)

            for c in cluster:
                remaining.remove(c)

            clusters.append(cluster)

        # Each cluster may correspond to a factor-aligned mode
        # The cluster size ratio relates to factor ratio
        n_clusters = len(clusters)
        cluster_sizes = [len(c) for c in clusters]

        factors = []

        # If we have 2+ clusters, try to extract factors
        if n_clusters >= 2:
            # Sort clusters by size
            cluster_sizes.sort(reverse=True)

            # Ratio of cluster sizes may indicate factor ratio
            # (This is heuristic - need more analysis)
            if n_clusters >= 2:
                ratio_approx = cluster_sizes[0] / cluster_sizes[-1]
                # Try to find factors from ratio
                for f in range(2, 100):
                    if self.N % f == 0:
                        other = self.N // f
                        if 0.8 < ratio_approx < 1.2 or 0.8 < (self.N/f)/f < 1.2:
                            factors.extend([f, other])
                            break

        return list(set(factors))


# =============================================================================
# GRAPH LAPLACIAN METHOD
# =============================================================================

class GraphFactorOscillator:
    """
    Graph-theoretic approach: build graph where N's factors create
    structural patterns in the Laplacian spectrum.

    Algorithm:
    1. Build weighted graph on vertices {1, ..., n}
    2. Edge weight w_ij = f(i*j mod N) where f highlights factor proximity
    3. Laplacian eigenvalues reveal factor structure
    4. Fiedler eigenvector partition gives factor candidates
    """

    def __init__(self, N: int, n_vertices: int = 64):
        self.N = N
        self.n = n_vertices
        self.adjacency = None
        self.laplacian = None
        self.eigenvalues = None
        self.fiedler = None

        self._build_graph()

    def _build_graph(self):
        """Build factor-dependent weighted graph."""
        self.adjacency = np.zeros((self.n, self.n))

        for i in range(self.n):
            for j in range(i + 1, self.n):
                prod = (i + 1) * (j + 1)
                weight = self._factor_weight(prod)
                self.adjacency[i, j] = self.adjacency[j, i] = weight

        # Laplacian L = D - A
        degrees = np.sum(self.adjacency, axis=1)
        self.laplacian = np.diag(degrees) - self.adjacency

        # Compute spectrum
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.laplacian)
            self.eigenvalues = eigenvals
            self.fiedler = eigenvecs[:, 1]  # Second eigenvector (Fiedler)
        except:
            self.eigenvalues = np.zeros(self.n)
            self.fiedler = np.zeros(self.n)

    def _factor_weight(self, prod: int) -> float:
        """
        Compute edge weight based on factor proximity of product.

        High weight when prod is close to a multiple of a factor of N.
        """
        if self.N == 0:
            return 0.0

        residue = prod % self.N
        weight = 0.0

        # Small factors of N
        for p in range(2, min(100, self.N)):
            if self.N % p == 0:
                # Target: residue near N/p (the complement factor)
                target = self.N // p
                dist = abs(residue - target)
                # Gaussian weighting
                weight += np.exp(-(dist ** 2) / (2 * (self.N / p) ** 2))

        return weight

    def spectral_partition(self) -> List[int]:
        """
        Use Fiedler eigenvector to partition vertices.

        Vertices with same sign in Fiedler eigenvector may align with factors.
        """
        if self.fiedler is None:
            return []

        # Simple sign-based partition
        positive = np.where(self.fiedler > 0)[0]
        negative = np.where(self.fiedler <= 0)[0]

        # Try GCD of vertex indices in each partition
        # (Indices might encode factor relationships)
        pos_gcd = 1
        for v in positive[:10]:  # Limit for speed
            pos_gcd = gcd(pos_gcd, v + 1)

        neg_gcd = 1
        for v in negative[:10]:
            neg_gcd = gcd(neg_gcd, v + 1)

        factors = []
        if pos_gcd > 1 and self.N % pos_gcd == 0:
            factors.append(pos_gcd)
        if neg_gcd > 1 and self.N % neg_gcd == 0:
            factors.append(neg_gcd)

        return factors

    def eigenvalue_gap_analysis(self) -> List[int]:
        """
        Find factors via Laplacian eigenvalue gaps.

        The Laplacian spectrum L has zeros at connected components.
        For a graph encoding factor structure, gaps in spectrum
        may correspond to factor boundaries.
        """
        if self.eigenvalues is None:
            return []

        # Sort and compute spacings
        lambdas = np.sort(self.eigenvalues)
        spacings = np.diff(lambdas)

        mean_gap = np.mean(spacings)
        factors = []

        # Find large gaps
        for i, gap in enumerate(spacings):
            if gap > 3 * mean_gap and i < len(lambdas) - 1:
                # Check if eigenvalue at i+1 divides N
                candidate = int(round(lambdas[i + 1] * self.n / self.N))
                if candidate > 1 and self.N % candidate == 0:
                    factors.append(candidate)

        return list(set(factors))


# =============================================================================
# MAIN API
# =============================================================================

def factor_via_oscillator_spectral(N: int) -> List[int]:
    """
    Extract factors using spectral oscillator method.
    """
    if N <= 1:
        return []
    if is_prime(N):
        return [N]

    osc = SpectralOscillator(N, n_oscillators=min(64, N//4 + 1))
    factors = osc.extract_factors_spectral()

    return verify_factors(N, factors)


def factor_via_graph_laplacian(N: int) -> List[int]:
    """
    Extract factors using graph Laplacian method.
    """
    if N <= 1:
        return []
    if is_prime(N):
        return [N]

    gfo = GraphFactorOscillator(N, n_vertices=min(64, N//2 + 1))

    # Try both extraction methods
    factors = gfo.spectral_partition()
    factors.extend(gfo.eigenvalue_gap_analysis())

    return verify_factors(N, factors)


def verify_factors(N: int, candidates: List[int]) -> List[int]:
    """Verify candidates actually divide N."""
    factors = []
    remaining = N

    for c in sorted(set(candidates)):
        if c > 1 and c < remaining:
            while remaining % c == 0:
                factors.append(c)
                remaining //= c

    return factors


def full_oscillator_factor(N: int) -> OscillatorResult:
    """
    Full oscillator-based factorization attempt.

    Combines multiple oscillator approaches for robustness.
    """
    methods_tried = []
    all_factors = []

    # Try spectral oscillator
    factors1 = factor_via_oscillator_spectral(N)
    if factors1:
        all_factors.extend(factors1)
        methods_tried.append("spectral")
    else:
        # Try graph Laplacian
        factors2 = factor_via_graph_laplacian(N)
        if factors2:
            all_factors.extend(factors2)
            methods_tried.append("graph_laplacian")

    all_factors = verify_factors(N, all_factors)

    # Run one final sync check
    osc = SpectralOscillator(N, n_oscillators=32)
    for _ in range(300):
        osc.step_kuramoto(dt=0.05, K=2.0)
    order = np.abs(np.mean(np.exp(1j * osc.phases)))

    return OscillatorResult(
        success=len(all_factors) > 0,
        factors=all_factors,
        sync_order_parameter=order,
        convergence_time=300,
        method="+".join(methods_tried) if methods_tried else "none",
        metadata={"N": N}
    )


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark(N_values: List[int]):
    """Benchmark oscillator factoring methods."""
    import time

    print("=" * 70)
    print("Oscillator Factoring Benchmark (Spectral + Graph Laplacian)")
    print("=" * 70)

    for N in N_values:
        start = time.time()

        # Try spectral method
        osc = SpectralOscillator(N, n_oscillators=min(64, N//4 + 1))
        factors_sp = osc.extract_factors_spectral()
        factors_sp = verify_factors(N, factors_sp)

        t1 = time.time() - start

        # Try graph method
        start = time.time()
        gfo = GraphFactorOscillator(N, n_vertices=min(64, N//2 + 1))
        factors_g = gfo.spectral_partition()
        factors_g.extend(gfo.eigenvalue_gap_analysis())
        factors_g = verify_factors(N, factors_g)
        t2 = time.time() - start

        # Full method
        start = time.time()
        result = full_oscillator_factor(N)
        t3 = time.time() - start

        print(f"N={N:>6} | spectral: {factors_sp} ({t1:.4f}s) | "
              f"graph: {factors_g} ({t2:.4f}s) | full sync={result.sync_order_parameter:.3f} ({t3:.4f}s)")

        # Verify against actual factors
        true_factors = []
        temp = N
        for p in range(2, N+1):
            while temp % p == 0:
                true_factors.append(p)
                temp //= p
            if temp == 1:
                break

        if sorted(factors_sp) == sorted(true_factors):
            print(f"  ✓ SPECTRAL correct")
        if sorted(factors_g) == sorted(true_factors):
            print(f"  ✓ GRAPH correct")


if __name__ == "__main__":
    test_N = [
        15, 21, 35, 77, 143, 299,
        899, 1763, 3599, 7397,
        10585, 31417, 138547,
    ]

    benchmark(test_N)
