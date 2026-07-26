# MCMC Factoring Algorithm — Research Notes

## Core Insight
Pollard's rho x_{n+1} = x_n^2 + c (mod N) IS a deterministic Markov chain on Z/NZ.
When N = pq, CRT decomposes this into independent chains on Z/pZ and Z/qZ.
The chain mixes at DIFFERENT rates on each component — this is the conductance bottleneck exploited by resonance_cascade.

## Key Questions
1. Can we design a stochastic chain (Metropolis-Hastings) with better mixing?
2. What is the stationary distribution that reveals factors?
3. Can absorption probabilities in factor cosets be computed?

## Theoretical Foundation

### Pollard's rho as Markov chain
- State space: Z/NZ
- Transition: x' = f(x) = x^2 + c (mod N)
- For N = pq, f decomposes via CRT into f_p on Z/pZ and f_q on Z/qZ
- Collision x_i ≡ x_j (mod p) but x_i ≠ x_j (mod N) → p | gcd(x_i - x_j, N)

### Stationary Distribution Approach
Consider stationary distribution π(x) ∝ gcd(x, N)^α:
- π(x) = gcd(x, N)^α / Σ_{y∈Z/NZ} gcd(y, N)^α
- This concentrates on states with nontrivial gcd structure
- Problem: computing π requires knowing factors — circular

### Metropolis-Hastings Design
Proposal: x' = x + u (mod N) where u is uniform in small step set S
Acceptance: min(1, π(x')/π(x) * Q(x|x')/Q(x'|x))
For symmetric proposal: min(1, gcd(x', N)/gcd(x, N))

### Key Insight: gcd(x - 1, N) Probe
Instead of looking for collisions, check gcd(x_n - 1, N) at each step.
When chain state satisfies x_n ≡ 1 (mod p) but x_n ≢ 1 (mod q),
gcd(x_n - 1, N) = p — factor revealed!

The CRT bottleneck: h(N) ≤ min(h(p), h(q))
The chain gets trapped at the slower component (larger of p, q).

## Algorithm Sketch

```python
def mcmc_factor(N, num_chains=8, max_iter=100000):
    for each chain with independent seed and constant c:
        x = random state
        for iter in range(max_iter):
            # MCMC step (optional — plain rho often sufficient)
            x_proposed = (x*x + c) % N  # Pollard step
            accept with probability min(1, gcd(x_proposed - 1, N) / max(gcd(x - 1, N), 1))
            
            # Check factor
            g = gcd(x - 1, N)
            if 1 < g < N: return (g, N//g)
            
            # Floyd cycle detection
            # ...
```

## Why MCMC Might Beat rho
1. **Random restart**: MCMC chains with different seeds explore the state space more uniformly
2. **Multiple independent trajectories**: 8 chains = 8x chance of hitting a factor-revealing state
3. **Adaptive proposals**: If chain gets stuck in a cycle mod p, random perturbation may escape
4. **Convergence diagnostics**: Can monitor autocorrelation time to detect factor structure

## Theoretical Limits
- Expected time: O(sqrt(p)) for smaller factor p — same as rho
- Quadratic speedup over trial division is the fundamental limit
- No polynomial-time classical factoring (would break RSA)

## Connection to Codebase
- resonance_cascade.py Stage 4 and 5 implement conductance + rho
- projective_collision.py has related collision detection
- adaptive_portfolio.py orchestrates multiple methods
