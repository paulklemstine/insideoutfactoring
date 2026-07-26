# Oscillator Factoring: What We Learned

## Direct Extraction Fails (As Expected)
Building J from factor candidates creates circular reasoning:
- J_ij = f(i*j mod N) where f peaks near multiples of p ∈ factors(N)
- But we're trying to FIND factors using J's spectrum
- The spectrum encodes what we put into J, not new information

## What DOES Work
**Smoothness Detection**: The oscillator network CAN distinguish:
- N with small factors → high sync order (resonance effects build up)
- N with large prime factors → low sync order (no coherent resonance)

This is useful for cascade gating, not direct extraction.

## Why Dynamics Might Reveal Factors (Theoretical)
The hope was that:
1. Coupling J built from i*j mod N creates "resonance modes" at factor-related frequencies
2. When oscillator i couples strongly to oscillator j, it means i*j mod N ≈ k*p for factor p
3. The network naturally partitions into clusters aligned with factors
4. Phase synchronization in different clusters reveals factor structure

**Why this is weak**:
- For composite N, the modular arithmetic i*j mod N is pseudorandom
- The "signal" from factors is drowned in noise from all other products
- Large prime factors produce almost no detectable signal

## Alternative Approaches to Explore

### 1. Iterative Enhancement
- Start with weak factor hints from resonance
- Use found factors to REBUILD J with stronger coupling
- Iterate until convergence

### 2. Forced Sync in Subgroups
- Select oscillator subset based on index structure
- Force that subgroup to synchronize
- Measure "pull" on remaining oscillators
- The pull strength indicates factor relationship

### 3. Temperature/Bifurcation Analysis
- Run network at various "temperatures" (noise levels)
- Track sync transitions
- Factor structure creates distinctive phase diagrams

### 4. Coupled Oscillator + GCD
- Use oscillator to find candidate smooth pairs (i, j) where i*j ≈ N
- Apply batch GCD to extract actual factors

## Key Insight: The N-dependence is Real
The sync order genuinely depends on N's value (smaller N with small factors sync better).
This means N's structure IS embedded in the dynamics.
The problem is extracting the factor information, not detecting its presence.

## Practical Verdict
- **NOT a standalone factoring algorithm** for large N
- **USEFUL as smoothness detector** in adaptive cascade
- Could be combined with SQUFOF or ECM as pre-screening
