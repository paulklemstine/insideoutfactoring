# Inside-Out Factoring Project

## Purpose
Research project exploring a novel geometric and spectral framework for integer factorization of large semiprimes. The approach embeds the factorization problem into Pythagorean tree topology via Berggren's matrix transformations, translating divisor search into a spectral resonance problem.

## Tech Stack
- Pure Python (math/algorithm implementation)
- No framework dependencies beyond stdlib
- Markdown for papers/documentation

## Current State
- Early research phase
- Single file: `paper.md` - the foundational paper on the theory
- No code implementation yet

## Key Concepts
1. **Pythagorean Tree**: Berggren's U, A, D matrices generate all PPTs from root (3,4,5)
2. **Energy Spectrum**: E(v) = ln(c) metric on tree nodes; factors manifest as degenerate energy states
3. **Inside-Out Algorithm**: Start at Central Approximation Well (p=q=√N), search radially outward
4. **Convergence**: O(ln²N) matrix operations when |p-q| = O(N^{1/4})

## Commands
- `python3` for running scripts
- `python3 -m pytest` for testing (once tests exist)