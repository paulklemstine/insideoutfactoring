# Bidirectional Cofactor Inside-Out (Mathematically Precise)

## Algorithm
Bidirectional meet-in-the-middle using Berggren tree and coordinate-zero covectors:

1. **Forward frontier**: From thin seed v0=(0,-1,1) mod N, ascend using branch matrices
2. **Backward frontier**: From goal covectors {e_a, e_b, e_c}, apply transposed pullbacks
3. **Meet check**: gcd(<ell, v> mod N, N) — exact factor certificate
4. **Replay**: Concatenate prefix+suffix words, replay from seed

## Transposed Pullback Formulas (mod N)
```
U: (x+2y+2z, -2x-y-2z, 2x+2y+3z)
A: (x+2y+2z,  2x+y+2z, 2x+2y+3z)
D: (-x-2y-2z,  2x+y+2z, 2x+2y+3z)
```

## Verified Results
| N | Factors | Depths | Result |
|---|---------|--------|--------|
| 8051 | 83×97 | (2,2) | ✓ |
| 10403 | 101×103 | (3,3) | ✓ |
| 100160063 | 10007×10009 | (4,4) | ✓ |

## Key Insight (from user's research)
Goal nodes are COVECTORS (e_a, e_b, e_c), not guessed triples. Pulling backward via C^T gives a meet without full tree construction. The unknown prime makes the goal hyperplane invisible until GCD detects it.

## File
- `insideout/inside_out.py` — `inside_out_factor_bidirectional()`
