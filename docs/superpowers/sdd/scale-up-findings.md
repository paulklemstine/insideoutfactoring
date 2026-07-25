# Inside-Out Factoring: Scale-Up Findings

Comprehensive benchmark measuring the Inside-Out factoring algorithm's
performance across bit sizes 16-128 and factor gap categories.

## Methodology

- **Algorithm**: Inside-Out factoring with multiple strategies tested individually
- **Strategies**: perfect_square detection, CF convergent precheck (cf_precheck),
  CF-steered best-first search, BFS fallback, wavefront search, trial_division
- **Timeout**: 10 seconds per factorization (hard subprocess kill)
- **Iteration limits**: tiered by bit size (50K-500K for steered/BFS, 500-5K for wavefront radius)
- **Factor gap categories**:
  - Perfect square: p = q (N = p^2)
  - Close factors: q/p < 1.1
  - Moderate gap: 1.1 <= q/p < 2
  - Wide gap: q/p >= 2

## Results by Bit Size and Factor Gap

| Bits | Category | Tests | OK | Avg ms | Med ms | Dominant Strategy | Timeouts |
|------|----------|-------|----|--------|--------|-------------------|----------|
| 16 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 16 | close | 5 | 5 | 4.9 | 4.5 | steered | 0/5 |
| 16 | moderate | 5 | 5 | 38.3 | 38.0 | steered | 0/5 |
| 16 | wide | 5 | 5 | 0.6 | 0.8 | steered | 0/5 |
| 24 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 24 | close | 5 | 5 | 78.4 | 71.4 | steered | 0/5 |
| 24 | moderate | 5 | 5 | 118.1 | 117.3 | steered | 0/5 |
| 24 | wide | 5 | 5 | 0.8 | 0.7 | steered | 0/5 |
| 32 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 32 | close | 5 | 5 | 1564.9 | 1888.6 | steered | 0/5 |
| 32 | moderate | 5 | 5 | 2005.5 | 1968.8 | steered | 0/5 |
| 32 | wide | 5 | 5 | 1.0 | 0.9 | steered | 0/5 |
| 40 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 40 | close | 5 | 0 | - | - | timeout | 5/5 |
| 40 | moderate | 5 | 0 | - | - | timeout | 5/5 |
| 40 | wide | 5 | 5 | 1.5 | 1.6 | steered | 0/5 |
| 48 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 48 | close | 5 | 0 | - | - | timeout | 5/5 |
| 48 | moderate | 5 | 0 | - | - | timeout | 5/5 |
| 48 | wide | 5 | 5 | 4.5 | 3.8 | steered | 0/5 |
| 56 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 56 | close | 5 | 1 | 0.1 | 0.1 | timeout | 4/5 |
| 56 | moderate | 5 | 0 | - | - | timeout | 5/5 |
| 56 | wide | 5 | 5 | 10.2 | 10.7 | steered | 0/5 |
| 64 | perfect_square | 5 | 5 | 0.0 | 0.0 | perfect_square | 0/5 |
| 64 | close | 5 | 0 | - | - | timeout | 5/5 |
| 64 | moderate | 5 | 0 | - | - | timeout | 5/5 |
| 64 | wide | 5 | 5 | 16.6 | 16.0 | steered | 0/5 |
| 96 | perfect_square | 3 | 3 | 0.0 | 0.0 | perfect_square | 0/3 |
| 96 | close | 3 | 0 | - | - | timeout | 3/3 |
| 96 | moderate | 3 | 0 | - | - | timeout | 3/3 |
| 96 | wide | 3 | 3 | 1928.5 | 1921.7 | steered | 0/3 |
| 128 | perfect_square | 3 | 3 | 0.0 | 0.0 | perfect_square | 0/3 |
| 128 | close | 3 | 0 | - | - | timeout | 3/3 |
| 128 | moderate | 3 | 0 | - | - | timeout | 3/3 |
| 128 | wide | 3 | 0 | - | - | timeout | 3/3 |

## Strategy Dominance by Bit Size

| Bits | perfect_sq | cf_precheck | steered | bfs | wavefront | trial_div | timeout/fail |
|------|-----------|-------------|---------|-----|-----------|-----------|-------------|
| 16 | 5 | 2 | 13 | 0 | 0 | 0 | 0 |
| 24 | 5 | 0 | 15 | 0 | 0 | 0 | 0 |
| 32 | 5 | 1 | 14 | 0 | 0 | 0 | 0 |
| 40 | 5 | 0 | 5 | 0 | 0 | 0 | 10 |
| 48 | 5 | 0 | 5 | 0 | 0 | 0 | 10 |
| 56 | 5 | 1 | 5 | 0 | 0 | 0 | 9 |
| 64 | 5 | 0 | 5 | 0 | 0 | 0 | 10 |
| 96 | 3 | 0 | 3 | 0 | 0 | 0 | 6 |
| 128 | 3 | 0 | 0 | 0 | 0 | 0 | 9 |

## Success Rate by Bit Size

| Bits | Total | OK | Fail | Timeout | Rate |
|------|-------|----|------|---------|------|
| 16 | 20 | 20 | 0 | 0 | 100% |
| 24 | 20 | 20 | 0 | 0 | 100% |
| 32 | 20 | 20 | 0 | 0 | 100% |
| 40 | 20 | 10 | 0 | 10 | 50% |
| 48 | 20 | 10 | 0 | 10 | 50% |
| 56 | 20 | 11 | 0 | 9 | 55% |
| 64 | 20 | 10 | 0 | 10 | 50% |
| 96 | 12 | 6 | 0 | 6 | 50% |
| 128 | 12 | 3 | 0 | 9 | 25% |

## Success Rate by Category and Bit Size

| Bits | perfect_sq | close | moderate | wide |
|------|-----------|-------|----------|------|
| 16 | 5/5 | 5/5 | 5/5 | 5/5 |
| 24 | 5/5 | 5/5 | 5/5 | 5/5 |
| 32 | 5/5 | 5/5 | 5/5 | 5/5 |
| 40 | 5/5 | 0/5 | 0/5 | 5/5 |
| 48 | 5/5 | 0/5 | 0/5 | 5/5 |
| 56 | 5/5 | 1/5 | 0/5 | 5/5 |
| 64 | 5/5 | 0/5 | 0/5 | 5/5 |
| 96 | 3/3 | 0/3 | 0/3 | 3/3 |
| 128 | 3/3 | 0/3 | 0/3 | 0/3 |

## Median Time (ms) by Bit Size and Category (successful only)

| Bits | perfect_sq | close | moderate | wide |
|------|-----------|-------|----------|------|
| 16 | 0.0 | 4.5 | 38.0 | 0.8 |
| 24 | 0.0 | 71.4 | 117.3 | 0.7 |
| 32 | 0.0 | 1888.6 | 1968.8 | 0.9 |
| 40 | 0.0 | - | - | 1.6 |
| 48 | 0.0 | - | - | 3.8 |
| 56 | 0.0 | 0.1 | - | 10.7 |
| 64 | 0.0 | - | - | 16.0 |
| 96 | 0.0 | - | - | 1921.7 |
| 128 | 0.0 | - | - | - |

## Detailed Results

| Bits | Cat | # | Ratio | Strategy | ms | Status |
|------|-----|---|-------|----------|----|--------|
| 16 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 16 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 16 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 16 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 16 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 16 | close | 1 | 1.02 | steered | 2.8 | OK |
| 16 | close | 2 | 1.02 | steered | 4.5 | OK |
| 16 | close | 3 | 1.01 | cf_precheck | 0.1 | OK |
| 16 | close | 4 | 1.01 | steered | 9.4 | OK |
| 16 | close | 5 | 1.02 | steered | 7.6 | OK |
| 16 | moder | 1 | 1.10 | steered | 2.8 | OK |
| 16 | moder | 2 | 1.13 | steered | 17.4 | OK |
| 16 | moder | 3 | 1.14 | steered | 38.0 | OK |
| 16 | moder | 4 | 1.14 | steered | 60.4 | OK |
| 16 | moder | 5 | 1.14 | steered | 72.9 | OK |
| 16 | wide | 1 | 241.12 | steered | 0.8 | OK |
| 16 | wide | 2 | 141.76 | steered | 0.9 | OK |
| 16 | wide | 3 | 133.13 | steered | 0.6 | OK |
| 16 | wide | 4 | 100.95 | steered | 0.8 | OK |
| 16 | wide | 5 | 88.36 | cf_precheck | 0.1 | OK |
| 24 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 24 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 24 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 24 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 24 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 24 | close | 1 | 1.00 | steered | 64.9 | OK |
| 24 | close | 2 | 1.00 | steered | 71.4 | OK |
| 24 | close | 3 | 1.01 | steered | 69.2 | OK |
| 24 | close | 4 | 1.00 | steered | 93.4 | OK |
| 24 | close | 5 | 1.00 | steered | 93.2 | OK |
| 24 | moder | 1 | 1.10 | steered | 68.6 | OK |
| 24 | moder | 2 | 1.11 | steered | 92.2 | OK |
| 24 | moder | 3 | 1.11 | steered | 117.3 | OK |
| 24 | moder | 4 | 1.12 | steered | 162.0 | OK |
| 24 | moder | 5 | 1.12 | steered | 150.3 | OK |
| 24 | wide | 1 | 61681.35 | steered | 0.6 | OK |
| 24 | wide | 2 | 36158.66 | steered | 0.7 | OK |
| 24 | wide | 3 | 33826.10 | steered | 0.7 | OK |
| 24 | wide | 4 | 25576.27 | steered | 0.9 | OK |
| 24 | wide | 5 | 22311.34 | steered | 0.9 | OK |
| 32 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 32 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 32 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 32 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 32 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 32 | close | 1 | 1.00 | steered | 1823.2 | OK |
| 32 | close | 2 | 1.00 | cf_precheck | 0.1 | OK |
| 32 | close | 3 | 1.00 | steered | 1953.6 | OK |
| 32 | close | 4 | 1.00 | steered | 1888.6 | OK |
| 32 | close | 5 | 1.00 | steered | 2158.8 | OK |
| 32 | moder | 1 | 1.10 | steered | 1965.9 | OK |
| 32 | moder | 2 | 1.10 | steered | 1968.8 | OK |
| 32 | moder | 3 | 1.10 | steered | 1940.7 | OK |
| 32 | moder | 4 | 1.10 | steered | 2074.2 | OK |
| 32 | moder | 5 | 1.10 | steered | 2077.7 | OK |
| 32 | wide | 1 | 3627506.95 | steered | 0.9 | OK |
| 32 | wide | 2 | 3273603.83 | steered | 0.9 | OK |
| 32 | wide | 3 | 2855696.96 | steered | 0.9 | OK |
| 32 | wide | 4 | 2274877.51 | steered | 1.0 | OK |
| 32 | wide | 5 | 2200291.49 | steered | 1.2 | OK |
| 40 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 40 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 40 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 40 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 40 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 40 | close | 1 | 1.00 | timeout | 10000.0 | TMO |
| 40 | close | 2 | 1.00 | timeout | 10000.0 | TMO |
| 40 | close | 3 | 1.00 | timeout | 10000.0 | TMO |
| 40 | close | 4 | 1.00 | timeout | 10000.0 | TMO |
| 40 | close | 5 | 1.00 | timeout | 10000.0 | TMO |
| 40 | moder | 1 | 1.10 | timeout | 10000.0 | TMO |
| 40 | moder | 2 | 1.10 | timeout | 10000.0 | TMO |
| 40 | moder | 3 | 1.10 | timeout | 10000.0 | TMO |
| 40 | moder | 4 | 1.10 | timeout | 10000.0 | TMO |
| 40 | moder | 5 | 1.10 | timeout | 10000.0 | TMO |
| 40 | wide | 1 | 256415958.34 | steered | 1.4 | OK |
| 40 | wide | 2 | 235340674.10 | steered | 1.6 | OK |
| 40 | wide | 3 | 217466699.53 | steered | 1.4 | OK |
| 40 | wide | 4 | 193032238.91 | steered | 1.8 | OK |
| 40 | wide | 5 | 177112054.26 | steered | 1.6 | OK |
| 48 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 48 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 48 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 48 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 48 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 48 | close | 1 | 1.00 | timeout | 10000.0 | TMO |
| 48 | close | 2 | 1.00 | timeout | 10000.0 | TMO |
| 48 | close | 3 | 1.00 | timeout | 10000.0 | TMO |
| 48 | close | 4 | 1.00 | timeout | 10000.0 | TMO |
| 48 | close | 5 | 1.00 | timeout | 10000.0 | TMO |
| 48 | moder | 1 | 1.10 | timeout | 10000.0 | TMO |
| 48 | moder | 2 | 1.10 | timeout | 10000.0 | TMO |
| 48 | moder | 3 | 1.10 | timeout | 10000.0 | TMO |
| 48 | moder | 4 | 1.10 | timeout | 10000.0 | TMO |
| 48 | moder | 5 | 1.10 | timeout | 10000.0 | TMO |
| 48 | wide | 1 | 4278255361.05 | steered | 3.5 | OK |
| 48 | wide | 2 | 4087403820.78 | steered | 6.0 | OK |
| 48 | wide | 3 | 4057238478.98 | steered | 3.8 | OK |
| 48 | wide | 4 | 3912852768.08 | steered | 3.8 | OK |
| 48 | wide | 5 | 3752599412.39 | steered | 5.2 | OK |
| 56 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 56 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 56 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 56 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 56 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 56 | close | 1 | 1.00 | cf_precheck | 0.1 | OK |
| 56 | close | 2 | 1.00 | timeout | 10000.0 | TMO |
| 56 | close | 3 | 1.00 | timeout | 10000.0 | TMO |
| 56 | close | 4 | 1.00 | timeout | 10000.0 | TMO |
| 56 | close | 5 | 1.00 | timeout | 10000.0 | TMO |
| 56 | moder | 1 | 1.10 | timeout | 10000.0 | TMO |
| 56 | moder | 2 | 1.10 | timeout | 10000.0 | TMO |
| 56 | moder | 3 | 1.10 | timeout | 10000.0 | TMO |
| 56 | moder | 4 | 1.10 | timeout | 10000.0 | TMO |
| 56 | moder | 5 | 1.10 | timeout | 10000.0 | TMO |
| 56 | wide | 1 | 270129536190.66 | steered | 8.4 | OK |
| 56 | wide | 2 | 270129536190.73 | steered | 8.7 | OK |
| 56 | wide | 3 | 260143231710.48 | steered | 10.7 | OK |
| 56 | wide | 4 | 260143231710.48 | steered | 12.4 | OK |
| 56 | wide | 5 | 260143231710.52 | steered | 10.9 | OK |
| 64 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 64 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 64 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 64 | perfe | 4 | 1.00 | perfect_square | 0.0 | OK |
| 64 | perfe | 5 | 1.00 | perfect_square | 0.0 | OK |
| 64 | close | 1 | 1.00 | timeout | 10000.0 | TMO |
| 64 | close | 2 | 1.00 | timeout | 10000.0 | TMO |
| 64 | close | 3 | 1.00 | timeout | 10000.0 | TMO |
| 64 | close | 4 | 1.00 | timeout | 10000.0 | TMO |
| 64 | close | 5 | 1.00 | timeout | 10000.0 | TMO |
| 64 | moder | 1 | 1.10 | timeout | 10000.0 | TMO |
| 64 | moder | 2 | 1.10 | timeout | 10000.0 | TMO |
| 64 | moder | 3 | 1.10 | timeout | 10000.0 | TMO |
| 64 | moder | 4 | 1.10 | timeout | 10000.0 | TMO |
| 64 | moder | 5 | 1.10 | timeout | 10000.0 | TMO |
| 64 | wide | 1 | 17472743462155.33 | steered | 17.7 | OK |
| 64 | wide | 2 | 17438914336381.55 | steered | 15.3 | OK |
| 64 | wide | 3 | 17338208382562.22 | steered | 16.0 | OK |
| 64 | wide | 4 | 17172925175864.77 | steered | 18.5 | OK |
| 64 | wide | 5 | 16978697935421.44 | steered | 15.4 | OK |
| 96 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 96 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 96 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 96 | close | 1 | 1.00 | timeout | 10000.0 | TMO |
| 96 | close | 2 | 1.00 | timeout | 10000.0 | TMO |
| 96 | close | 3 | 1.00 | timeout | 10000.0 | TMO |
| 96 | moder | 1 | 1.10 | timeout | 10000.0 | TMO |
| 96 | moder | 2 | 1.10 | timeout | 10000.0 | TMO |
| 96 | moder | 3 | 1.10 | timeout | 10000.0 | TMO |
| 96 | wide | 1 | 18446462603027742720.00 | steered | 1974.2 | OK |
| 96 | wide | 2 | 18442522915205400576.00 | steered | 1889.5 | OK |
| 96 | wide | 3 | 18442522915205400576.00 | steered | 1921.7 | OK |
| 128 | perfe | 1 | 1.00 | perfect_square | 0.0 | OK |
| 128 | perfe | 2 | 1.00 | perfect_square | 0.0 | OK |
| 128 | perfe | 3 | 1.00 | perfect_square | 0.0 | OK |
| 128 | close | 1 | 1.00 | timeout | 10000.0 | TMO |
| 128 | close | 2 | 1.00 | timeout | 10000.0 | TMO |
| 128 | close | 3 | 1.00 | timeout | 10000.0 | TMO |
| 128 | moder | 1 | 1.10 | timeout | 10000.0 | TMO |
| 128 | moder | 2 | 1.10 | timeout | 10000.0 | TMO |
| 128 | moder | 3 | 1.10 | timeout | 10000.0 | TMO |
| 128 | wide | 1 | 77370625271121859873603584.00 | timeout | 10000.0 | TMO |
| 128 | wide | 2 | 77370625271121859873603584.00 | timeout | 10000.0 | TMO |
| 128 | wide | 3 | 77370625271121859873603584.00 | timeout | 10000.0 | TMO |

## Analysis

### Failure Boundary

First bit size with failures: **40-bit**
- 16-bit: 100% success rate
- 24-bit: 100% success rate
- 32-bit: 100% success rate
- 40-bit: 50% success rate
- 48-bit: 50% success rate
- 56-bit: 55% success rate
- 64-bit: 50% success rate
- 96-bit: 50% success rate
- 128-bit: 25% success rate

### Category-Specific Failure Points

- **perfect_square**: No failures in tested range
- **close**: Failures at [40, 48, 56, 64, 96, 128]
- **moderate**: Failures at [40, 48, 56, 64, 96, 128]
- **wide**: Failures at [128]

## Recommendations

### Dominant Strategies by Size Range

**16-32 bit**:
- steered: 42
- perfect_square: 15
- cf_precheck: 3

**40-64 bit**:
- perfect_square: 20
- steered: 20
- cf_precheck: 1

**96-128 bit**:
- perfect_square: 6
- steered: 3

### Timeout Analysis

Timeouts first appear at **40-bit**.

- close: 25 timeouts at [40, 48, 56, 64, 96, 128]
- moderate: 26 timeouts at [40, 48, 56, 64, 96, 128]
- wide: 3 timeouts at [128]

### Scaling Improvement Priorities

1. **Adaptive iteration budgets**: Scale iteration limits with N's bit length
2. **Extend CF convergent terms**: Increase max_terms for larger N to improve cf_precheck effectiveness
3. **Wavefront radius scaling**: Increase max_radius proportionally to bit size
4. **Add Pollard's rho**: As fallback for medium-factor semiprimes where steered search struggles
5. **Quadratic sieve**: Consider for 64+ bit sizes as the primary fallback
6. **Close-factor specialization**: The steered search is most effective for close factors; wider gaps need different strategies

---

*Generated by scale_up_benchmark.py*