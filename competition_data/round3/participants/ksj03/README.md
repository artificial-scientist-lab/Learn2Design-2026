# ksj03 — Round 3 evaluation data

## Summary

- Place: 30 of 67
- Mean feasible loss: 0.311884740
- Sample standard deviation: 0.163038199
- Standard error of the mean: 0.051557205
- Median feasible loss: 0.394489380
- Minimum / maximum run score: -0.036602418 / 0.473427793
- Mean time to best: 239.64 minutes
- Mean evaluations: 134020.4
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoints are conservative upper bounds unless marked exact.
The 240-minute values are official run scores; missing values and fallbacks
are marked.
