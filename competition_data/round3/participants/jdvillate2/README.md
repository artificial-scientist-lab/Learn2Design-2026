# jdvillate2 — Round 3 evaluation data

## Summary

- Place: 51 of 67
- Mean feasible loss: 0.544111059
- Sample standard deviation: 0.159893754
- Standard error of the mean: 0.050562845
- Median feasible loss: 0.554751143
- Minimum / maximum run score: 0.255882225 / 0.870890411
- Mean time to best: 163.89 minutes
- Mean evaluations: 56058.2
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
