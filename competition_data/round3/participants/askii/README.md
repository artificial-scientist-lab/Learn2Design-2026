# askii — Round 3 evaluation data

## Summary

- Place: 14 of 67
- Mean feasible loss: 0.136870376
- Sample standard deviation: 0.212436413
- Standard error of the mean: 0.067178292
- Median feasible loss: 0.159527925
- Minimum / maximum run score: -0.265945148 / 0.422506990
- Mean time to best: 197.83 minutes
- Mean evaluations: 139139.2
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
