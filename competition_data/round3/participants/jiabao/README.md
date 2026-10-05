# jiabao — Round 3 evaluation data

## Summary

- Place: 59 of 67
- Mean feasible loss: 0.992508119
- Sample standard deviation: 1.260436919
- Standard error of the mean: 0.398585151
- Median feasible loss: 0.489498003
- Minimum / maximum run score: 0.344577517 / 4.484783246
- Mean time to best: 225.48 minutes
- Mean evaluations: 1.0
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
