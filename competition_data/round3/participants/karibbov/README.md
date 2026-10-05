# karibbov — Round 3 evaluation data

## Summary

- Place: 37 of 67
- Mean feasible loss: 0.361540535
- Sample standard deviation: 0.133346409
- Standard error of the mean: 0.042167837
- Median feasible loss: 0.424342631
- Minimum / maximum run score: 0.135319153 / 0.519824217
- Mean time to best: 164.19 minutes
- Mean evaluations: 162131.9
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
