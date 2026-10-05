# kevingill — Round 3 evaluation data

## Summary

- Place: 62 of 67
- Mean feasible loss: 1.262669459
- Sample standard deviation: 1.096640358
- Standard error of the mean: 0.346788130
- Median feasible loss: 0.719829718
- Minimum / maximum run score: 0.162518011 / 2.841020618
- Mean time to best: 226.79 minutes
- Mean evaluations: 56192.7
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
