# talmansour — Round 3 evaluation data

## Summary

- Place: 25 of 67
- Mean feasible loss: 0.285609917
- Sample standard deviation: 0.164990762
- Standard error of the mean: 0.052174660
- Median feasible loss: 0.343049943
- Minimum / maximum run score: -0.092868011 / 0.453914594
- Mean time to best: 184.27 minutes
- Mean evaluations: 160909.0
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
