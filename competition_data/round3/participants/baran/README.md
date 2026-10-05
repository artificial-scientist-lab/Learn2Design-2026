# baran — Round 3 evaluation data

## Summary

- Place: 52 of 67
- Mean feasible loss: 0.564034945
- Sample standard deviation: 0.152062149
- Standard error of the mean: 0.048086274
- Median feasible loss: 0.523400420
- Minimum / maximum run score: 0.413286829 / 0.885774159
- Mean time to best: 237.94 minutes
- Mean evaluations: 113174.8
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
