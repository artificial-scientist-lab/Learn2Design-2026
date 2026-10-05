# jnapoles — Round 3 evaluation data

## Summary

- Place: 49 of 67
- Mean feasible loss: 0.502500508
- Sample standard deviation: 0.120323722
- Standard error of the mean: 0.038049702
- Median feasible loss: 0.483857896
- Minimum / maximum run score: 0.308490211 / 0.656899682
- Mean time to best: 171.80 minutes
- Mean evaluations: 56777.3
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
