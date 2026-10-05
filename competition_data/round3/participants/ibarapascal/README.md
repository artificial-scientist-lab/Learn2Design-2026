# ibarapascal — Round 3 evaluation data

## Summary

- Place: 20 of 67
- Mean feasible loss: 0.222324622
- Sample standard deviation: 0.118390577
- Standard error of the mean: 0.037438388
- Median feasible loss: 0.232317802
- Minimum / maximum run score: -0.021437503 / 0.408573184
- Mean time to best: 237.09 minutes
- Mean evaluations: 153008.0
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
