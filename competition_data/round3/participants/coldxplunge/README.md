# coldxplunge — Round 3 evaluation data

## Summary

- Place: 41 of 67
- Mean feasible loss: 0.387077638
- Sample standard deviation: 0.107291304
- Standard error of the mean: 0.033928490
- Median feasible loss: 0.411917197
- Minimum / maximum run score: 0.116966133 / 0.521598553
- Mean time to best: 216.25 minutes
- Mean evaluations: 132583.6
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
