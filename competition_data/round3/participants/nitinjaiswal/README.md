# nitinjaiswal — Round 3 evaluation data

## Summary

- Place: 28 of 67
- Mean feasible loss: 0.306610806
- Sample standard deviation: 0.105369658
- Standard error of the mean: 0.033320811
- Median feasible loss: 0.338938031
- Minimum / maximum run score: 0.113556256 / 0.421518136
- Mean time to best: 231.14 minutes
- Mean evaluations: 55667.7
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
