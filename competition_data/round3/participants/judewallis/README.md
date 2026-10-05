# judewallis — Round 3 evaluation data

## Summary

- Place: 1 of 67
- Mean feasible loss: -0.540301049
- Sample standard deviation: 0.109963765
- Standard error of the mean: 0.034773596
- Median feasible loss: -0.562463345
- Minimum / maximum run score: -0.680910303 / -0.313371349
- Mean time to best: 237.33 minutes
- Mean evaluations: 1957.6
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
