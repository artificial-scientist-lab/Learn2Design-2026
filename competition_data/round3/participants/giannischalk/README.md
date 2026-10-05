# giannischalk — Round 3 evaluation data

## Summary

- Place: 2 of 67
- Mean feasible loss: -0.443577549
- Sample standard deviation: 0.052111354
- Standard error of the mean: 0.016479057
- Median feasible loss: -0.451859065
- Minimum / maximum run score: -0.510124127 / -0.325293296
- Mean time to best: 225.15 minutes
- Mean evaluations: 137708.0
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
