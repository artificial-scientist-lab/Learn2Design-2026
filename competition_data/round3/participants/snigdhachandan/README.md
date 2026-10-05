# snigdhachandan — Round 3 evaluation data

## Summary

- Place: 61 of 67
- Mean feasible loss: 1.238099508
- Sample standard deviation: 0.728724763
- Standard error of the mean: 0.230443004
- Median feasible loss: 0.954436474
- Minimum / maximum run score: 0.427953261 / 2.167103752
- Mean time to best: 239.37 minutes
- Mean evaluations: 165622.8
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
