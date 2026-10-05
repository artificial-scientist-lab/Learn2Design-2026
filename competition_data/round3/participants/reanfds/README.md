# reanfds — Round 3 evaluation data

## Summary

- Place: 56 of 67
- Mean feasible loss: 0.969814649
- Sample standard deviation: 1.057787420
- Standard error of the mean: 0.334501753
- Median feasible loss: 0.583015549
- Minimum / maximum run score: 0.429373798 / 3.907612879
- Mean time to best: 148.98 minutes
- Mean evaluations: 108505.9
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
