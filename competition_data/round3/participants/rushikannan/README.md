# rushikannan — Round 3 evaluation data

## Summary

- Place: 60 of 67
- Mean feasible loss: 1.169502191
- Sample standard deviation: 1.328878866
- Standard error of the mean: 0.420228395
- Median feasible loss: 0.573191208
- Minimum / maximum run score: 0.287032765 / 4.323107964
- Mean time to best: 144.70 minutes
- Mean evaluations: 53164.0
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
