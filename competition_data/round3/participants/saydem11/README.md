# saydem11 — Round 3 evaluation data

## Summary

- Place: 39 of 67
- Mean feasible loss: 0.368833044
- Sample standard deviation: 0.095369133
- Standard error of the mean: 0.030158368
- Median feasible loss: 0.401659192
- Minimum / maximum run score: 0.204690605 / 0.514830308
- Mean time to best: 181.71 minutes
- Mean evaluations: 160872.6
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
