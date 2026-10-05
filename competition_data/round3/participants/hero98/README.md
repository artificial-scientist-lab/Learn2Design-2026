# hero98 — Round 3 evaluation data

## Summary

- Place: 33 of 67
- Mean feasible loss: 0.324581518
- Sample standard deviation: 0.226075898
- Standard error of the mean: 0.071491476
- Median feasible loss: 0.401600192
- Minimum / maximum run score: -0.309227068 / 0.450399519
- Mean time to best: 238.71 minutes
- Mean evaluations: 145952.8
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
