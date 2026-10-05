# miscusi — Round 3 evaluation data

## Summary

- Place: 15 of 67
- Mean feasible loss: 0.151528639
- Sample standard deviation: 0.179393181
- Standard error of the mean: 0.056729105
- Median feasible loss: 0.190840511
- Minimum / maximum run score: -0.170256245 / 0.353912169
- Mean time to best: 236.66 minutes
- Mean evaluations: 159107.4
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
