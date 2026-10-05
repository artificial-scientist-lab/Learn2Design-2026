# seament — Round 3 evaluation data

## Summary

- Place: 3 of 67
- Mean feasible loss: -0.238182698
- Sample standard deviation: 0.225790863
- Standard error of the mean: 0.071401340
- Median feasible loss: -0.258764835
- Minimum / maximum run score: -0.688119534 / 0.132239985
- Mean time to best: 177.23 minutes
- Mean evaluations: 116205.3
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
