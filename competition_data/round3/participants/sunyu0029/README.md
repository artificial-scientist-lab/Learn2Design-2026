# sunyu0029 — Round 3 evaluation data

## Summary

- Place: 55 of 67
- Mean feasible loss: 0.965870058
- Sample standard deviation: 0.733650271
- Standard error of the mean: 0.232000586
- Median feasible loss: 0.715726101
- Minimum / maximum run score: 0.306478548 / 2.730345271
- Mean time to best: 239.69 minutes
- Mean evaluations: 107091.5
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
