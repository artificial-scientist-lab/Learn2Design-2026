# jackqiu — Round 3 evaluation data

## Summary

- Place: 16 of 67
- Mean feasible loss: 0.176782614
- Sample standard deviation: 0.109841993
- Standard error of the mean: 0.034735088
- Median feasible loss: 0.155009730
- Minimum / maximum run score: 0.014577647 / 0.352208440
- Mean time to best: 239.75 minutes
- Mean evaluations: 163088.2
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
