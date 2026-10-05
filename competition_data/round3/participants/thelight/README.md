# thelight — Round 3 evaluation data

## Summary

- Place: 64 of 67
- Mean feasible loss: 1.539726362
- Sample standard deviation: 1.334317787
- Standard error of the mean: 0.421948333
- Median feasible loss: 0.994153105
- Minimum / maximum run score: 0.420028177 / 3.806316602
- Mean time to best: 177.52 minutes
- Mean evaluations: 56310.8
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
