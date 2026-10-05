# jeetu_sk — Round 3 evaluation data

## Summary

- Place: 23 of 67
- Mean feasible loss: 0.258092495
- Sample standard deviation: 0.192421783
- Standard error of the mean: 0.060849111
- Median feasible loss: 0.315621896
- Minimum / maximum run score: -0.184796178 / 0.424104125
- Mean time to best: 232.95 minutes
- Mean evaluations: 162746.8
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
