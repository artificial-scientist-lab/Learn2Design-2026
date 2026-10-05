# bgkang — Round 3 evaluation data

## Summary

- Place: 22 of 67
- Mean feasible loss: 0.237997956
- Sample standard deviation: 0.134021202
- Standard error of the mean: 0.042381225
- Median feasible loss: 0.272338343
- Minimum / maximum run score: -0.044880447 / 0.407614337
- Mean time to best: 196.93 minutes
- Mean evaluations: 90328.0
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
