# matevz — Round 2 evaluation data

## Summary

- Place: 12 of 60
- Mean feasible loss: 0.215952097
- Sample standard deviation: 0.216962507
- Standard error of the mean: 0.068609569
- Median feasible loss: 0.270340445
- Minimum / maximum run score: -0.261112446 / 0.424704337
- Mean time to best: 225.51 minutes
- Mean evaluations: 138025.6
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoint values are marked as exact when the final best was
already known to have been reached. Otherwise they are sampled upper bounds.
The 240-minute values are the exact official run scores.
