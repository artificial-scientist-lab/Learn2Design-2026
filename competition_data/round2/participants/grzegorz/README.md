# grzegorz — Round 2 evaluation data

## Summary

- Place: 24 of 60
- Mean feasible loss: 0.357378340
- Sample standard deviation: 0.111299053
- Standard error of the mean: 0.035195851
- Median feasible loss: 0.365017449
- Minimum / maximum run score: 0.174451958 / 0.520838366
- Mean time to best: 199.35 minutes
- Mean evaluations: 50325.1
- Failed runs: 0
- Random-search fallback runs: 0

## Files

- `summary.json`: machine-readable aggregate statistics.
- `runs.csv`: exact final outcome and efficiency statistics for all ten runs.
- `checkpoints.csv`: per-run best feasible loss at selected elapsed times.
- `convergence.csv`: mean and uncertainty across runs at those checkpoints.
- `convergence.png`: visual summary of the convergence data.

Intermediate checkpoint values are marked as exact when the final best was
already known to have been reached. Otherwise they are sampled upper bounds
derived from one stored candidate every 10 evaluations. The 240-minute values
are the exact official run scores.
